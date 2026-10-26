#!/usr/bin/env python3
"""
Driver Drowsiness Detector
==========================

Watches a driver's face and raises an alarm when drowsiness signs appear:

* **Eye Aspect Ratio (EAR)** -- how open the eyes are. Eyes held shut
  beyond a short threshold mean the driver is not just blinking.
* **Mouth Aspect Ratio (MAR)** -- how open the mouth is. A wide, sustained
  mouth opening counts as a yawn.
* **Alarm state** -- a long eye closure triggers a DROWSY alarm (flashing
  banner + alert counter) that only clears once the eyes stay open again.

One shared drowsiness pipeline serves both run modes:

* **Real mode** (``--source``) -- MediaPipe Face Mesh tracks the face in a
  video file or webcam feed and supplies eye/mouth landmarks.
* **Simulation mode** (``--simulate``) -- a driver is rendered with
  OpenCV (car interior, road through the windscreen, steering wheel) and
  the scene publishes eye/mouth landmarks through exactly the same
  landmark interface MediaPipe uses.

How drowsiness is judged -- and what it is NOT
-----------------------------------------------
The analyser never sees the simulation's choreography labels (it does not
know "a yawn is scheduled now" or "this closure is a drowsy event"). It
receives only landmark coordinates -- the same coordinates MediaPipe
would produce -- and derives everything from EAR and MAR:

* EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|) per eye, averaged over both
  eyes, where p1..p6 are the six eye-corner/eyelid points.
* MAR = vertical lip distance / horizontal mouth-corner distance.
* A **blink** is a short closure: EAR below threshold, then the eyes
  reopen before the drowsy frame limit.
* A **drowsy event** is a closure lasting at least ``--drowsy-frames``
  consecutive frames (default 30, i.e. ~1 second at 30 fps).
* A **yawn** is MAR above threshold for at least ``--yawn-frames``
  consecutive frames (default 15), re-armed only after the mouth closes.

The rendered face in the simulation is drawn from the same eye/mouth
geometry the landmarks describe (an eye drawn shut has near-zero eyelid
distance), so what the viewer sees matches what the analyser measures.
This is a landmark-ratio heuristic, not a trained drowsiness classifier:
it is honest about that in the README, and real-mode reliability depends
on a clear, front-facing, well-lit view of the face.

Examples
--------
    python driver_drowsiness_detector.py --simulate
    python driver_drowsiness_detector.py --simulate --output demo.mp4 --no-display
    python driver_drowsiness_detector.py --source driver_cam.mp4
    python driver_drowsiness_detector.py --source 0

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_WIDTH, SIM_HEIGHT = 1280, 720   # simulation canvas
SIM_FPS = 30                        # simulation frame rate

EAR_THRESHOLD = 0.22    # below this the eyes count as closed
MAR_THRESHOLD = 0.30    # above this the mouth counts as yawning open

COLOUR_ALERT = (0, 190, 70)         # green (BGR)
COLOUR_WARN = (0, 200, 230)         # amber (BGR)
COLOUR_ALARM = (0, 55, 230)         # red (BGR)
COLOUR_TEXT = (255, 255, 255)
COLOUR_LANDMARK = (0, 255, 255)     # yellow landmark dots

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Landmarks and the landmark-provider interface
# ---------------------------------------------------------------------------

@dataclass
class FaceLandmarks:
    """Six points per eye plus four mouth points, in image coordinates.

    Eye point order (per eye): p1 left corner, p2 upper-left, p3
    upper-right, p4 right corner, p5 lower-right, p6 lower-left.
    Mouth points: left corner, right corner, upper-lip centre,
    lower-lip centre.
    """

    left_eye: list[tuple[float, float]]    # 6 points
    right_eye: list[tuple[float, float]]   # 6 points
    mouth_left: tuple[float, float]
    mouth_right: tuple[float, float]
    mouth_upper: tuple[float, float]
    mouth_lower: tuple[float, float]

    @property
    def all_points(self) -> list[tuple[float, float]]:
        return (self.left_eye + self.right_eye
                + [self.mouth_left, self.mouth_right,
                   self.mouth_upper, self.mouth_lower])

    @property
    def face_box(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.all_points]
        ys = [p[1] for p in self.all_points]
        pad_x, pad_y = 90, 110
        return (max(0, int(min(xs) - pad_x)), max(0, int(min(ys) - pad_y)),
                int(max(xs) + pad_x), int(max(ys) + pad_y))


class LandmarkProvider:
    """Interface every landmark source implements: frame in, landmarks out."""

    def landmarks(self, frame: np.ndarray) -> FaceLandmarks | None:
        raise NotImplementedError


class MediaPipeLandmarkProvider(LandmarkProvider):
    """Real-mode provider backed by MediaPipe Face Mesh (lazily imported).

    Eye indices follow the standard 468-point Face Mesh layout:
    left eye  [33, 160, 158, 133, 153, 144], right eye
    [362, 385, 387, 263, 373, 380]; mouth corners 61 / 291, upper lip
    13, lower lip 14.
    """

    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    def __init__(self):
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "Real mode needs the 'mediapipe' package for Face Mesh.\n"
                "Install it with:  pip install mediapipe\n"
                "(Simulation mode --simulate does not need it.)"
            ) from exc
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def landmarks(self, frame: np.ndarray) -> FaceLandmarks | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        def pt(idx: int) -> tuple[float, float]:
            return (lm[idx].x * w, lm[idx].y * h)

        return FaceLandmarks(
            left_eye=[pt(i) for i in self.LEFT_EYE_IDX],
            right_eye=[pt(i) for i in self.RIGHT_EYE_IDX],
            mouth_left=pt(61), mouth_right=pt(291),
            mouth_upper=pt(13), mouth_lower=pt(14))


class SimulatedLandmarkProvider(LandmarkProvider):
    """Simulation-mode provider: forwards the scene's face landmarks.

    The scene publishes only landmark coordinates -- the same shape
    MediaPipe produces. It publishes NO event labels: the analyser cannot
    tell a blink from a drowsy closure except through EAR, and cannot
    tell a yawn except through MAR.
    """

    def __init__(self, scene: "DriverScene"):
        self.scene = scene

    def landmarks(self, frame: np.ndarray) -> FaceLandmarks | None:  # unused
        return self.scene.face_landmarks()


# ---------------------------------------------------------------------------
# The shared EAR / MAR drowsiness analyser
# ---------------------------------------------------------------------------

def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(eye: list[tuple[float, float]]) -> float:
    """Standard 6-point EAR: vertical eyelid distances over eye width."""
    p1, p2, p3, p4, p5, p6 = eye
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * max(1e-6, _dist(p1, p4)))


def mouth_aspect_ratio(lm: FaceLandmarks) -> float:
    """Vertical lip distance over horizontal mouth-corner distance."""
    vertical = _dist(lm.mouth_upper, lm.mouth_lower)
    horizontal = _dist(lm.mouth_left, lm.mouth_right)
    return vertical / max(1e-6, horizontal)


@dataclass
class DrowsinessState:
    """Everything the analyser knows, derived only from EAR / MAR."""

    ear: float = 1.0
    mar: float = 0.0
    eyes_closed: bool = False
    status: str = "ALERT"           # ALERT | BLINKING | YAWNING | DROWSY
    alarm: bool = False
    blinks: int = 0
    yawns: int = 0
    drowsy_events: int = 0
    closed_frames: int = 0
    longest_closure_frames: int = 0
    frames_seen: int = 0


class DrowsinessAnalyser:
    """EAR/MAR state machine shared by both modes."""

    def __init__(self, drowsy_frames: int = 30, yawn_frames: int = 15,
                 ear_threshold: float = EAR_THRESHOLD,
                 mar_threshold: float = MAR_THRESHOLD):
        self.drowsy_frames = max(1, drowsy_frames)
        self.yawn_frames = max(1, yawn_frames)
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.state = DrowsinessState()
        self._yawn_streak = 0
        self._yawn_armed = True
        self._open_streak = 0   # consecutive open-eye frames (alarm clear)

    def process(self, lm: FaceLandmarks | None) -> DrowsinessState:
        s = self.state
        s.frames_seen += 1
        if lm is None:
            # No face visible: hold the previous verdict, do not invent one.
            s.status = "NO FACE" if s.frames_seen > 1 else "ALERT"
            return s

        s.ear = (eye_aspect_ratio(lm.left_eye)
                 + eye_aspect_ratio(lm.right_eye)) / 2.0
        s.mar = mouth_aspect_ratio(lm)
        s.eyes_closed = s.ear < self.ear_threshold

        # --- eye closure bookkeeping -----------------------------------
        if s.eyes_closed:
            s.closed_frames += 1
            s.longest_closure_frames = max(s.longest_closure_frames,
                                           s.closed_frames)
            self._open_streak = 0
            if s.closed_frames >= self.drowsy_frames and not s.alarm:
                s.alarm = True
                s.drowsy_events += 1
                s.status = "DROWSY"
        else:
            if 0 < s.closed_frames < self.drowsy_frames:
                s.blinks += 1          # a short closure that just ended
            s.closed_frames = 0
            self._open_streak += 1
            if s.alarm and self._open_streak >= 10:
                s.alarm = False        # eyes reliably open again
            if not s.alarm:
                s.status = "ALERT"

        # --- yawn bookkeeping ------------------------------------------
        if s.mar > self.mar_threshold:
            self._yawn_streak += 1
            if self._yawn_streak >= self.yawn_frames and self._yawn_armed:
                s.yawns += 1
                self._yawn_armed = False
            if not s.alarm:
                s.status = "YAWNING"
        else:
            self._yawn_streak = 0
            self._yawn_armed = True
            if not s.alarm and not s.eyes_closed:
                s.status = "ALERT"

        if s.alarm:
            s.status = "DROWSY"
        elif s.eyes_closed and s.closed_frames > 0:
            s.status = "BLINKING"
        return s

    def summary_line(self) -> str:
        s = self.state
        longest_s = s.longest_closure_frames / SIM_FPS
        return (f"[summary] blinks={s.blinks} yawns={s.yawns} "
                f"drowsy_events={s.drowsy_events} "
                f"longest_eye_closure={longest_s:.1f}s "
                f"frames_seen={s.frames_seen} final_status={s.status}")


# ---------------------------------------------------------------------------
# Drawing the annotated view (shared by both modes)
# ---------------------------------------------------------------------------

def draw_annotation(frame: np.ndarray, lm: FaceLandmarks | None,
                    s: DrowsinessState, fps: float) -> np.ndarray:
    out = frame.copy()
    if lm is not None:
        x1, y1, x2, y2 = lm.face_box
        box_colour = COLOUR_ALARM if s.alarm else COLOUR_ALERT
        cv2.rectangle(out, (x1, y1), (x2, y2), box_colour, 2)
        for eye in (lm.left_eye, lm.right_eye):
            pts = np.array([(int(x), int(y)) for x, y in eye], dtype=np.int32)
            cv2.polylines(out, [pts], True, COLOUR_LANDMARK, 1,
                          cv2.LINE_AA)
        for p in lm.all_points:
            cv2.circle(out, (int(p[0]), int(p[1])), 2, COLOUR_LANDMARK, -1)

    # HUD panel.
    panel_w, panel_h = 560, 170
    overlay = out.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, out, 0.32, 0, out)
    cv2.rectangle(out, (8, 8), (8 + panel_w, 8 + panel_h), (90, 90, 90), 1)
    status_colour = COLOUR_ALARM if s.alarm else (
        COLOUR_WARN if s.status in ("YAWNING", "BLINKING") else COLOUR_ALERT)
    cv2.putText(out, f"STATUS: {s.status}", (20, 44), FONT, 0.8,
                status_colour, 2, cv2.LINE_AA)
    cv2.putText(out,
                f"EAR: {s.ear:.2f}   MAR: {s.mar:.2f}   "
                f"Closed: {s.closed_frames} frames",
                (20, 78), FONT, 0.6, COLOUR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(out,
                f"Blinks: {s.blinks}   Yawns: {s.yawns}   "
                f"Drowsy events: {s.drowsy_events}",
                (20, 110), FONT, 0.6, COLOUR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(out,
                f"Longest closure: {s.longest_closure_frames / SIM_FPS:.1f}s"
                f"    FPS: {fps:.1f}",
                (20, 142), FONT, 0.55, COLOUR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(out, "DRIVER DROWSINESS MONITOR -- EAR + MAR",
                (20, 168), FONT, 0.5, COLOUR_TEXT, 1, cv2.LINE_AA)

    if s.alarm:
        # Flashing alarm banner (alternates each ~0.5 s at 30 fps).
        if (s.frames_seen // 15) % 2 == 0:
            cv2.rectangle(out, (0, SIM_HEIGHT - 90), (SIM_WIDTH, SIM_HEIGHT),
                          COLOUR_ALARM, -1)
            cv2.putText(out, "!!  DROWSY DRIVER -- WAKE UP  !!",
                        (SIM_WIDTH // 2 - 330, SIM_HEIGHT - 32), FONT, 1.0,
                        COLOUR_TEXT, 3, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Simulation: a driver rendered with OpenCV
# ---------------------------------------------------------------------------

# Face geometry (image coordinates, driver centred right of frame).
HEAD_CX, HEAD_CY = 880, 330
EYE_DX, EYE_Y = 62, 305
EYE_WIDTH = 84.0
EYE_OPEN_HEIGHT = 30.0
MOUTH_CX, MOUTH_Y = 880, 415
MOUTH_WIDTH = 110.0
MOUTH_CLOSED_HEIGHT = 8.0
MOUTH_YAWN_HEIGHT = 62.0


def eye_points(cx: float, cy: float, openness: float
               ) -> list[tuple[float, float]]:
    """Six eye points from centre + openness (1.0 open, ~0.08 shut).

    The eyelid half-height scales with openness, so EAR tracks the drawn
    eye exactly: a drawn-shut eye yields a near-zero EAR, a drawn-open
    eye yields a healthy EAR, with no other information involved.
    """
    half_h = max(1.5, (EYE_OPEN_HEIGHT / 2.0) * openness)
    half_w = EYE_WIDTH / 2.0
    return [
        (cx - half_w, cy),                    # p1 left corner
        (cx - half_w * 0.5, cy - half_h),     # p2 upper-left
        (cx + half_w * 0.5, cy - half_h),     # p3 upper-right
        (cx + half_w, cy),                    # p4 right corner
        (cx + half_w * 0.5, cy + half_h),     # p5 lower-right
        (cx - half_w * 0.5, cy + half_h),     # p6 lower-left
    ]


def mouth_points(openness: float) -> tuple[tuple[float, float], ...]:
    """Mouth corners + lip centres from openness (0 closed, 1 full yawn)."""
    half_h = MOUTH_CLOSED_HEIGHT / 2.0 + (
        MOUTH_YAWN_HEIGHT - MOUTH_CLOSED_HEIGHT) / 2.0 * openness
    half_w = MOUTH_WIDTH / 2.0
    return ((MOUTH_CX - half_w, MOUTH_Y), (MOUTH_CX + half_w, MOUTH_Y),
            (MOUTH_CX, MOUTH_Y - half_h), (MOUTH_CX, MOUTH_Y + half_h))


def choreography(t: float) -> tuple[float, float]:
    """Eye openness and mouth openness at simulation time t (seconds).

    Scripted driving story (the analyser never sees this table -- it only
    ever sees the landmarks these values produce):

    * 0-5 s:   alert driving, two short blinks
    * 5-9 s:   alert, one blink
    * 9-12 s:  yawn (mouth opens wide, ~2.5 s)
    * 13-16 s: eyes droop, then a long closure (~2.5 s) -> DROWSY alarm
    * 17-19 s: alert recovery, one blink
    * 19-22 s: second yawn
    * 23-26 s: second long closure (~2.5 s) -> second DROWSY alarm
    * 26-28 s: alert recovery
    """
    eye, mouth = 1.0, 0.0

    def blink(centre: float, width: float = 0.18) -> float:
        """A short dip to (near) shut centred at `centre` seconds."""
        d = abs(t - centre)
        if d >= width:
            return 1.0
        return 0.08 + 0.92 * (d / width)

    eye = min(eye, blink(2.0), blink(4.2), blink(7.0), blink(18.0))

    # Yawns: smooth mouth opening, held, then closing.
    for start, dur in ((9.0, 3.0), (19.5, 2.5)):
        if start <= t < start + dur:
            phase = (t - start) / dur
            mouth = max(mouth, math.sin(math.pi * min(1.0, phase)))

    # Long closures: eyes ease shut, stay shut, then reopen.
    for start, dur in ((13.5, 3.0), (23.0, 3.0)):
        if start <= t < start + dur:
            phase = (t - start) / dur
            if phase < 0.25:      # easing shut
                eye = min(eye, 1.0 - 0.92 * (phase / 0.25))
            elif phase < 0.8:     # held shut
                eye = min(eye, 0.08)
            else:                 # reopening
                eye = min(eye, 0.08 + 0.92 * ((phase - 0.8) / 0.2))

    # Slight droop before each long closure (heavy eyelids).
    for start in (13.0, 22.5):
        if start <= t < start + 0.5:
            eye = min(eye, 0.75)
    return eye, mouth


class DriverScene:
    """The simulated car interior + driver, advancing with time."""

    def __init__(self, duration: float, seed: int = 42):
        self.duration = duration
        self.rng = random.Random(seed)
        self.time = 0.0
        self.road_offset = 0.0
        self.background = self._render_background()

    # -- background ----------------------------------------------------------

    @staticmethod
    def _render_background() -> np.ndarray:
        img = np.full((SIM_HEIGHT, SIM_WIDTH, 3), (60, 60, 60), np.uint8)
        # Windscreen: sky + road ahead (left two-thirds of the frame).
        cv2.rectangle(img, (0, 0), (760, SIM_HEIGHT), (168, 188, 205), -1)
        cv2.rectangle(img, (0, 300), (760, SIM_HEIGHT), (90, 90, 90), -1)
        # Roadside: trees and a distant car.
        for x in range(40, 720, 130):
            cv2.circle(img, (x, 280), 26, (40, 120, 60), -1)
            cv2.rectangle(img, (x - 4, 300), (x + 4, 340), (90, 60, 30), -1)
        cv2.rectangle(img, (300, 380), (430, 450), (40, 80, 160), -1)
        cv2.rectangle(img, (320, 355), (410, 385), (150, 200, 235), -1)
        cv2.circle(img, (330, 450), 14, (20, 20, 20), -1)
        cv2.circle(img, (400, 450), 14, (20, 20, 20), -1)
        # Dashboard strip across the bottom.
        cv2.rectangle(img, (0, 560), (SIM_WIDTH, SIM_HEIGHT), (45, 45, 45), -1)
        cv2.rectangle(img, (0, 560), (SIM_WIDTH, 575), (25, 25, 25), -1)
        # Instrument cluster + air vents.
        cv2.rectangle(img, (60, 600), (300, 680), (20, 20, 20), -1)
        cv2.circle(img, (130, 640), 26, (0, 60, 120), 2)
        cv2.circle(img, (230, 640), 26, (0, 60, 120), 2)
        for vx in (340, 470):
            cv2.rectangle(img, (vx, 600), (vx + 70, 640), (15, 15, 15), -1)
        # Steering wheel (left) and driver seat area (right).
        cv2.ellipse(img, (560, 620), (120, 95), 0, 0, 360, (30, 30, 30), 14)
        cv2.ellipse(img, (560, 620), (120, 95), 0, 0, 360, (70, 70, 70), 3)
        cv2.rectangle(img, (760, 120), (SIM_WIDTH, SIM_HEIGHT), (52, 48, 44), -1)
        # No title text here: the HUD panel covers the top-left corner,
        # and the HUD itself already labels the monitor.
        return img

    # -- behaviour -------------------------------------------------------------

    def step(self, dt: float) -> None:
        self.time += dt
        self.road_offset = (self.road_offset + 260.0 * dt) % 90.0

    def face_landmarks(self) -> FaceLandmarks:
        eye_open, mouth_open = choreography(self.time)
        # Gentle head bob so the face is not perfectly static.
        bob = 3.0 * math.sin(self.time * 1.4)
        left = eye_points(HEAD_CX - EYE_DX, EYE_Y + bob, eye_open)
        right = eye_points(HEAD_CX + EYE_DX, EYE_Y + bob, eye_open)
        ml, mr, mu, md = mouth_points(mouth_open)
        md = (md[0], md[1] + bob)
        mu = (mu[0], mu[1] + bob)
        ml = (ml[0], ml[1] + bob)
        mr = (mr[0], mr[1] + bob)
        return FaceLandmarks(left, right, ml, mr, mu, md)

    # -- frame -----------------------------------------------------------------

    def render(self) -> np.ndarray:
        img = self.background.copy()
        # Moving lane dashes (road scrolls towards the viewer).
        for i in range(6):
            y = 340 + i * 55 + int(self.road_offset * 0.55)
            if y > 560:
                continue
            dash_w = 6 + i * 3
            cv2.rectangle(img, (370 - dash_w // 2, y),
                          (370 + dash_w // 2, y + 18 + i * 4),
                          (240, 240, 220), -1)
        eye_open, mouth_open = choreography(self.time)
        bob = 3.0 * math.sin(self.time * 1.4)
        cx, cy = HEAD_CX, int(HEAD_CY + bob)

        # Neck + shoulders.
        cv2.rectangle(img, (cx - 55, cy + 95), (cx + 55, cy + 190),
                      (120, 170, 215), -1)
        cv2.ellipse(img, (cx, cy + 200), (150, 70), 0, 180, 360,
                    (40, 90, 160), -1)
        # Head.
        cv2.ellipse(img, (cx, cy), (115, 135), 0, 0, 360, (120, 170, 215), -1)
        # Hair cap.
        cv2.ellipse(img, (cx, cy - 25), (115, 95), 0, 180, 360, (45, 32, 22), -1)
        # Ears.
        cv2.ellipse(img, (cx - 115, cy + 5), (14, 26), 0, 0, 360,
                    (120, 170, 215), -1)
        cv2.ellipse(img, (cx + 115, cy + 5), (14, 26), 0, 0, 360,
                    (120, 170, 215), -1)
        # Eyebrows.
        for ex in (cx - EYE_DX, cx + EYE_DX):
            cv2.rectangle(img, (ex - 48, int(EYE_Y + bob) - 52),
                          (ex + 48, int(EYE_Y + bob) - 42), (45, 32, 22), -1)
        # Eyes: white sclera + iris when open, a shut line when closed.
        eye_h = max(2.0, (EYE_OPEN_HEIGHT / 2.0) * eye_open)
        for ex in (cx - EYE_DX, cx + EYE_DX):
            ey = int(EYE_Y + bob)
            if eye_open > 0.25:
                cv2.ellipse(img, (ex, ey), (42, int(eye_h)), 0, 0, 360,
                            (245, 245, 245), -1)
                cv2.circle(img, (ex, ey), max(6, int(eye_h * 0.55)),
                           (30, 60, 110), -1)
                cv2.circle(img, (ex, ey), max(3, int(eye_h * 0.25)),
                           (10, 10, 10), -1)
            else:
                cv2.line(img, (ex - 42, ey), (ex + 42, ey), (30, 30, 30), 4)
        # Nose (simple) + mouth: drawn height matches the landmark height.
        cv2.line(img, (cx, int(EYE_Y + bob) + 35), (cx - 10, MOUTH_Y - 35),
                 (170, 140, 115), 3)
        mouth_half_h = max(3.0, MOUTH_CLOSED_HEIGHT / 2.0 + (
            MOUTH_YAWN_HEIGHT - MOUTH_CLOSED_HEIGHT) / 2.0 * mouth_open)
        cv2.ellipse(img, (MOUTH_CX, int(MOUTH_Y + bob)),
                    (int(MOUTH_WIDTH / 2.0), int(mouth_half_h)), 0, 0, 360,
                    (90, 40, 60), -1)
        # Seatbelt strap across the shoulder.
        cv2.line(img, (cx - 130, cy + 150), (cx + 60, cy + 260),
                 (25, 25, 25), 12)
        return img


# ---------------------------------------------------------------------------
# Run loops
# ---------------------------------------------------------------------------

def safe_destroy_windows() -> None:
    """cv2.destroyAllWindows() crashes some headless OpenCV builds; guard it."""
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def make_writer(path: str, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"could not open output video for writing: {path}")
    return writer


def run_simulation(args: argparse.Namespace) -> int:
    scene = DriverScene(duration=args.duration, seed=args.seed)
    provider = SimulatedLandmarkProvider(scene)
    analyser = DrowsinessAnalyser(drowsy_frames=args.drowsy_frames,
                                  yawn_frames=args.yawn_frames)

    writer = None
    if args.output:
        writer = make_writer(args.output, SIM_FPS, SIM_WIDTH, SIM_HEIGHT)

    total_frames = int(args.duration * SIM_FPS)
    try:
        for frame_no in range(total_frames):
            scene.step(1.0 / SIM_FPS)
            frame = scene.render()
            lm = provider.landmarks(frame)
            state = analyser.process(lm)
            annotated = draw_annotation(frame, lm, state, SIM_FPS)
            if state.status == "DROWSY" and frame_no % SIM_FPS == 0:
                print(f"[event] t={scene.time:5.1f}s  DROWSY ALARM "
                      f"(closure {state.closed_frames} frames)")
            if state.status == "YAWNING" and frame_no % SIM_FPS == 0:
                print(f"[event] t={scene.time:5.1f}s  yawning "
                      f"(MAR {state.mar:.2f})")
            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("Driver Drowsiness Detector - simulation", annotated)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    print("[info] stopped by user")
                    break
            if frame_no % (SIM_FPS * 5) == 0:
                print(f"[info] t={scene.time:5.1f}s  status={state.status} "
                      f"EAR={state.ear:.2f} MAR={state.mar:.2f}")
    finally:
        if writer is not None:
            writer.release()
        if not args.no_display:
            safe_destroy_windows()

    print(analyser.summary_line())
    return 0


def run_real(args: argparse.Namespace) -> int:
    source: int | str = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"error: could not open source: {args.source}", file=sys.stderr)
        return 2

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 30.0

    try:
        provider: LandmarkProvider = MediaPipeLandmarkProvider()
    except RuntimeError as exc:
        cap.release()
        print(f"error: {exc}", file=sys.stderr)
        return 2

    analyser = DrowsinessAnalyser(drowsy_frames=args.drowsy_frames,
                                  yawn_frames=args.yawn_frames)
    writer = None
    if args.output:
        writer = make_writer(args.output, src_fps, width, height)

    ema_fps = float(src_fps)
    last = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                ema_fps = 0.9 * ema_fps + 0.1 * (1.0 / dt)

            lm = provider.landmarks(frame)
            state = analyser.process(lm)
            annotated = draw_annotation(frame, lm, state, ema_fps)
            if state.alarm:
                print(f"[event] DROWSY ALARM (closure "
                      f"{state.closed_frames} frames)")
            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("Driver Drowsiness Detector", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[info] stopped by user")
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            safe_destroy_windows()

    print(analyser.summary_line())
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Driver drowsiness detector: EAR + MAR monitoring with "
                    "a DROWSY alarm on long eye closures.",
        epilog="examples:\n"
               "  python driver_drowsiness_detector.py --simulate\n"
               "  python driver_drowsiness_detector.py --simulate --output "
               "demo.mp4 --no-display\n"
               "  python driver_drowsiness_detector.py --source driver_cam.mp4\n"
               "  python driver_drowsiness_detector.py --source 0\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_argument_group("mode (choose one)")
    mode.add_argument("--simulate", action="store_true",
                      help="run the built-in car-interior simulation "
                           "(OpenCV + NumPy only)")
    mode.add_argument("--source", metavar="SRC",
                      help="real mode: video file path or webcam index")

    parser.add_argument("--drowsy-frames", type=int, default=30,
                        help="consecutive closed-eye frames that trigger "
                             "the DROWSY alarm (default: 30, ~1 s at 30 fps)")
    parser.add_argument("--yawn-frames", type=int, default=15,
                        help="consecutive wide-mouth frames that count as "
                             "a yawn (default: 15)")
    parser.add_argument("--duration", type=float, default=28.0,
                        help="simulation length in seconds (default: 28)")
    parser.add_argument("--seed", type=int, default=42,
                        help="simulation random seed (default: 42)")
    parser.add_argument("--output", metavar="FILE",
                        help="save the annotated video to FILE")
    parser.add_argument("--no-display", action="store_true",
                        help="do not open a preview window (headless)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.simulate and args.source is not None:
        parser.error("choose only one mode: --simulate OR --source")
    if not args.simulate and args.source is None:
        parser.error("choose a mode: --simulate or --source <video file|webcam index>")
    if args.simulate and args.duration <= 0:
        parser.error("--duration must be positive")

    if args.source is not None and not str(args.source).isdigit():
        if not os.path.exists(args.source):
            parser.error(f"source file not found: {args.source}")

    if args.simulate:
        return run_simulation(args)
    return run_real(args)


if __name__ == "__main__":
    sys.exit(main())
