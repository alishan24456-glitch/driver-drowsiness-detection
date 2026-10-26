# Driver Drowsiness Detector

A live driver-monitoring system: point it at a driver's face and it
tracks **eye closure** and **yawning** in real time, counts blinks and
yawns, and raises a flashing **DROWSY alarm** when the eyes stay shut
beyond a short threshold. This is the core of the driver-monitoring
systems used in fleet safety, long-haul trucking and driver-assistance
features.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-real--mode-orange)

## How drowsiness is judged

1. **Landmarks** — every frame produces face landmarks through one
   interface: MediaPipe Face Mesh in real mode, the simulation scene in
   demo mode. Each set has six points per eye and four mouth points.
2. **EAR (Eye Aspect Ratio)** — for each eye, the vertical eyelid
   distances divided by twice the eye width, averaged over both eyes.
   Below 0.22 the eyes count as closed.
3. **MAR (Mouth Aspect Ratio)** — vertical lip distance divided by the
   mouth-corner distance. Above 0.30 the mouth counts as yawning open.
4. **State machine** — a short closure that ends before 30 consecutive
   frames counts as a **blink**; a closure reaching 30 frames (~1 second
   at 30 fps) triggers a **drowsy event** and a flashing DROWSY alarm
   that only clears after 10 consecutive open-eye frames; a mouth held
   wide for 15 consecutive frames counts as a **yawn** (re-armed only
   after the mouth closes). A HUD shows the live status, EAR/MAR values,
   blink, yawn and drowsy-event counters, and the longest eye closure.
5. **No-face handling** — when no face is visible the analyser holds its
   previous verdict instead of inventing one.

## Two modes

**Real mode** (`--source`) runs MediaPipe Face Mesh on a video file or
webcam and feeds the standard 468-point landmarks (eyes 33/160/158/133/
153/144 and 362/385/387/263/373/380; mouth corners 61/291, lips 13/14)
into the same EAR/MAR analyser. Any camera angle that keeps the face
front-facing and well lit works best.

**Simulation mode** (`--simulate`) renders a car interior with OpenCV —
windscreen with a moving road, dashboard, instrument cluster, steering
wheel — and animates a driver who blinks, yawns twice and twice lets
his eyes close for several seconds. The scene publishes only landmark
coordinates through the *same interface* MediaPipe uses, and publishes
no event labels: every blink, yawn and DROWSY alarm in the demo is
derived by the analyser from EAR and MAR alone, exactly as on real
footage. The drawn face matches the landmark geometry (an eye drawn
shut has near-zero eyelid distance), so what you see is what is
measured. Simulation needs only `opencv-python` and `numpy` — no model
download, no GPU.

## Tech stack

- **Python 3.9+**
- **OpenCV** — rendering, overlay, video I/O
- **NumPy** — landmark geometry
- **MediaPipe** — real-mode face landmarks only

## Install

```bash
git clone https://github.com/alishan24456-glitch/driver-drowsiness-detection.git
cd driver-drowsiness-detection
pip install -r requirements.txt
```

## Usage

Simulation, live preview window:

```bash
python driver_drowsiness_detector.py --simulate
```

Simulation, rendered to a video file (headless-friendly):

```bash
python driver_drowsiness_detector.py --simulate --output demo.mp4 --no-display --duration 28
```

Real footage or webcam:

```bash
python driver_drowsiness_detector.py --source driver_cam.mp4
python driver_drowsiness_detector.py --source driver_cam.mp4 --output annotated.mp4
python driver_drowsiness_detector.py --source 0                       # webcam
```

Useful options: `--drowsy-frames 45` (longer closure before the alarm),
`--yawn-frames 20` (stricter yawn rule), `--seed 7` (simulation
randomness). Press `q` or `Esc` to stop a preview. A run ends with a
summary line, e.g. `[summary] blinks=4 yawns=2 drowsy_events=2
longest_eye_closure=2.4s frames_seen=840 final_status=ALERT`.

## Demo

The demo video for this project (`driver-drowsiness-detection-demo.mp4`)
is the **simulation mode rendered through the actual drowsiness
pipeline** — the driver is drawn by the scene, but every EAR/MAR value,
status, counter and alarm banner is produced by the same analyser that
grades real footage. It is a simulation, not real driving footage. In
the 28-second demo, 4 blinks are counted, 2 yawns are counted, 2 drowsy
events trigger the alarm, and the longest eye closure is 2.4 seconds,
ending with the driver alert again.

## Real-world uses

- Fleet and long-haul driver safety: alert a driver (and dispatcher)
  before a microsleep becomes a crash
- Taxi / rideshare driver monitoring on a dashboard camera
- Driver-assistance research: blink-rate and yawn-rate baselines per trip
- Training material: annotated examples of alert vs drowsy behaviour

## Honest limitations

- **This is a landmark-ratio heuristic, not a trained drowsiness
  classifier.** It measures eyelid and lip geometry only. It does not
  know head pose, gaze direction or PERCLOS over minutes, and no
  accuracy figures are claimed here; they would have to be measured on
  specific driver footage. Nothing here replaces a certified driver
  monitoring system or a rested driver.
- Real-mode quality depends on a clear, front-facing, well-lit view:
  sunglasses, a hand over the face, strong side light, profile angles
  and low resolution all degrade the landmarks.
- Thresholds (EAR 0.22, 30 closed frames, MAR 0.30, 15 yawn frames) are
  sensible defaults, not personalised ones: blink and yawn behaviour
  varies between people, so tune `--drowsy-frames` / `--yawn-frames`
  for the driver and camera in use.
- The simulation's face is always front-facing by design; real driving
  includes head turns, which is a further reason the heuristic is a
  demo-grade monitor rather than a safety-certified system.

## Future improvements

- **PERCLOS + head-pose tracking**: percentage of eyelid closure over a
  rolling minute plus pitch/yaw nodding detection
- **Personalised calibration**: a short alert-baseline capture that sets
  per-driver EAR thresholds
- **Alert output**: audible alarm, seat vibration trigger, or fleet
  dispatch webhook when an alarm persists
- **Gaze tracking**: off-road glance duration as an additional
  distraction signal
- **Night mode**: infrared-camera landmark support for low-light driving

## Author

**Abdul Hanan** — Computer Vision & AI Developer
GitHub: [alishan24456-glitch](https://github.com/alishan24456-glitch)
LinkedIn: https://www.linkedin.com/in/ali-raza-computervision/
