# Body-as-Brush

Contactless drawing with a webcam: **MediaPipe Pose** + **Hands**, **OpenCV**. Pinch with your right hand to draw; raise your left arm to change color; bring wrists together to clear the canvas. Press **q** to quit.

**Gestures:** pinch (right hand) — draw · left arm up — cycle color · wrists close — clear canvas · **squat** — toggle body skeleton overlay on/off.

## Requirements

| | |
|---|---|
| **Python** | 3.9+ (3.11 works) |
| **Hardware** | Webcam |
| **Packages** | Install from [`requirements.txt`](requirements.txt) only (`pip install -r requirements.txt`). No other installs or manual downloads. |

Python packages (same as `requirements.txt`):

- `opencv-python`
- `mediapipe==0.10.14` — pinned so `mp.solutions` (Pose / Hands) matches this code
- `numpy`

## Setup

**1.** Open a terminal and go to the project folder (the one that contains `requirements.txt`):

```bash
cd /path/to/Body-as-Brush
```

**2.** Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\activate` instead of `source .venv/bin/activate`. If `python3` is not found, try `python`.

## Run

Stay in that **same** folder (project root) and run:

```bash
python src/body_as_brush.py
```

If you see `Error: Cannot open webcam`, plug in or enable the camera and allow access in system privacy settings. Grant camera access when the OS prompts.

## Layout

```
Body-as-Brush/
├── requirements.txt
└── src/
    └── body_as_brush.py
```
