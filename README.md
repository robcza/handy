# Handy

Cursor 15 minutes minihackathon for computer control without mouse and keyboard.

Uses your MacBook's built-in webcam and MediaPipe hand tracking to control the system cursor with hand gestures.

## Gestures

| Gesture | Action |
|---|---|
| Point index finger | Move cursor |
| Pinch index + thumb (quick) | Left click |
| Pinch index + thumb (hold) | Drag and drop |
| Pinch middle finger + thumb | Right click |
| Raise middle finger only | Quit |

## Setup

```bash
pip install -r requirements.txt
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
python main.py
```

Press **q** in the preview window or raise your middle finger to quit.
