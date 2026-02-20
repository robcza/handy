import os
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False

SMOOTHING_ALPHA = 0.3
PINCH_THRESHOLD = 40
CLICK_COOLDOWN = 0.5
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# ── MediaPipe Tasks API setup ────────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
draw_landmarks = mp.tasks.vision.drawing_utils.draw_landmarks
HandConnections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
)

# ── Screen / webcam setup ────────────────────────────────────────────────────
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(
        "ERROR: Cannot access the camera.\n"
        "On macOS, go to System Settings > Privacy & Security > Camera\n"
        "and grant access to Terminal (or whichever app you're running this from).\n"
        "Then re-run the script."
    )
    raise SystemExit(1)

prev_x, prev_y = 0.0, 0.0
last_right_click_time = 0.0
is_dragging = False
frame_ts = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33  # ~30 fps in milliseconds
        results = landmarker.detect_for_video(mp_image, frame_ts)

        if results.hand_landmarks:
            lm = results.hand_landmarks[0]
            draw_landmarks(frame, lm, HandConnections)

            idx_tip = lm[8]
            thb_tip = lm[4]

            raw_x = idx_tip.x * frame_w
            raw_y = idx_tip.y * frame_h

            # EMA smoothing
            smooth_x = prev_x + SMOOTHING_ALPHA * (raw_x - prev_x)
            smooth_y = prev_y + SMOOTHING_ALPHA * (raw_y - prev_y)
            prev_x, prev_y = smooth_x, smooth_y

            # Map frame coordinates to screen coordinates
            scr_x = np.interp(smooth_x, [0, frame_w], [0, screen_w])
            scr_y = np.interp(smooth_y, [0, frame_h], [0, screen_h])
            pyautogui.moveTo(scr_x, scr_y, _pause=False)

            # Thumb pixel coords (shared by drag + right-click)
            tx, ty = thb_tip.x * frame_w, thb_tip.y * frame_h

            # Drag / left-click: index finger + thumb pinch
            dist = np.hypot(raw_x - tx, raw_y - ty)
            if dist < PINCH_THRESHOLD:
                if not is_dragging:
                    pyautogui.mouseDown(_pause=False)
                    is_dragging = True
            else:
                if is_dragging:
                    pyautogui.mouseUp(_pause=False)
                    is_dragging = False

            # Right-click: middle finger + thumb pinch
            mid_tip = lm[12]
            mx, my = mid_tip.x * frame_w, mid_tip.y * frame_h
            mid_dist = np.hypot(mx - tx, my - ty)
            if mid_dist < PINCH_THRESHOLD and time.time() - last_right_click_time > CLICK_COOLDOWN:
                pyautogui.rightClick(_pause=False)
                last_right_click_time = time.time()

        else:
            if is_dragging:
                pyautogui.mouseUp(_pause=False)
                is_dragging = False

        cv2.imshow("Handy", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
