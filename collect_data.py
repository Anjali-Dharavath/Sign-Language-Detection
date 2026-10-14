"""
Collect hand gesture images for training using MediaPipe Hand Landmarker.
Allows recording existing gestures or easily adding new words/gestures.

Controls:
    1-9     : Select predefined gesture
    N       : Type a NEW custom word/gesture
    SPACE   : Start / Pause recording frames
    S       : Stop recording current gesture
    Q       : Quit
"""

import os
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"

HAND_PADDING = 30
FRAMES_PER_GESTURE = 250

# Predefined gesture map
GESTURES = {
    "1": "hello",
    "2": "yes",
    "3": "no",
    "4": "thanks",
    "5": "iloveyou",
    "6": "please",
    "7": "sorry",
    "8": "help",
    "9": "goodbye",
}

# MediaPipe Hand Connections (Pairs of landmark indices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17)                                    # Palm base
]


def draw_hand_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    
    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, coords[start_idx], coords[end_idx], (0, 255, 180), 2)
        
    # Draw landmark joints
    for x, y in coords:
        cv2.circle(frame, (x, y), 4, (0, 120, 255), -1)
        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)


def main():
    if not MODEL_PATH.exists():
        print(f"Error: hand_landmarker.task not found at {MODEL_PATH}")
        return

    # Initialize Hand Landmarker
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Ensure dataset dirs exist
    for split in ["train", "val"]:
        for gesture in GESTURES.values():
            (DATASET_ROOT / split / gesture).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam. Please check connection.")
        return

    recording = False
    current_gesture = "hello"
    frame_count = len(list((DATASET_ROOT / "train" / current_gesture).glob("*.jpg"))) + \
                  len(list((DATASET_ROOT / "val" / current_gesture).glob("*.jpg")))

    print("\n========================================================")
    print("         SIGN LANGUAGE DATA COLLECTOR                   ")
    print("========================================================")
    print(" 1-9    : Choose preset gesture (hello, yes, no, etc.)")
    print(" N      : Type a NEW custom word/gesture")
    print(" SPACE  : START / PAUSE recording frames")
    print(" S      : STOP recording")
    print(" Q      : QUIT and finish")
    print("========================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = detector.detect(mp_image)

        hand_crop = None
        hand_detected = False

        if detection_result.hand_landmarks:
            hand_detected = True
            landmarks = detection_result.hand_landmarks[0]
            draw_hand_landmarks(frame, landmarks)

            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]

            x1 = max(0, min(xs) - HAND_PADDING)
            y1 = max(0, min(ys) - HAND_PADDING)
            x2 = min(w, max(xs) + HAND_PADDING)
            y2 = min(h, max(ys) + HAND_PADDING)

            if x2 > x1 and y2 > y1:
                hand_crop = frame[y1:y2, x1:x2].copy()
                # Draw hand box
                box_color = (0, 255, 0) if recording else (255, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Save frame if recording and hand is present
        if recording and hand_crop is not None and hand_crop.size > 0:
            # 80% train / 20% val split
            split = "train" if (frame_count % 5 != 0) else "val"
            save_dir = DATASET_ROOT / split / current_gesture
            save_dir.mkdir(parents=True, exist_ok=True)
            
            filename = save_dir / f"{frame_count:05d}.jpg"
            cv2.imwrite(str(filename), hand_crop)
            frame_count += 1

        # HUD Top Banner
        hud_bg = frame.copy()
        cv2.rectangle(hud_bg, (0, 0), (w, 85), (20, 20, 20), -1)
        cv2.addWeighted(hud_bg, 0.75, frame, 0.25, 0, frame)

        status_text = f"RECORDING: {current_gesture.upper()} [{frame_count} frames]" if recording \
                      else f"READY: {current_gesture.upper()} [{frame_count} frames]"
        status_color = (0, 255, 0) if recording else (0, 200, 255)
        cv2.putText(frame, status_text, (15, 32), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)

        hand_status = "Hand: DETECTED" if hand_detected else "Hand: NOT DETECTED (Show hand)"
        hand_color = (0, 255, 120) if hand_detected else (0, 0, 255)
        cv2.putText(frame, hand_status, (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hand_color, 1)

        # Bottom Hints
        cv2.putText(frame, "SPACE: Rec/Pause | S: Stop | 1-9: Presets | N: New Word | Q: Quit",
                    (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        cv2.imshow("Sign Language - Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break
        elif key == 32:  # SPACE
            if hand_detected or recording:
                recording = not recording
                if recording:
                    print(f"\n[+] Recording started for '{current_gesture}'...")
                else:
                    print(f"[-] Recording paused for '{current_gesture}'. Total frames: {frame_count}")
        elif key == ord("s") or key == ord("S"):
            recording = False
            print(f"[-] Stopped recording '{current_gesture}'. Saved {frame_count} frames.")
        elif chr(key) in GESTURES:
            recording = False
            current_gesture = GESTURES[chr(key)]
            frame_count = len(list((DATASET_ROOT / "train" / current_gesture).glob("*.jpg"))) + \
                          len(list((DATASET_ROOT / "val" / current_gesture).glob("*.jpg")))
            print(f"\n[>] Selected gesture: '{current_gesture}' (existing frames: {frame_count})")
        elif key == ord("n") or key == ord("N"):
            recording = False
            print("\nEnter new word/gesture name (letters and numbers only): ", end="", flush=True)
            try:
                new_word = input().strip().lower().replace(" ", "_")
                if new_word:
                    current_gesture = new_word
                    (DATASET_ROOT / "train" / current_gesture).mkdir(parents=True, exist_ok=True)
                    (DATASET_ROOT / "val" / current_gesture).mkdir(parents=True, exist_ok=True)
                    frame_count = len(list((DATASET_ROOT / "train" / current_gesture).glob("*.jpg"))) + \
                                  len(list((DATASET_ROOT / "val" / current_gesture).glob("*.jpg")))
                    print(f"[+] Custom gesture '{current_gesture}' set! Press SPACE to start recording.")
            except Exception as e:
                print(f"Error reading input: {e}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection finished.")
    print("To train the model on your dataset, run: python train.py")


if __name__ == "__main__":
    main()
