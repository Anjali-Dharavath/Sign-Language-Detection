"""
Real-time Sign Language Detection: MediaPipe HandLandmarker + YOLOv8 Classifier.
Translates live hand gestures into written text with confidence tracking and sentence construction.

Controls:
    Q         : Quit
    C         : Clear sentence
    Backspace : Delete last word from sentence
    H         : Toggle skeleton landmark overlay
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# -----------------------------
# PATH CONFIGURATION
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Check for model weights
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "best.pt",
    PROJECT_ROOT.parent / "runs" / "classify" / "train6" / "weights" / "best.pt",
    PROJECT_ROOT.parent / "runs" / "classify" / "train4" / "weights" / "best.pt",
]

YOLO_PATH = None
for p in MODEL_CANDIDATES:
    if p.exists():
        YOLO_PATH = p
        break

LANDMARKER_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
if not LANDMARKER_PATH.exists():
    LANDMARKER_PATH = PROJECT_ROOT / "hand_landmarker.task"

# -----------------------------
# ACCURACY & SMOOTHING SETTINGS
# -----------------------------
SMOOTH_WINDOW = 8            # Rolling average frame window
CONSECUTIVE_FRAMES = 3       # Same gesture must lead for N frames
DISPLAY_CONFIDENCE = 0.65    # Minimum confidence to display
SENTENCE_COMMIT_FRAMES = 25  # Frames holding same gesture to commit to sentence
HAND_PADDING = 30            # Crop padding around hand box
NO_HAND_RESET_FRAMES = 12    # Clear current label after N frames without hands

# MediaPipe Hand Connections (Pairs of landmark indices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17)                                    # Base
]


def draw_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for p1, p2 in HAND_CONNECTIONS:
        cv2.line(frame, coords[p1], coords[p2], (0, 255, 180), 2)
    for x, y in coords:
        cv2.circle(frame, (x, y), 4, (0, 140, 255), -1)
        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)


def main():
    if YOLO_PATH is None:
        print("Error: No trained model weights found in models/ or runs/classify/.")
        print("Please train a model first using: python train.py")
        return

    if not LANDMARKER_PATH.exists():
        print(f"Error: MediaPipe landmarker model not found at {LANDMARKER_PATH}")
        return

    print(f"Loading YOLO Classifier from: {YOLO_PATH}")
    yolo_model = YOLO(str(YOLO_PATH))
    class_names = list(yolo_model.names.values())
    print(f"Recognized Gestures ({len(class_names)}): {class_names}")

    print(f"Initializing MediaPipe HandLandmarker from: {LANDMARKER_PATH}")
    base_options = python.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Tracking states
    prob_history = deque(maxlen=SMOOTH_WINDOW)
    no_hand_count = 0
    display_label = None
    display_confidence = 0.0
    consecutive_same = 0
    last_winner_id = -1
    commit_counter = 0
    last_committed_word = None
    sentence = []
    show_skeleton = True

    print("\n[+] Starting Real-Time Detector...")
    print("Controls: Press Q to Quit | C to Clear Sentence | Backspace to Delete | H to Toggle Skeleton\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for natural mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = detector.detect(mp_image)

        hand_found = False

        if detection_result.hand_landmarks:
            hand_found = True
            no_hand_count = 0
            landmarks = detection_result.hand_landmarks[0]

            if show_skeleton:
                draw_skeleton(frame, landmarks)

            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]

            pad = HAND_PADDING
            xmin = max(0, min(xs) - pad)
            ymin = max(0, min(ys) - pad)
            xmax = min(w, max(xs) + pad)
            ymax = min(h, max(ys) + pad)

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 230, 255), 2)

            hand_crop = frame[ymin:ymax, xmin:xmax]

            if hand_crop.size > 0 and hand_crop.shape[0] >= 32 and hand_crop.shape[1] >= 32:
                results_cls = yolo_model(hand_crop, verbose=False)
                probs = results_cls[0].probs.data.cpu().numpy()
                prob_history.append(probs)

                # Smoothed prediction
                smoothed_probs = np.mean(prob_history, axis=0)
                winner_id = int(np.argmax(smoothed_probs))
                winner_conf = float(smoothed_probs[winner_id])

                if winner_id == last_winner_id:
                    consecutive_same += 1
                else:
                    consecutive_same = 1
                    last_winner_id = winner_id
                    commit_counter = 0

                if consecutive_same >= CONSECUTIVE_FRAMES and winner_conf >= DISPLAY_CONFIDENCE:
                    display_label = class_names[winner_id]
                    display_confidence = winner_conf
                    commit_counter += 1

                    # Commit to sentence when held steady
                    if commit_counter >= SENTENCE_COMMIT_FRAMES:
                        if display_label != last_committed_word:
                            sentence.append(display_label.upper())
                            last_committed_word = display_label
                            commit_counter = 0
                else:
                    commit_counter = 0
        else:
            no_hand_count += 1
            if no_hand_count >= NO_HAND_RESET_FRAMES:
                prob_history.clear()
                display_label = None
                display_confidence = 0.0
                consecutive_same = 0
                last_winner_id = -1
                commit_counter = 0
                last_committed_word = None

        # -----------------------------
        # UI & WRITTEN FORM DISPLAY
        # -----------------------------
        # Top HUD for detected gesture
        top_bar = frame.copy()
        cv2.rectangle(top_bar, (0, 0), (w, 85), (20, 20, 24), -1)
        cv2.addWeighted(top_bar, 0.75, frame, 0.25, 0, frame)

        if display_label is not None:
            # Word output (the written form)
            word_text = display_label.upper()
            cv2.putText(frame, word_text, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 120), 2)

            # Confidence score & mini bar
            conf_str = f"{display_confidence * 100:.1f}%"
            cv2.putText(frame, conf_str, (w - 130, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)
            
            bar_w = int(100 * min(display_confidence, 1.0))
            cv2.rectangle(frame, (w - 130, 60), (w - 30, 70), (60, 60, 60), -1)
            cv2.rectangle(frame, (w - 130, 60), (w - 130 + bar_w, 70), (0, 255, 120), -1)
        else:
            hint = "Show hand gesture to webcam..." if hand_found else "Waiting for hand..."
            cv2.putText(frame, hint, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (140, 140, 140), 1)

        # Bottom HUD for Sentence Construction
        bottom_bar = frame.copy()
        cv2.rectangle(bottom_bar, (0, h - 85), (w, h), (18, 18, 22), -1)
        cv2.addWeighted(bottom_bar, 0.8, frame, 0.2, 0, frame)

        sentence_str = " ".join(sentence[-7:]) if sentence else "..."
        cv2.putText(frame, "Sentence: " + sentence_str, (20, h - 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

        cv2.putText(frame, "Q: Quit | C: Clear | Backspace: Delete Word | H: Skeleton",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("Sign Language Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("c") or key == ord("C"):
            sentence.clear()
            last_committed_word = None
        elif key == 8:  # Backspace
            if sentence:
                sentence.pop()
                last_committed_word = sentence[-1].lower() if sentence else None
        elif key == ord("h") or key == ord("H"):
            show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed successfully.")


if __name__ == "__main__":
    main()
