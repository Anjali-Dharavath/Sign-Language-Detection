# 🖐️ Real-Time Sign Language Detection & Translator

An AI-powered computer vision application that detects sign language hand gestures in real-time through your webcam and translates them into **written text** on screen, complete with real-time confidence scores and a running sentence builder.

---

## 🚀 Features

- **Real-Time Translation**: Converts live hand gestures into bold, high-visibility written text.
- **MediaPipe + YOLOv8 Architecture**: Accurate hand localization combined with deep learning classification.
- **Temporal Smoothing**: Rolling confidence filtering prevents screen flickering and jitter.
- **Sentence Builder**: Hold a gesture steady to automatically append words to your running sentence.
- **Easy Custom Word Addition**: Capture any new custom word/gesture and retrain in minutes.

---

## 📁 Clean Project Structure

```text
sign_language_detection/
├── runs/                           # YOLO training artifacts & checkpoints
│   └── classify/
└── signlang_Detection/             # Main application codebase
    ├── models/
    │   ├── best.pt                 # Active trained YOLO gesture model
    │   └── hand_landmarker.task    # MediaPipe hand tracking model
    ├── dataset/
    │   ├── train/                  # Training images per gesture
    │   └── val/                    # Validation images per gesture
    ├── realtime.py                 # Main real-time webcam detector
    ├── collect_data.py             # Interactive gesture data collector
    ├── train.py                    # 1-command model trainer
    ├── requirements.txt            # Python dependencies
    ├── run_realtime.bat            # Windows 1-click launcher
    ├── run_realtime.ps1            # PowerShell launcher
    └── README.md                   # Documentation
```

---

## ⚡ Quick Start

### 1. Run Real-Time Detection
Run the detector from your terminal:
```bash
python realtime.py
```
*(Or simply double-click `run_realtime.bat` on Windows)*

### Controls in Real-Time Window:
| Key | Action |
| :--- | :--- |
| **`Q`** | Quit application |
| **`C`** | Clear running sentence |
| **`Backspace`** | Delete the last word from sentence |
| **`H`** | Toggle hand landmark skeleton overlay on/off |

---

## ➕ How to Add New Words & Gestures

You can add any new sign gesture whenever you want:

### Step 1: Collect Gesture Images
Run the interactive data collector:
```bash
python collect_data.py
```
- Press **`1` to `9`** to select an existing preset gesture, **OR**
- Press **`N`** and type a **NEW** custom word (e.g. `water`, `food`, `family`, `friend`).
- Position your hand in front of the webcam.
- Press **`SPACE`** to start recording (it automatically crops the hand and splits 80% train / 20% validation).
- Press **`S`** to stop recording.
- Press **`Q`** to quit when done.

### Step 2: Retrain the Model
Once you have collected images for your new word, retrain the model with a single command:
```bash
python train.py
```
*(Optional custom parameters: `python train.py --epochs 30 --imgsz 128 --batch 32`)*

When training finishes, `train.py` automatically updates `models/best.pt` with your new gestures. Run `python realtime.py` to immediately test your new words!
