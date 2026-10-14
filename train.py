"""
Training script for Sign Language Gesture Classifier using YOLOv8 Classification.
Trains on images in dataset/ and saves the best model to models/best.pt.

Usage:
    python train.py
    python train.py --epochs 30 --imgsz 128 --batch 32
"""

import argparse
import os
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO


def train_model(epochs=30, imgsz=128, batch=32, model_name="yolov8n-cls.pt"):
    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "dataset"
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists() or not (dataset_dir / "train").exists():
        raise FileNotFoundError(f"Dataset directory not found at {dataset_dir}. Run collect_data.py first.")

    train_classes = sorted([d.name for d in (dataset_dir / "train").iterdir() if d.is_dir()])
    print(f"\n==========================================")
    print(f" Detected {len(train_classes)} gesture classes:")
    for idx, name in enumerate(train_classes):
        print(f"   [{idx}] {name}")
    print(f"==========================================\n")

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Starting training on device: {device}")
    print(f"Epochs: {epochs} | Image size: {imgsz}x{imgsz} | Batch size: {batch}\n")

    # Load YOLO classification base model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=str(dataset_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project_dir.parent / "runs" / "classify"),
        name="train_custom",
        exist_ok=True,
        verbose=True
    )

    # Find the resulting best.pt
    run_save_dir = Path(results.save_dir)
    best_weights = run_save_dir / "weights" / "best.pt"

    if best_weights.exists():
        target_path = models_dir / "best.pt"
        shutil.copy(best_weights, target_path)
        print(f"\n Success! Trained model saved to: {target_path}")
        print("You can now run: python realtime.py to use your new gestures!")
    else:
        print(f"Warning: Could not locate best.pt in {run_save_dir / 'weights'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sign Language Gesture Classifier")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--imgsz", type=int, default=128, help="Image size for classification (default: 128)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt", help="Base YOLO classification model")
    args = parser.parse_args()

    train_model(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, model_name=args.model)
