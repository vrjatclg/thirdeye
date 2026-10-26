"""
ThirdEye Autonomous Defense AI - Military Armor Training Pipeline
Fine-tunes YOLO11 on Aerial Combat & Military Vehicle Reconnaissance Datasets.
Supports:
  - Multi-scale mosaic augmentation (combat camouflage & smoke resilience)
  - Mixup augmentation (artillery blast & partial tree-line occlusion)
  - AdamW optimizer with cosine learning rate scheduling
  - High resolution imgsz=1024 for long-range drone target preservation
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO

def train_military_model(
    data_yaml: str,
    base_model: str = "yolo11s.pt",
    epochs: int = 50,
    imgsz: int = 1024,
    batch_size: int = 8,
    project_name: str = "ThirdEye_Drone_Armor",
    run_name: str = "yolo11_aerial_combat_v1",
    device: str = "0"
):
    print("=" * 70)
    print("  THIRDEYE DEFENSE AI // MILITARY ARMOR FINE-TUNING PIPELINE")
    print("=" * 70)
    print(f"[*] Base Weights:    {base_model}")
    print(f"[*] Dataset Config:  {data_yaml}")
    print(f"[*] Resolution:      {imgsz}x{imgsz} (Small target preservation)")
    print(f"[*] Epochs / Batch:  {epochs} epochs, batch={batch_size}")
    print(f"[*] Hardware:        CUDA Device {device}")
    print("=" * 70)

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset configuration file not found at: {data_yaml}")

    # Load base pretrained model
    model = YOLO(base_model)

    # Train with defense-optimized hyperparameters
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        # Tactical Augmentation Suite
        mosaic=1.0,           # 4-image mosaic for small target scale variety
        mixup=0.15,          # Blends images to simulate artillery smoke & dust
        close_mosaic=10,     # Disables mosaic last 10 epochs for sharp localization
        scale=0.5,           # Multi-scale zooming (50% to 150%)
        shear=2.0,           # Oblique aerial drone pitch angles
        perspective=0.0005,  # 3D aerial camera slant
        fliplr=0.5,
        plots=True,
        save=True,
        save_period=5,
        project=project_name,
        name=run_name
    )

    best_weights = Path(project_name) / run_name / "weights" / "best.pt"
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE // WEIGHTS EXPORTED")
    print("=" * 70)
    print(f"[+] Best weights saved to: {best_weights.resolve()}")
    print("[+] To activate this model in ThirdEye, copy to: models/military_convoy_yolo11l.pt")
    return str(best_weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ThirdEye on Aerial Combat Datasets")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--base", type=str, default="yolo11s.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (8 fits 8GB VRAM)")
    args = parser.parse_args()

    train_military_model(
        data_yaml=args.data,
        base_model=args.base,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch
    )
