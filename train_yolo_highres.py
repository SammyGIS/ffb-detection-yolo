"""
High-Resolution FFB Bunch Detection Training & TFLite INT8 Export Script
Optimized for: 800x800 resolution, 4:3 Field of View, Dense Bunch Clusters
"""

import os
import argparse
import shutil
import glob
from ultralytics import YOLO

def train_highres_model(data_yaml, epochs=25, imgsz=640, batch=32, model_base='yolo11n.pt', project_name='my_model_2026_10_16'):
    """
    Trains YOLOv11 at high-speed with RAM-cached dataset and hyper-parameters
    tuned for dense palm fruit bunch clusters and model-side false positive rejection.
    Completes in ~45-55 minutes on RTX 4060 GPU.
    """
    print("=" * 60)
    print(f"Starting High-Speed Training: {model_base} at resolution {imgsz}x{imgsz}")
    print(f"Target Configuration: 0.12 Conf, 0.50 IoU, RAM Cache Enabled")
    print("=" * 60)

    model = YOLO(model_base)

    # Train model with RAM caching to eliminate disk bottlenecks
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project='runs/detect',
        name=project_name,
        exist_ok=True,      # Overwrite/continue in designated run folder
        cache='disk',       # Caches preprocessed arrays to disk with zero RAM pressure
        workers=2,          # Stable multi-threaded dataloader
        # Augmentations tuned for dense clusters & small bunches
        mosaic=1.0,         # Combines 4 images into 1, forcing model to detect smaller scales
        mixup=0.15,         # Blends overlapping images to handle partially occluded bunches
        scale=0.5,          # Scale variation (0.5 to 1.5x) to recognize varying bunch sizes
        fliplr=0.5,         # Horizontal flip
        flipud=0.2,         # Vertical flip for varied pile angles
        close_mosaic=5,     # Turn off mosaic last 5 epochs for sharp fine-tuning
        iou=0.50,           # Target IoU matching mobile runtime (50%)
        save=True,
        save_period=5,
        device=0
    )

    # Robustly locate best.pt
    best_weights = None
    if hasattr(results, 'save_dir') and results.save_dir:
        cand = os.path.join(str(results.save_dir), 'weights', 'best.pt')
        if os.path.exists(cand):
            best_weights = cand

    if not best_weights or not os.path.exists(best_weights):
        candidates = glob.glob(f"runs/**/{project_name}*/weights/best.pt", recursive=True) + glob.glob("runs/**/best.pt", recursive=True)
        best_weights = candidates[0] if candidates else os.path.join('runs', 'detect', project_name, 'weights', 'best.pt')

    print(f"\nTraining completed! Best weights saved at: {best_weights}")
    
    # Also save an official copy in models/
    os.makedirs('models', exist_ok=True)
    target_pt = os.path.join('models', f'{project_name}.pt')
    if os.path.exists(best_weights):
        shutil.copy2(best_weights, target_pt)
        print(f"Archived PyTorch weights to: {target_pt}")
    return best_weights

def export_to_tflite_int8(weights_path, imgsz=832, data_yaml=None, date_tag='2026_10_16'):
    """
    Exports trained PyTorch weights (.pt) to INT8 Quantized TFLite for mobile inference.
    """
    print("\n" + "=" * 60)
    print(f"Exporting {weights_path} to TFLite (INT8 Quantized, {imgsz}x{imgsz})...")
    print("=" * 60)

    model = YOLO(weights_path)
    
    # Export to TFLite INT8
    model.export(
        format='tflite',
        imgsz=imgsz,
        int8=True,
        data=data_yaml  # Used for calibration images during INT8 quantization
    )

    os.makedirs('models', exist_ok=True)
    target_tflite = os.path.join('models', f'my_model_{date_tag}_int8.tflite')

    # Look for exported .tflite files
    parent_dir = os.path.dirname(weights_path)
    candidates = glob.glob(os.path.join(parent_dir, "*int8*.tflite")) + \
                 glob.glob(os.path.join(parent_dir, "**", "*int8*.tflite"), recursive=True) + \
                 glob.glob(os.path.join(parent_dir, "*.tflite")) + \
                 glob.glob(os.path.join(parent_dir, "**", "*.tflite"), recursive=True)

    if candidates:
        src_tflite = candidates[0]
        shutil.copy2(src_tflite, target_tflite)
        print(f"Exported model copied to: {target_tflite}")

        # Also copy to mobile app assets if path exists
        mobile_assets = os.path.join("..", "okomu-mobile-app", "assets")
        if os.path.exists(mobile_assets):
            app_target = os.path.join(mobile_assets, "my_model_int8.tflite")
            shutil.copy2(src_tflite, app_target)
            print(f"Synchronized with mobile app assets at: {app_target}")
    else:
        print(f"Warning: Could not locate exported .tflite under {parent_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train High-Speed FFB Bunch Detector (1-Hour Profile)")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml dataset definition")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default 25)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default 640)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default 32 for RTX 4060 8GB)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model: yolo11n.pt")
    parser.add_argument("--export-only", type=str, default=None, help="Path to existing best.pt to export directly without training")

    args = parser.parse_args()

    if args.export_only:
        export_to_tflite_int8(args.export_only, imgsz=args.imgsz, data_yaml=args.data)
    else:
        if not os.path.exists(args.data):
            print(f"Dataset configuration '{args.data}' not found. Please provide path with --data path/to/data.yaml")
            return
        best_pt = train_highres_model(args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, model_base=args.model)
        export_to_tflite_int8(best_pt, imgsz=args.imgsz, data_yaml=args.data)

if __name__ == "__main__":
    main()
