"""
High-Resolution FFB Bunch Detection Training & TFLite INT8 Export Script
Optimized for: 800x800 resolution, 4:3 Field of View, Dense Bunch Clusters
"""

import os
import argparse
from ultralytics import YOLO

def train_highres_model(data_yaml, epochs=100, imgsz=800, batch=16, model_base='yolo11n.pt', project_name='my_model_800'):
    """
    Trains YOLOv11 at 800x800 resolution with hyper-parameters tuned for
    dense palm fruit bunch clusters (avoiding false suppression of small/nested bunches).
    """
    print("=" * 60)
    print(f"Starting Training: {model_base} at resolution {imgsz}x{imgsz}")
    print(f"Target Configuration: 4:3 native aspect ratio, 0.12 Conf, 0.60 IoU")
    print("=" * 60)

    model = YOLO(model_base)

    # Train model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project='runs/detect',
        name=project_name,
        # Augmentations tuned for dense clusters & small bunches
        mosaic=1.0,         # Combines 4 images into 1, forcing model to detect smaller scales
        mixup=0.15,         # Blends overlapping images to handle partially occluded bunches
        scale=0.5,          # Scale variation (0.5 to 1.5x) to recognize varying bunch sizes
        fliplr=0.5,         # Horizontal flip
        flipud=0.2,         # Vertical flip for varied pile angles
        close_mosaic=10,    # Turn off mosaic last 10 epochs for sharp fine-tuning
        iou=0.60,           # Target IoU matching mobile runtime
        save=True,
        save_period=10,
        device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else 'cpu'
    )

    best_weights = os.path.join('runs', 'detect', project_name, 'weights', 'best.pt')
    print(f"\nTraining completed! Best weights saved at: {best_weights}")
    return best_weights

def export_to_tflite_int8(weights_path, imgsz=800, data_yaml=None):
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

    tflite_path = weights_path.replace('.pt', '_saved_model')
    print(f"\nExport complete! Check folder: {tflite_path}")
    print(f"Copy the exported int8 .tflite file to: okomu-mobile-app/assets/my_model_800_int8.tflite")

def main():
    parser = argparse.ArgumentParser(description="Train 800x800 High-Res FFB Bunch Detector")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml dataset definition")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=800, help="Image resolution (default 800)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (e.g. 16 or 8)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model: yolo11n.pt, yolo11s.pt, or yolo11m.pt")
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
