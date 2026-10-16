#!/usr/bin/env python3
"""
Okomu FFB Hard Negative Fine-Tuning Pipeline
Integrates:
- All Okomu farm elements (ground, leaves, fronds, trunks, empty sacks, loose fruit)
- Field test hard negatives (fertilizer sacks with loose fruit, indoor desk fans, room clutter)
Fine-tunes the production model (my_model_2026_10_16.pt) on NVIDIA RTX 4060 GPU.
"""

import os
import glob
import shutil
import random
from PIL import Image, ImageOps
from ultralytics import YOLO

BASE_DIR = r"c:\Users\Admin\Documents\DOYSAM\Okomu\Okomu App Development Project\ffb-count-detection"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_IMG = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LBL = os.path.join(DATASET_DIR, "train", "labels")
VAL_IMG = os.path.join(DATASET_DIR, "val", "images")
VAL_LBL = os.path.join(DATASET_DIR, "val", "labels")

NEGATIVES_DIR = os.path.join(BASE_DIR, "negatives")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "..", "okomu-mobile-app", "screenshot")

def inject_screenshot_negatives():
    """Extract camera photos from field test screenshots and augment with 4 rotations."""
    print("\n[1/5] Ingesting screenshot hard negatives (fan, sacks, room)...")
    screenshots = glob.glob(os.path.join(SCREENSHOT_DIR, "*.jpg"))
    added = 0
    crop_box = (66, 700, 1013, 1820)
    
    for idx, sc_path in enumerate(screenshots, 1):
        try:
            with Image.open(sc_path) as im:
                crop = im.crop(crop_box)
                rotations = [
                    (0, crop),
                    (90, crop.transpose(Image.Transpose.ROTATE_270)),
                    (180, crop.transpose(Image.Transpose.ROTATE_180)),
                    (270, crop.transpose(Image.Transpose.ROTATE_90)),
                ]
                for angle, rot_im in rotations:
                    name = f"hardneg_screenshot_{idx:03d}_r{angle}.jpg"
                    # 80% train, 20% val
                    target_img_dir = TRAIN_IMG if random.random() < 0.85 else VAL_IMG
                    target_lbl_dir = TRAIN_LBL if target_img_dir == TRAIN_IMG else VAL_LBL
                    
                    dst_img = os.path.join(target_img_dir, name)
                    dst_lbl = os.path.join(target_lbl_dir, name.replace(".jpg", ".txt"))
                    
                    rot_im.save(dst_img, "JPEG", quality=95)
                    open(dst_lbl, "w").close()
                    added += 1
        except Exception as e:
            print(f"  Error processing {sc_path}: {e}")
            
    print(f"  -> Added {added} augmented screenshot negative samples.")

def inject_farm_negatives():
    """Inject all available raw farm negative images (ground, fronds, trunks, sacks, loose fruit)."""
    print("\n[2/5] Ingesting Okomu farm negatives (ground, leaves, fronds, sacks, tools)...")
    farm_folders = [
        "1_loose_fruit",
        "2_ground_and_platforms",
        "3_fronds_and_trunks",
        "4_sacks_tools_wheelbarrows",
        "6_lookalikes",
        "7_random_objects",
        "8_west_africa_commons",
        "9_field_web_reviewed",
    ]
    
    all_neg_files = []
    for fld in farm_folders:
        p = os.path.join(NEGATIVES_DIR, fld)
        if os.path.exists(p):
            imgs = glob.glob(os.path.join(p, "**", "*.jpg"), recursive=True) + \
                   glob.glob(os.path.join(p, "**", "*.png"), recursive=True)
            all_neg_files.extend(imgs)
            
    print(f"  Found {len(all_neg_files)} raw farm negative images.")
    random.seed(42)
    random.shuffle(all_neg_files)
    
    # Ingest up to 3,000 diverse farm negatives
    selected = all_neg_files[:3000]
    added = 0
    for idx, src_p in enumerate(selected, 1):
        ext = os.path.splitext(src_p)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            continue
        name = f"farm_neg_{idx:05d}{ext}"
        target_img_dir = TRAIN_IMG if random.random() < 0.90 else VAL_IMG
        target_lbl_dir = TRAIN_LBL if target_img_dir == TRAIN_IMG else VAL_LBL
        
        dst_img = os.path.join(target_img_dir, name)
        dst_lbl = os.path.join(target_lbl_dir, os.path.splitext(name)[0] + ".txt")
        
        if not os.path.exists(dst_img):
            shutil.copy2(src_p, dst_img)
            open(dst_lbl, "w").close()
            added += 1
            
    print(f"  -> Successfully added {added} farm negative images to dataset.")

def clear_caches():
    """Remove YOLO cache files so it re-reads all new negatives."""
    print("\n[3/5] Clearing dataset caches...")
    for c in glob.glob(os.path.join(DATASET_DIR, "**", "*.cache"), recursive=True):
        try:
            os.remove(c)
            print(f"  Removed cache: {c}")
        except Exception:
            pass

def finetune_model(epochs=8, imgsz=640, batch=32):
    """Fine-tune the production model on the enriched dataset."""
    print("\n[4/5] Fine-tuning YOLOv11 on NVIDIA RTX 4060 GPU...")
    base_model_path = os.path.join(BASE_DIR, "models", "my_model_2026_10_16.pt")
    data_yaml = os.path.join(DATASET_DIR, "data.yaml")
    
    model = YOLO(base_model_path)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=os.path.join(BASE_DIR, "runs", "detect"),
        name="finetune_hard_negatives",
        exist_ok=True,
        cache="disk",
        workers=2,
        mosaic=0.5,
        mixup=0.1,
        fliplr=0.5,
        flipud=0.2,
        iou=0.50,
        lr0=0.001,  # Lower learning rate for fine-tuning to protect existing weights
        lrf=0.01,
        save=True,
        device=0
    )
    
    best_pt = os.path.join(BASE_DIR, "runs", "detect", "finetune_hard_negatives", "weights", "best.pt")
    if not os.path.exists(best_pt):
        candidates = glob.glob(os.path.join(BASE_DIR, "runs", "detect", "**", "best.pt"), recursive=True)
        best_pt = max(candidates, key=os.path.getmtime)
        
    print(f"  Fine-tuning complete! Best weights: {best_pt}")
    
    # Save as new official checkpoint
    target_pt = os.path.join(BASE_DIR, "models", "my_model_2026_10_16.pt")
    shutil.copy2(best_pt, target_pt)
    print(f"  Updated master checkpoint: {target_pt}")
    return target_pt

def main():
    inject_screenshot_negatives()
    inject_farm_negatives()
    clear_caches()
    finetune_model(epochs=8, imgsz=640, batch=32)
    print("\n[5/5] Pipeline finished successfully! Ready for export and APK rebuild.")

if __name__ == "__main__":
    main()
