from pathlib import Path

# Paths
ROOT_DIR = Path("unwelded_images")
SPLITS = ["train", "val"]
MASK_SUBDIR = "masks_clean"

for split in SPLITS:
    images_dir = ROOT_DIR / split / "images"
    masks_dir = ROOT_DIR / split / MASK_SUBDIR

    print(f"\nChecking {split} split:")
    
    # Build mask map by stem
    masks_map = {m.stem: m for m in masks_dir.iterdir() if m.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    
    # List all images
    images = [img for img in images_dir.iterdir() if img.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    
    if not images:
        print(f"  [WARN] No images found in {images_dir}")
        continue
    
    missing_masks = []
    for img in images:
        if img.stem not in masks_map:
            missing_masks.append(img.name)
    
    if missing_masks:
        print(f"  [MISSING] {len(missing_masks)} images have no corresponding mask:")
        for m in missing_masks:
            print(f"    - {m}")
    else:
        print(f"  All {len(images)} images have corresponding masks.")
