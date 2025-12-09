import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------- Config ----------
ROOT_DIR = Path("unwelded_images")   # root containing train/ and val/
SPLITS = ["train", "val"]            # the splits we expect
MASK_SUBDIR = "masks_clean"          # clean masks will be saved here
IMG_SIZE = (256, 256)                # resize for training

# Edge detection / mask generation tunables
IMG_MAX_W   = 1024
CANNY_LO    = 50
CANNY_HI    = 150
HOUGH_TH    = 60
MIN_LINE_FRAC = 0.15
MAX_LINE_GAP  = 10
DILATE_PX     = 2

# ------------------ Mask Generation ------------------

def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    h, w = img.shape[:2]
    if max(h, w) > IMG_MAX_W:
        scale = IMG_MAX_W / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img

def enhance_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def detect_edges(gray):
    gray = cv2.GaussianBlur(gray, (3,3), 0)  # blur to reduce noise
    edges = cv2.Canny(gray, CANNY_LO, CANNY_HI, L2gradient=True)
    return edges

def merge_lines(lines, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if lines is None:
        return mask
    h, w = img_shape[:2]
    cx = w // 2  # center x
    for x1, y1, x2, y2 in lines[:,0,:]:
        # keep lines near center
        if abs((x1+x2)//2 - cx) < w * 0.4:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if -45 <= angle <= 45:  # mostly horizontal
                cv2.line(mask, (x1,y1),(x2,y2),255,1,cv2.LINE_AA)
    return mask

def thin(mask):
    try:
        import cv2.ximgproc as xi
        return xi.thinning(mask, xi.THINNING_ZHANGSUEN)
    except Exception:
        # morphological skeleton as fallback
        skel = np.zeros_like(mask)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        done = False
        working = mask.copy()
        while not done:
            eroded = cv2.erode(working, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(working, temp)
            skel = cv2.bitwise_or(skel, temp)
            working = eroded.copy()
            done = (cv2.countNonZero(working) == 0)
        return skel

def process_one(img_path, mask_dir):
    img = read_image(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = enhance_contrast(gray)

    edges = detect_edges(gray)
    h, w = edges.shape
    min_len = int(min(h, w) * MIN_LINE_FRAC)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_TH,
                            minLineLength=min_len, maxLineGap=MAX_LINE_GAP)
    line_mask = merge_lines(lines, img.shape)

    if line_mask.sum() == 0:
        line_mask = edges  # fallback

    seam_mask = thin(line_mask)
    if DILATE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*DILATE_PX+1, 2*DILATE_PX+1))
        seam_mask = cv2.dilate(seam_mask, k, iterations=1)

    seam_mask[seam_mask > 0] = 255

    mask_dir.mkdir(parents=True, exist_ok=True)
    out_path = mask_dir / (img_path.stem + ".png")
    cv2.imwrite(str(out_path), seam_mask)
    return out_path

def generate_masks():
    for split in SPLITS:
        split_dir = ROOT_DIR / split / "images"
        mask_dir  = ROOT_DIR / split / MASK_SUBDIR

        mask_dir.mkdir(parents=True, exist_ok=True)  # always create folder

        imgs = sorted([p for p in split_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}])
        if not imgs:
            print(f"[WARN] No images in {split_dir}")
            continue

        print(f"Processing {len(imgs)} images in {split_dir}...")
        for img_path in imgs:
            try:
                out = process_one(img_path, mask_dir)
                print(f"  [OK] {img_path.name} -> {out.name}")
            except Exception as e:
                print(f"  [FAIL] {img_path.name}: {e}")

    print("Done. Masks generated in masks_clean/.")

# ------------------ Dataset for Training ------------------

class SeamDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        # Map masks by stem (ignore extension)
        self.masks_map = {m.stem: m for m in self.masks_dir.iterdir() 
                          if m.suffix.lower() in {".jpg", ".jpeg", ".png"}}

        # Keep only images with a corresponding mask
        self.images = [img for img in sorted(self.images_dir.iterdir()) 
                       if img.suffix.lower() in {".jpg",".jpeg",".png"} and img.stem in self.masks_map]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_map[img_path.stem]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize for training
        image = image.resize(IMG_SIZE, Image.BILINEAR)
        mask = mask.resize(IMG_SIZE, Image.NEAREST)

        mask = np.array(mask)
        mask = (mask > 128).astype(np.uint8)

        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(mask).long()
        return image, mask

# ------------------ Example: creating dataloaders ------------------

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = SeamDataset(ROOT_DIR/"train/images", ROOT_DIR/"train"/MASK_SUBDIR, transform=train_transform)
val_dataset   = SeamDataset(ROOT_DIR/"val/images", ROOT_DIR/"val"/MASK_SUBDIR, transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False)

# ------------------ Run mask generation ------------------
if __name__ == "__main__":
    print("Generating masks for all images...")
    generate_masks()
