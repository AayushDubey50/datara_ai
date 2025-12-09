import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import random
import os

# ------------------ Config ------------------
ROOT_DIR = Path("unwelded_images")
TRAIN_IMG_DIR = ROOT_DIR / "train/images"
TRAIN_MASK_DIR = ROOT_DIR / "train/masks_clean"
VAL_IMG_DIR   = ROOT_DIR / "val/images"
VAL_MASK_DIR  = ROOT_DIR / "val/masks_clean"

IMG_SIZE = (320, 320)       # Slightly larger to preserve thin seams
BATCH_SIZE = 4
EPOCHS = 12
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "seam_path_model.pth"

# ------------------ Dataset ------------------
class SeamDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        # Map masks by stem
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

        # Resize
        image = image.resize(IMG_SIZE, Image.BILINEAR)
        mask = mask.resize(IMG_SIZE, Image.NEAREST)

        # Convert mask to binary 0/1
        mask = np.array(mask)
        mask = (mask > 128).astype(np.uint8)
        mask = torch.from_numpy(mask).long()

        # Data augmentation
        if self.transform:
            image = self.transform(image)

        # Random horizontal flip
        if random.random() > 0.5:
            image = F.hflip(image)
            mask = torch.flip(mask, [1])

        # Random rotation
        angle = random.uniform(-10, 10)
        image = F.rotate(image, angle)
        mask = F.rotate(mask.unsqueeze(0).float(), angle).squeeze(0).long()

        return image, mask

# ------------------ Transforms ------------------
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------------ Datasets & Dataloaders ------------------
train_dataset = SeamDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
val_dataset   = SeamDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------ Model ------------------
model = models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=False)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)  # 2 classes: background & seam
model = model.to(DEVICE)

# ------------------ Weighted Loss ------------------
# Count seam pixels for weighting
num_seam_pixels = sum([torch.sum(mask) for _, mask in train_dataset])
num_total_pixels = len(train_dataset) * IMG_SIZE[0] * IMG_SIZE[1]
weight_bg = 1.0
weight_seam = num_total_pixels / (num_seam_pixels + 1e-6)
class_weights = torch.tensor([weight_bg, weight_seam]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# ------------------ Optimizer & Scheduler ------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ------------------ Dice Metric ------------------
def dice_coeff(pred, target, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice

# ------------------ Training Loop ------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)['out']
            val_loss += criterion(outputs, masks).item() * images.size(0)
            val_dice += dice_coeff(outputs, masks).item() * images.size(0)
    val_loss /= len(val_loader.dataset)
    val_dice /= len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

    scheduler.step()

# ------------------ Save Model ------------------
torch.save(model.state_dict(), SAVE_PATH)
print(f"Training complete. Model saved to {SAVE_PATH}")
