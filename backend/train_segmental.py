import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from PIL import Image
import os
import glob

# --- Custom Dataset ---
class WeldDataset(Dataset):
    def __init__(self, image_dir, mask_dir, mask_suffix="_combined.png", transform=None):
        self.image_paths = []
        self.mask_paths = []

        for cls in ["good", "bad"]:
            img_subdir = os.path.join(image_dir, cls)
            mask_subdir = os.path.join(mask_dir, cls)

            exts = ["*.png", "*.jpg", "*.jpeg"]
            all_imgs = []
            for ext in exts:
                all_imgs.extend(glob.glob(os.path.join(img_subdir, ext)))

            for img_path in all_imgs:
                if not os.path.isfile(img_path):
                    continue

                base = os.path.splitext(os.path.basename(img_path))[0]
                mask_filename = base + mask_suffix
                mask_path = os.path.join(mask_subdir, mask_filename)

                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"❌ Mask not found for {img_path} (expected {mask_path})")

                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)

        self.transform = transform

        # Stats
        num_good = len([p for p in self.image_paths if "good" in p])
        num_bad = len([p for p in self.image_paths if "bad" in p])
        print(f"[INFO] Loaded {len(self.image_paths)} samples ({num_good} good, {num_bad} bad)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert mask to class labels: 0=good, 1=defect 
        mask = (mask > 0.5).long().squeeze(0)  # shape [H,W]

        return image, mask


# --- Training Parameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4

# --- Transforms ---
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

# --- Load Datasets ---
train_dataset = WeldDataset("dataset/train/images", "dataset/train/masks", transform=transform)
val_dataset = WeldDataset("dataset/val/images", "dataset/val/masks", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Model ---
model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier[-1] = nn.Conv2d(256, 2, kernel_size=1)  # 2 classes (good vs defect)
model.to(DEVICE)

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        outputs = model(images)["out"]  # shape [B,2,H,W]
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, "
          f"Train Loss: {train_loss/len(train_loader):.4f}, "
          f"Val Loss: {val_loss/len(val_loader):.4f}")

# --- Save Trained Model ---
torch.save(model.state_dict(), "segmentation_model.pth")
print("✅ Model saved as segmentation_model.pth")
