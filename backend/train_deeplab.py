import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

# --------------------
# Dataset Class
# --------------------
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform_img=None, transform_mask=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        # Gather all image files
        image_files = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_files.append(os.path.join(root, f))

        # Gather only _seam masks
        mask_files = []
        for root, _, files in os.walk(mask_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")) and "_seam" in f.lower():
                    mask_files.append(os.path.join(root, f))

        # Helper to normalize names
        def normalize_name(path):
            return os.path.splitext(os.path.basename(path))[0].lower().replace("_seam", "")

        image_dict = {normalize_name(f): f for f in image_files}
        mask_dict  = {normalize_name(f): f for f in mask_files}

        # Match pairs
        common_keys = sorted(set(image_dict.keys()) & set(mask_dict.keys()))
        self.pairs = [(image_dict[k], mask_dict[k]) for k in common_keys]

        print(f"✅ Found {len(self.pairs)} valid image-mask pairs (from {len(image_files)} images, {len(mask_files)} masks)")
        if len(self.pairs) > 0:
            print("Example pair:", self.pairs[0])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        mask = (mask > 127).long().squeeze(0)  # binary {0,1}

        return image, mask

# --------------------
# Paths
# --------------------
train_img_dir = "dataset/train/images"
train_mask_dir = "dataset/train/masks"
val_img_dir = "dataset/val/images"
val_mask_dir = "dataset/val/masks"

# --------------------
# Transforms
# --------------------
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# --------------------
# Datasets & Loaders
# --------------------
train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, transform_img, transform_mask)
val_dataset = SegmentationDataset(val_img_dir, val_mask_dir, transform_img, transform_mask)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# --------------------
# Model (DeeplabV3)
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)  # binary classes
model = model.to(device)

# --------------------
# Loss & Optimizer
# --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --------------------
# Training Loop
# --------------------
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

torch.save(model.state_dict(), "deeplab_seam.pth")
print("✅ Training complete. Model saved to deeplab_seam.pth")
