import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# Allowed image extensions
IMG_EXTS = (".jpg", ".jpeg", ".png")

class WeldDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform

        # Recursively collect all image paths
        self.image_paths = []
        self.mask_paths = []

        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(IMG_EXTS):
                    img_path = os.path.join(root, f)

                    # Find corresponding mask
                    rel_path = os.path.relpath(img_path, image_dir)
                    mask_path = os.path.join(mask_dir, rel_path)

                    # Try with same extension first
                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                    else:
                        # Try switching extensions (e.g. jpg ↔ png)
                        base, _ = os.path.splitext(mask_path)
                        for ext in IMG_EXTS:
                            alt_path = base + ext
                            if os.path.exists(alt_path):
                                self.image_paths.append(img_path)
                                self.mask_paths.append(alt_path)
                                break

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in dataset at {image_dir}. "
                f"Check subfolders and file extensions {IMG_EXTS}."
            )

        print(f"[WeldDataset] Loaded {len(self.image_paths)} samples from {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale mask

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        # Convert mask to class indices (0 or 1)
        mask = (mask > 0.5).long().squeeze(0)

        return image, mask
