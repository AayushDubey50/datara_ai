import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms as T
from tqdm import tqdm

from dataset_segmentation import WeldSegmentationDataset

def train_segmentation(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    for images, masks in tqdm(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate_segmentation(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            running_loss += loss.item()
    return running_loss / len(dataloader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Change this to "_seam.png" or "_weld.png" if you want
    mask_suffix = "_combined.png"

    transform = T.Compose([
        T.Resize((256,256)),
        T.ToTensor(),
    ])

    train_dataset = WeldSegmentationDataset(
        "dataset/train/images",
        "dataset/train/masks",
        mask_suffix=mask_suffix,
        transform=transform
    )
    val_dataset = WeldSegmentationDataset(
        "dataset/val/images",
        "dataset/val/masks",
        mask_suffix=mask_suffix,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[-1] = nn.Conv2d(256,1,kernel_size=1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    for epoch in range(epochs):
        train_loss = train_segmentation(model, train_loader, optimizer, criterion, device)
        val_loss = validate_segmentation(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "deeplabv3_weld_segmentation.pth")
