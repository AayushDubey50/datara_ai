import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import copy

# ----------------------------
# CONFIG
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4
EARLY_STOPPING_PATIENCE = 5
DATA_DIR = "dataset"  # expects train/images/{good,bad} and val/images/{good,bad}

# ----------------------------
# TRANSFORMS
# ----------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# DATASETS
# ----------------------------
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train/images"), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val/images"), transform=val_transforms)

print("Class to index mapping:", train_dataset.class_to_idx)

# Handle class imbalance with WeightedRandomSampler
class_counts = Counter([label for _, label in train_dataset.samples])
print("Class counts:", class_counts)
class_weights = [1.0 / class_counts[label] for _, label in train_dataset.samples]
train_sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(class_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# MODEL
# ----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_ftrs, 2)
)
model = model.to(DEVICE)

# ----------------------------
# LOSS & OPTIMIZER
# ----------------------------
weight_tensor = torch.tensor([class_counts[1]/class_counts[0], 1.0], device=DEVICE)  # "bad" vs "good"
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# ----------------------------
# TRAINING LOOP WITH EARLY STOPPING
# ----------------------------
best_val_acc = 0
best_model_wts = copy.deepcopy(model.state_dict())
early_stop_counter = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0
    per_class_correct = Counter()
    per_class_total = Counter()

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for label, pred in zip(labels, preds):
            per_class_correct[label.item()] += (pred == label).item()
            per_class_total[label.item()] += 1

    train_loss = running_loss / total
    train_acc = correct / total
    per_class_acc = {cls: per_class_correct[cls]/per_class_total[cls] for cls in per_class_correct}

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    val_per_class_correct = Counter()
    val_per_class_total = Counter()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            for label, pred in zip(labels, preds):
                val_per_class_correct[label.item()] += (pred == label).item()
                val_per_class_total[label.item()] += 1

    val_acc = val_correct / val_total
    val_per_class_acc = {cls: val_per_class_correct[cls]/val_per_class_total[cls] for cls in val_per_class_correct}

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Per-class: {per_class_acc}")
    print(f"Validation Acc: {val_acc:.4f}, Per-class: {val_per_class_acc}")

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), "classifier_resnet18_best.pth")
        print("✅ Saved new best model")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print("⏹ Early stopping triggered")
            break

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "classifier_resnet18_final.pth")
print(f"Training complete. Best Validation Acc: {best_val_acc:.4f}")
print("✅ Final model saved as classifier_resnet18_final.pth")
