import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# --------------------------
# Load Model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)  # background + seam path
model.load_state_dict(torch.load("models/seam_path_model.pth", map_location=device))
model.to(device)
model.eval()

# --------------------------
# Prediction Function
# --------------------------
def predict_image(image_path, save_path=None):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
    ])

    # Load & preprocess
    image = read_image(image_path).float() / 255.0
    orig_image = image.permute(1, 2, 0).numpy()  # HWC for plotting
    image = transform(image).unsqueeze(0).to(device)  # add batch dim

    # Run inference
    with torch.no_grad():
        output = model(image)["out"]
        preds = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Overlay mask (red seam path)
    overlay = orig_image.copy()
    overlay[preds == 1] = [1.0, 0.0, 0.0]  # red overlay for seam

    # Plot
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(orig_image)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Seam Path Prediction")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Saved prediction overlay at {save_path}")
    else:
        plt.show()

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    test_image = "unwelded_images/val/images/example.jpg"  # change to your test file
    predict_image(test_image, save_path="prediction_example.png")
