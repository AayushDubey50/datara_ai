import matplotlib.pyplot as plt

def visualize_predictions(model, dataset, device, num_images=8):
    """
    Display a few images with their true and predicted labels side by side.
    """
    model.eval()
    fig = plt.figure(figsize=(15, 5))
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            if i >= num_images:
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            img = inputs.cpu().squeeze().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            ax = fig.add_subplot(2, num_images//2, i+1)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"True: {dataset.classes[labels.item()]}\nPred: {dataset.classes[preds.item()]}")
    
    plt.tight_layout()
    plt.show()

# Usage example:
visualize_predictions(model, val_dataset, device, num_images=8)
