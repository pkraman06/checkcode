import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from dataset import BrainMRIDataset, get_transforms, download_kaggle_dataset
from model import ResNetSEBrainTumor

def train_model(data_dir="./data/brain-tumor-mri-dataset", epochs=15, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    train_tf, val_tf = get_transforms()
    train_ds = BrainMRIDataset(os.path.join(data_dir, "Training"), transform=train_tf)
    val_ds = BrainMRIDataset(os.path.join(data_dir, "Testing"), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = ResNetSEBrainTumor(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    os.makedirs("./checkpoints", exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = (correct / total) * 100

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = (val_correct / val_total) * 100
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save Checkpoints (for Epoch Probing & Best Weight)
        torch.save(model.state_dict(), f"./checkpoints/model_epoch_{epoch}.pth")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./checkpoints/best_model.pth")

    print(f"\nTraining Complete. Best Test Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    download_kaggle_dataset()
    train_model()
