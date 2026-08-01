import os
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from dataset import BrainMRIDataset, get_transforms, get_mri_dataset_path
from model import ResNetSEBrainTumor

def extract_features_at_layer(model, dataloader, layer_idx, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls, _ in dataloader:
            imgs = imgs.to(device)
            feats = model.forward_features(imgs)
            layer_feat = feats[layer_idx]
            pooled_feat = torch.nn.functional.adaptive_avg_pool2d(layer_feat, (1, 1)).flatten(1)
            features.append(pooled_feat.cpu().numpy())
            labels.append(lbls.numpy())
    return np.vstack(features), np.concatenate(labels)

def run_linear_probing():
    data_dir = get_mri_dataset_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, val_tf = get_transforms()
    
    train_loader = DataLoader(BrainMRIDataset(os.path.join(data_dir, "Training"), transform=train_tf), batch_size=64, shuffle=False)
    val_loader = DataLoader(BrainMRIDataset(os.path.join(data_dir, "Testing"), transform=val_tf), batch_size=64, shuffle=False)

    model = ResNetSEBrainTumor(num_classes=4).to(device)
    model.load_state_dict(torch.load("./checkpoints/best_model.pth", map_location=device))

    print("\n--- Running Linear Depth Probing Across Layers ---")
    layer_accs = []
    for layer_idx in range(4):
        X_train, y_train = extract_features_at_layer(model, train_loader, layer_idx, device)
        X_val, y_val = extract_features_at_layer(model, val_loader, layer_idx, device)

        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val)) * 100
        layer_accs.append(acc)
        print(f"Layer {layer_idx + 1} Linear Probing Acc: {acc:.2f}%")

    plt.figure(figsize=(6, 4))
    plt.plot([1, 2, 3, 4], layer_accs, marker='o', color='crimson')
    plt.title("Representation Separability across Network Depth")
    plt.xlabel("ResNet Layer Depth")
    plt.ylabel("Linear Probe Accuracy (%)")
    plt.grid(True)
    plt.savefig("./depth_probing.png")
    plt.close()

if __name__ == "__main__":
    run_linear_probing()
