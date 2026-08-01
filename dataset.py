import os
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def download_kaggle_dataset(dataset_slug="masoudnickparvar/brain-tumor-mri-dataset", download_dir="./data"):
    """
    Downloads and extracts dataset from Kaggle.
    Requires kaggle.json in ~/.kaggle/ or KAGGLE_USERNAME and KAGGLE_KEY set.
    """
    os.makedirs(download_dir, exist_ok=True)
    target_path = os.path.join(download_dir, "brain-tumor-mri-dataset")
    
    if not os.path.exists(target_path):
        print(f"Downloading Kaggle dataset: {dataset_slug}...")
        os.system(f"kaggle datasets download -d {dataset_slug} -p {download_dir}")
        zip_file = [f for f in os.listdir(download_dir) if f.endswith('.zip')][0]
        zip_path = os.path.join(download_dir, zip_file)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_path)
        os.remove(zip_path)
        print("Dataset setup complete.")
    else:
        print("Dataset already exists locally.")

def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filepaths = []
        self.labels = []

        for c in CLASSES:
            folder = os.path.join(root_dir, c)
            if not os.path.exists(folder):
                continue
            for img_name in os.listdir(folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.filepaths.append(os.path.join(folder, img_name))
                    self.labels.append(CLASS_TO_IDX[c])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path
