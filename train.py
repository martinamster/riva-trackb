import os
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob
from tqdm import tqdm


def train_yolo(dataset_path, model_name):
    yaml_content = (
        f"path: {dataset_path}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['celula_cancerosa']"
    )
    yaml_file_path = 'data.yaml'
        
    with open(yaml_file_path, 'w') as f:
        f.write(yaml_content.strip())

    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data=yaml_file_path,
        epochs=150,
        imgsz=1024,
        batch=16,
        patience=50,
        name=model_name
    )
    
    return results, model





class CellHeatmapDataset(Dataset):
    def __init__(self, base_dir, split="train", img_size=1024, transform=None):
        self.img_dir = os.path.join(base_dir, "images", split)
        self.lbl_dir = os.path.join(base_dir, "labels", split)
        self.transform = transform
        self.images = sorted(glob(os.path.join(self.img_dir, "*.png")))
        self.img_names = [os.path.basename(x) for x in self.images]

    def __len__(self):
        return len(self.images)

    def _generate_heatmap(self, width, height, label_path):
        heatmap = np.zeros((height, width), dtype=np.float32)
        if not os.path.exists(label_path): return heatmap
        
        with open(label_path, 'r') as f: 
            lines = f.readlines()

        box_ratio = 100.0 / 1024.0 
        current_box_size = width * box_ratio 
        sigma = max(1, current_box_size / 6.0)

        k_size = int(sigma * 6)
        if k_size % 2 == 0: k_size += 1
        
        x = np.arange(0, k_size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = k_size // 2
        g = np.exp(- ((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

        for line in lines:
            parts = list(map(float, line.strip().split()))
            cx, cy = int(parts[1] * width), int(parts[2] * height)
            
            x_min, x_max = cx - k_size//2, cx + k_size//2 + 1
            y_min, y_max = cy - k_size//2, cy + k_size//2 + 1
            
            g_x_min, g_y_min = int(max(0, -(cx - k_size//2))), int(max(0, -(cy - k_size//2)))
            t_w, t_h = min(width, x_max) - max(0, x_min), min(height, y_max) - max(0, y_min)
            
            if t_w > 0 and t_h > 0:
                h_y, h_x = max(0, y_min), max(0, x_min)
                heatmap[h_y:h_y+t_h, h_x:h_x+t_w] = np.maximum(heatmap[h_y:h_y+t_h, h_x:h_x+t_w], g[g_y_min:g_y_min+t_h, g_x_min:g_x_min+t_w])
        return heatmap

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        h_orig, w_orig = image.shape[:2]
        mask = self._generate_heatmap(w_orig, h_orig, os.path.join(self.lbl_dir, os.path.splitext(self.img_names[idx])[0] + ".txt"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
            
        mask = torch.from_numpy(mask).float().unsqueeze(0) if not isinstance(mask, torch.Tensor) else mask.float()
        return image, mask


def train_unet(base_dir, epochs=15, batch_size=4, lr=1e-4, img_size=1024):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_ds = CellHeatmapDataset(base_dir, split="train", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f" Training U-Net on {device}...")
    for epoch in range(epochs):
        model.train()
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, masks = images.to(device), masks.to(device)
            outputs = torch.sigmoid(model(images))
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), "best_unet_trained.pth")
    return model

