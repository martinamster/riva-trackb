import argparse
import os
import shutil
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from inference import predictYOLO, run_unet_inference
from post_process import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── CONFIG ────────────────────────────────────────────────────────────────────
YOLO20_PATH   = 'models/best_20x20_yolo.pt'
YOLO50_PATH   = 'models/best_50x50_yolo.pt'
UNET_PATH     = 'models/best_unet.pth'
OUTPUT_DIR    = 'binary_dataset'
MODEL_OUT     = 'garbage_classifier.pth'

IOU_THRESHOLD_FP  = 0.1    
CONF_THRESHOLD_FP = 0.001  
IMG_SIZE          = 224   
BATCH_SIZE        = 64
EPOCHS            = 30
LR                = 0.001
# ───────────────────────────────────────────────────────────────────────────────


def _iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    b1_x1, b1_y1, b1_x2, b1_y2 = x1-w1/2, y1-h1/2, x1+w1/2, y1+h1/2
    b2_x1, b2_y1, b2_x2, b2_y2 = x2-w2/2, y2-h2/2, x2+w2/2, y2+h2/2
    inter_x1, inter_y1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    inter_x2, inter_y2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = (w1*h1) + (w2*h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0


def generate_train_predictions(images_dir):

    model_20 = YOLO(YOLO20_PATH)
    pred1 = predictYOLO(0.001, 0.3,  model_20, images_dir)

    model_50 = YOLO(YOLO50_PATH)
    pred2 = predictYOLO(0.001, 0.48, model_50, images_dir)

    pred3 = run_unet_inference(UNET_PATH, images_dir, threshold=0.2, k_size=25,
                               multiscale=True, device=DEVICE)

    df, _ = ensemble(pred2, pred1, 12, 0.3)
    df, _ = ensemble(df, pred3, 12, 0)
    df    = apply_nms(df, iou_thresh=0.75)

    return df


def prepare_binary_dataset(gt_csv, df_pred, images_dir, output_dir=OUTPUT_DIR):

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(os.path.join(output_dir, '0_garbage'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '1_cell'),    exist_ok=True)

    df_gt = pd.read_csv(gt_csv)
    images = set(df_gt['image_filename']) | set(df_pred['image_filename'])
    count_cells, count_garbage = 0, 0

    for img_name in tqdm(images, desc="Generating crops"):
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        gts = df_gt[df_gt['image_filename'] == img_name]
        gt_boxes = []
        for idx, row in gts.iterrows():
            x, y, w, h = row['x'], row['y'], row['width'], row['height']
            gt_boxes.append([x, y, w, h])
            x1 = max(0, int(x - w/2)); y1 = max(0, int(y - h/2))
            x2 = min(w_img, int(x + w/2)); y2 = min(h_img, int(y + h/2))
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                cv2.imwrite(f"{output_dir}/1_cell/{img_name}_{idx}.jpg", crop)
                count_cells += 1

        preds = df_pred[df_pred['image_filename'] == img_name]
        for idx, row in preds.iterrows():
            if row['conf'] < CONF_THRESHOLD_FP:
                continue
            p_box = [row['x'], row['y'], row['width'], row['height']]
            max_iou = max((_iou(p_box, g) for g in gt_boxes), default=0)
            if max_iou < IOU_THRESHOLD_FP:
                x, y, w, h = row['x'], row['y'], row['width'], row['height']
                x1 = max(0, int(x - w/2)); y1 = max(0, int(y - h/2))
                x2 = min(w_img, int(x + w/2)); y2 = min(h_img, int(y + h/2))
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    cv2.imwrite(f"{output_dir}/0_garbage/{img_name}_fp_{idx}.jpg", crop)
                    count_garbage += 1

    print(f"\nDataset ready — Cells: {count_cells} | Garbage: {count_garbage}")
    return count_cells, count_garbage


class _BinaryDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels     = labels
        self.transform  = transform

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label


def train_garbage_classifier(data_dir=OUTPUT_DIR, model_save_path=MODEL_OUT):
    garbage_files = glob(os.path.join(data_dir, '0_garbage', '*.jpg'))
    cell_files    = glob(os.path.join(data_dir, '1_cell',    '*.jpg'))
    all_files     = garbage_files + cell_files
    all_labels    = [0] * len(garbage_files) + [1] * len(cell_files)

    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(_BinaryDataset(train_files, train_labels, train_tf),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(_BinaryDataset(val_files,   val_labels,   val_tf),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(DEVICE)

    pos_weight = torch.tensor([len(garbage_files) / max(len(cell_files), 1)]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.Adam(model.parameters(), lr=LR)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

    best_acc = 0.0
    print(f"Training garbage classifier on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).unsqueeze(1)
                predicted = (torch.sigmoid(model(imgs)) > 0.2).float()
                total   += lbls.size(0)
                correct += (predicted == lbls).sum().item()

        val_acc = correct / total
        print(f"Epoch {epoch+1} — Loss: {train_loss/len(train_loader):.4f} — Val Acc: {val_acc:.4f}")
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  Model saved (Acc: {best_acc:.4f})")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train garbage classifier")
    parser.add_argument("competition_path", type=str, help="Path to competition data folder")
    args = parser.parse_args()

    if not os.path.exists(args.competition_path):
        print("Path not found.")
        quit()

    train_csv     = os.path.join(args.competition_path, "annotations", "train.csv")
    train_img_dir = os.path.join(args.competition_path, "images", "train")

    df_preds = generate_train_predictions(train_img_dir)
    prepare_binary_dataset(train_csv, df_preds, train_img_dir)
    train_garbage_classifier()
