import argparse
import os
from preparation import create_running_folder_with_cage_size
from train import train_unet, train_yolo

parser = argparse.ArgumentParser(description="Training Pipeline")
parser.add_argument("competition_path", type=str)
args = parser.parse_args()

root = args.competition_path
if not os.path.exists(root):
    print("Path not found.")
    quit()

train_csv = os.path.join(root, "annotations", "train.csv")
val_csv = os.path.join(root, "annotations", "val.csv")

train_img_dir = os.path.join(root, "images", "train")
val_img_dir = os.path.join(root, "images", "val")

train_lbl_dir = os.path.join(root, "labels", "train")
val_lbl_dir = os.path.join(root, "labels", "val")

os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

print("Processing labels...")
create_running_folder_with_cage_size(100, 100, train_csv, train_img_dir, train_lbl_dir)
create_running_folder_with_cage_size(100, 100, val_csv, val_img_dir, val_lbl_dir)

print("Training U-Net...")
train_unet(root, epochs=15, batch_size=4, lr=1e-4, img_size=1024)

print("Training YOLO...")
train_yolo(root)

""""
This is just an example of how to run the training functions. It can be modified as needed
For example, in the competition we trained two YOLO models with different cage sizes and also a U-Net model.
"""