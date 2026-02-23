import argparse
import os
import torch
import segmentation_models_pytorch as smp
from ultralytics import YOLO
from inference import *
from preparation import *
from train import *
from post_process import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Pipeline for competence")
parser.add_argument("competition_path", type=str, help="Path to the competition data folder")
args = parser.parse_args()

data_path = args.competition_path

if not os.path.exists(data_path):
    print("Path not found.")
    quit()



model_20 = YOLO("models/best_20x20_yolo.pt")
prediction1 = predictYOLO(0.001, 0.3, model_20, data_path)

model_50 = YOLO('models/best_50x50_yolo.pt')
prediction2 = predictYOLO(0.001, 0.48, model_50, data_path)

unet_path = 'models/best_unet.pth'

prediction3 = run_unet_inference(unet_path, data_path, threshold=0.2, k_size=25, multiscale=True, device=DEVICE) 


df_final, _ = ensemble(prediction2, prediction1, 12, 0.35) 
df_final, _ = ensemble(df_final, prediction3, 12, 0)


df_nms = apply_nms(df_final, iou_thresh=0.75)
df_filter = dynamic_threshold_filter(df_nms, 4, detections=30) 

df_submission = filter_predictions(df_filter, 'models/garbage_classifier.pth', data_path, yolo_conf_upper=0.01, binary_threshold=0.05, img_size=224, device=DEVICE)
fix_dup_ids(df_submission, "submission.csv")