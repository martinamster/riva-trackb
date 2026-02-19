import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

def fix_dup_ids(df, output_csv):
    df['id'] = range(1, len(df) + 1)
    df.to_csv(output_csv, index=False)



def ensemble(df1, df2, maxDist, minConf):

  df1 = df1.sort_values(by=['image_filename', 'conf'], ascending=[True, False])
  df2 = df2.sort_values(by=['image_filename', 'conf'], ascending=[True, False])
  df_result = pd.DataFrame(columns=['id', 'image_filename', 'class', 'x', 'y', 'width', 'height', 'conf'])
  df_trash = pd.DataFrame(columns=['id', 'image_filename', 'class', 'x', 'y', 'width', 'height', 'conf'])

  nombres_imagenes = set(df1['image_filename']) | set(df2['image_filename'])
  for imagen in nombres_imagenes:
    df1_imagen = df1[df1['image_filename'] == imagen].reset_index(drop=True)
    df2_imagen = df2[df2['image_filename'] == imagen].reset_index(drop=True)

    for i in range(len(df1_imagen)):
      nearest_dist = 1000000
      nearest_id = -1
      for j in range(len(df2_imagen)):
        point1 = (df1_imagen.loc[i,'x'], df1_imagen.loc[i,'y'])
        point2 = (df2_imagen.loc[j,'x'], df2_imagen.loc[j,'y'])
        dist = math.dist(point1, point2)
        if dist<nearest_dist and dist<maxDist:
          nearest_dist = dist
          nearest_id = j

      if nearest_id!=-1:
        x = (df1_imagen.loc[i,'x'] + df2_imagen.loc[nearest_id,'x'])/2
        y = (df1_imagen.loc[i,'y'] + df2_imagen.loc[nearest_id,'y'])/2
        conf = (df1_imagen.loc[i,'conf'] + df2_imagen.loc[nearest_id,'conf'])/2
        df2_imagen.loc[nearest_id,'x'] = round(x, 2)
        df2_imagen.loc[nearest_id,'y'] = round(y,2)
        df2_imagen.loc[nearest_id,'conf'] = round(conf,4)
        df_result = pd.concat([df_result, pd.DataFrame([df2_imagen.iloc[nearest_id]])])
        df2_imagen = df2_imagen.drop(nearest_id)
        df2_imagen = df2_imagen.reset_index(drop=True)
      elif df1_imagen.loc[i,'conf'] >= minConf:
        df_result = pd.concat([df_result, pd.DataFrame([df1_imagen.iloc[i]])])
      else:
       df_trash = pd.concat([df_trash, pd.DataFrame([df1_imagen.iloc[i]])])

    for k in range(len(df2_imagen)):
      if df2_imagen.loc[k,'conf'] >= minConf:
        df_result = pd.concat([df_result, pd.DataFrame([df2_imagen.iloc[k]])])
      else:
        df_trash = pd.concat([df_trash, pd.DataFrame([df2_imagen.iloc[k]])])

  return df_result.reset_index(drop=True), df_trash.reset_index(drop=True)




def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    b1_x1, b1_y1, b1_x2, b1_y2 = x1-w1/2, y1-h1/2, x1+w1/2, y1+h1/2
    b2_x1, b2_y1, b2_x2, b2_y2 = x2-w2/2, y2-h2/2, x2+w2/2, y2+h2/2
    
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = (w1*h1) + (w2*h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0


def apply_nms(df, iou_thresh=0.75):
    
    
    df = df.sort_values(by='conf', ascending=False).reset_index(drop=True)
    
    true_keeps = []
    grouped = df.groupby('image_filename')
    
    for img_name, group in tqdm(grouped, desc="Applying NMS"):
        indices = group.index.tolist()
        boxes = group[['x', 'y', 'width', 'height']].values
        
        is_suppressed = [False] * len(indices)
        
        for i in range(len(indices)):
            if is_suppressed[i]:
                continue
            
            true_keeps.append(indices[i])
            
            for j in range(i + 1, len(indices)):
                if not is_suppressed[j]:
                    if calculate_iou(boxes[i], boxes[j]) > iou_thresh:
                        is_suppressed[j] = True

    df_final = df.loc[true_keeps].sort_values(by=['image_filename', 'conf'], ascending=[True, False])
    return df_final.reset_index(drop=True)
    


def dynamic_threshold_filter(df_full, grid_size,detections,img_width=1024, img_height=1024):
    
    filtered_dfs = []
    
    grouped = df_full.groupby('image_filename')
    
    for img_name, df_img in grouped:
        cell_w = img_width // grid_size
        cell_h = img_height // grid_size
        
        current_img_preds = []
        
        for grid_x in range(grid_size):
            for grid_y in range(grid_size):
                x1, y1 = grid_x * cell_w, grid_y * cell_h
                x2, y2 = x1 + cell_w, y1 + cell_h
                

                cell_mask = (
                    (df_img['x'] >= x1) & (df_img['x'] < x2) &
                    (df_img['y'] >= y1) & (df_img['y'] < y2)
                )
                cell_preds = df_img[cell_mask]
                
                if cell_preds.empty:
                    continue
                
                n_dets = len(cell_preds)
                
                if n_dets > detections: 
                    threshold = 0.1
                else:
                    threshold = 0.001
                
                keepers = cell_preds[cell_preds['conf'] >= threshold].copy()
                
                if keepers.empty and not cell_preds.empty:
                    merged_row = {
                        'image_filename': img_name,
                        'class': cell_preds['class'].iloc[0],
                        'x': cell_preds['x'].mean(),
                        'y': cell_preds['y'].mean(),
                        'width': cell_preds['width'].mean(),
                        'height': cell_preds['height'].mean(),
                        'conf': cell_preds['conf'].max() 
                    }
                    keepers = pd.DataFrame([merged_row])
                
                current_img_preds.append(keepers)
        
        if current_img_preds:
            filtered_dfs.append(pd.concat(current_img_preds))

    
    if filtered_dfs:
        df_result = pd.concat(filtered_dfs).reset_index(drop=True)
        df_result['id'] = range(len(df_result))
        return df_result
    else:
        return pd.DataFrame(columns=df_full.columns)




def load_model(model_path, device):
    model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def filter_predictions(df_full, model_path, images_dir, yolo_conf_upper, binary_threshold, img_size=299, device='cuda'):
    model = load_model(model_path, device)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Starting binary classification...")
    
    grouped = df_full.groupby('image_filename')
    final_rows = []

    for img_name, group in tqdm(grouped):
        if group['conf'].min() > yolo_conf_upper:
            for _, row in group.iterrows():
                final_rows.append(row.to_dict())
            continue

        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            for _, row in group.iterrows(): 
                final_rows.append(row.to_dict())
            continue

        img_pil = Image.open(img_path).convert("RGB")
        w_img, h_img = img_pil.size
        
        batch_crops = []
        batch_indices = [] 
        rows_list = group.to_dict('records')

        for i, row in enumerate(rows_list):
            if row['conf'] > yolo_conf_upper:
                final_rows.append(row)
            else:
                x, y, w, h = row['x'], row['y'], row['width'], row['height']
                left = max(0, int(x - w/2))
                upper = max(0, int(y - h/2))
                right = min(w_img, int(x + w/2))
                lower = min(h_img, int(y + h/2))
                
                crop = img_pil.crop((left, upper, right, lower))
                
                if crop.size[0] > 0 and crop.size[1] > 0:
                    batch_crops.append(transform(crop))
                    batch_indices.append(i)

        if batch_crops:
            # Internal mini-batching to prevent OutOfMemory errors
            MINI_BATCH_SIZE = 8
            all_probs = []

            for i in range(0, len(batch_crops), MINI_BATCH_SIZE):
                mini_batch = torch.stack(batch_crops[i : i + MINI_BATCH_SIZE]).to(device)
                with torch.no_grad():
                    outputs = model(mini_batch)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    all_probs.extend(probs)
                del mini_batch
                torch.cuda.empty_cache()
            
            for idx_in_batch, prob in enumerate(all_probs):
                original_row_idx = batch_indices[idx_in_batch]
                row = rows_list[original_row_idx]
                if prob > binary_threshold:
                    final_rows.append(row)

    return pd.DataFrame(final_rows)