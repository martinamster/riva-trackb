import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

class CellTestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.img_names = [os.path.basename(x) for x in self.images]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        orig_size = image.shape[:2]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, self.img_names[idx], torch.tensor(orig_size)

def _predict_multi_scale(model, image_tensor, device, scales=[0.8, 1.0, 1.2]):
    b, c, h, w = image_tensor.shape
    final_output = torch.zeros((b, 1, h, w), device=device)
    
    for scale in scales:
        target_h, target_w = int(h * scale), int(w * scale)
        scaled_img = F.interpolate(image_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            output = torch.sigmoid(model(scaled_img))
        
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
        final_output += output
        
    final_output /= len(scales)
    return final_output

def _get_coords_from_heatmap(heatmap, threshold, k_size):
    heatmap = heatmap.float()
    padding = k_size // 2
    hmax = F.max_pool2d(heatmap, kernel_size=k_size, stride=1, padding=padding)
    keep = (hmax == heatmap) & (heatmap > threshold)
    indices = torch.nonzero(keep, as_tuple=False)
    
    detections = []
    for idx in indices:
        batch, chan, y, x = idx.tolist()
        conf = heatmap[batch, chan, y, x].item()
        detections.append({'x': x, 'y': y, 'conf': conf})
    return detections

def run_unet_inference(model_path, img_dir, threshold=0.2, k_size=25, multiscale=True, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights=None, 
        in_channels=3, 
        classes=1
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    test_transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    test_ds = CellTestDataset(img_dir, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)
    
    results = []
    
    with torch.no_grad():
        for images, names, orig_sizes in tqdm(test_loader): 
            images = images.to(device)
            
            if multiscale:
                preds = _predict_multi_scale(model, images, device, scales=[0.8, 1.0, 1.2])
            else:
                preds = torch.sigmoid(model(images))
            
            for i in range(len(images)):
                dets = _get_coords_from_heatmap(preds[i:i+1], threshold=threshold, k_size=k_size)
                
                h_orig, w_orig = orig_sizes[i][0].item(), orig_sizes[i][1].item()
                scale_x = w_orig / 1024.0
                scale_y = h_orig / 1024.0
                
                for d in dets:
                    results.append({
                        "image_filename": names[i],
                        "class": 0,
                        "x": d['x'] * scale_x,
                        "y": d['y'] * scale_y,
                        "width": 100, 
                        "height": 100,
                        "conf": d['conf']
                    })

    df = pd.DataFrame(results)
    if not df.empty:
        df.insert(0, 'id', range(len(df)))
        return df[["id", "image_filename", "class", "x", "y", "width", "height", "conf"]]
    return pd.DataFrame()



def predictYOLO(conf_param, iou_param, model, image_path):
    
    results = model.predict(
        source=image_path, 
        conf=conf_param, 
        imgsz=1024, 
        augment=True, 
        iou=iou_param, 
        max_det=3000, 
        stream=True,  
        verbose=False 
    )
    
    data_csv = []
    counter_id = 0
    
    for result in results:
        nombre_archivo = os.path.basename(result.path)
        
        boxes = result.boxes.cpu().numpy()
        
        if len(boxes) == 0: continue
            
        squares = boxes.xywh
        classes = boxes.cls
        confs = boxes.conf
    
        for i in range(len(squares)):
            x, y, w, h = squares[i]
            
            data_csv.append({
                'id': counter_id,
                'image_filename': nombre_archivo,
                'class': int(classes[i]),
                'x': x,
                'y': y,
                'width': 100, 
                'height': 100,
                'conf': confs[i]
            })
            counter_id += 1
    
    df = pd.DataFrame(data_csv)
    
    if not df.empty:
        sorted_cols = ['id', 'image_filename', 'class', 'x', 'y', 'width', 'height', 'conf']
        df = df[sorted_cols]
        return df
    else:
        return pd.DataFrame(columns=['id', 'image_filename', 'class', 'x', 'y', 'width', 'height', 'conf'])