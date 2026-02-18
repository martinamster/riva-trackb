import pandas as pd
import os
from PIL import Image
from tqdm import tqdm 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_running_folder_with_cage_size(width, height, csv_path, images_path, output_path):
    CSV_PATH = csv_path
    IMAGES_PATH = images_path
    OUTPUT_PATH = output_path

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    df = pd.read_csv(CSV_PATH)
    df['width'] = width
    df['height'] = height
    unique_images = df['image_filename'].unique()
    
    for image_name in tqdm(unique_images):
        
        image_data = df[df['image_filename'] == image_name]
        
        ruta_img = os.path.join(IMAGES_PATH, image_name)
        
        try:
            with Image.open(ruta_img) as img:
                img_width, img_height = img.size
        except FileNotFoundError:
            print(f"{image_name}: File not found.")
            continue 
      
        lines_txt = []
        
        for _, row_im in image_data.iterrows():
    
            x_center_raw = row_im['x']
            y_center_raw = row_im['y']
            w_box = row_im['width']
            h_box = row_im['height']
            class_type = 0 
    
            x_norm = x_center_raw / img_width
            y_norm = y_center_raw / img_height
            w_norm = w_box / img_width
            h_norm = h_box / img_height
            
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))
            
            line = f"{class_type} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
            lines_txt.append(line)
        
        nombre_txt = image_name.replace('.png', '.txt').replace('.jpg', '.txt')
        ruta_txt = os.path.join(OUTPUT_PATH, nombre_txt)
        
        with open(ruta_txt, 'w') as f:
            f.write('\n'.join(lines_txt))
