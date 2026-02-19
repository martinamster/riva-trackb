# Riva Cytology Challenge - Track b

This project implements a Deep Learning pipeline for cancer cell detection, combining object detection (**YOLOv8**) and segmentation-based localization (**U-Net**) through heatmap regression.

The system uses an ensemble approach to maximize detection accuracy:
* **YOLOv8 (20x20):** 
* **YOLOv8 (50x50):** 
* **U-Net (ResNet34):** 

## 📁 File Structure

* `run_train.py`: Main entry point to prepare data and trigger training sessions.
* `main.py`: Inference script to load models and generate predictions.
* `train.py`: Training logic for both YOLO and U-Net architectures.
* `preparation.py`: Utilities to convert CSV annotations to YOLO format and normalize labels.
* `inference.py`: Helper functions for prediction and post-processing.

AGREGAR ACA DECIR COMO TIENEN QUE ESTAR LOS ARCHIVOS QUE PASAN COMO PARAMETROS. EN MAIN.PY TIENEN QUE ESTAR DIRECATMENTE COMO SE DESCARGAN DE LA PAGINA DEL CHALLENGE. EN RUN_TRAIN.PY HAY QUE BORRAR LAS CARPETAS INTERMEDIAS
