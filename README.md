# Riva Cytology Challenge - Track b

This project implements a Deep Learning pipeline for cancer cell detection, combining object detection (**YOLOv8**) and segmentation-based localization (**U-Net**) through heatmap regression.

The system uses an ensemble approach to maximize detection accuracy:
* **YOLOv8 nano (20x20)** 
* **YOLOv8 nano (50x50)** 
* **U-Net (ResNet34)** 

## 📁 File Structure

* `run_train.py`: Main entry point to prepare data and trigger training sessions.
* `main.py`: Script to generate the exact predictions for the challenge. Can be used for other dataset but it's optimized for Riva.
* `train.py`: Training logic for both YOLO and U-Net architectures.
* `preparation.py`: Utilities to convert CSV annotations to YOLO format and normalize labels.
* `inference.py`: Helper functions for prediction for both YOLO and U-net.
* `post_process.py`: Functions to optimize the results post training.
* `train_classifier.py`: All the code necessary to train the binary (cell/garbage) classifier
* `requirements.txt`: Packages needed for the project.

## How to run
### Packages
First, install all the packages necessary for the project:
```
pip install -r requirements.txt
``` 
### Model training
Training the U-net and YOLO 100x100:
```

```
