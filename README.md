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
* `Models`: Pre-trained models.

## How to run
### Packages
First, install all the packages necessary for the project:
```
pip install -r requirements.txt
``` 
### Model training
Training the U-net and YOLO 100x100:
```
python3 run_train.py route_to_folder
```
Where route_to_folder is the route on your device where you have the training data. The training data should be organized exactly like this:

route_to_folder/ \
├── annotations/ \
│   ├── train.csv  \        
│   └── val.csv    \       
└── images/    \
    ├── train/      \           
    └── val/              
This training is an example, the parameters may be changed. For the competition we trained the YOLO models with different box sizes

### Binary classificator training
If you want to train the binary classificator using the train part of the dataset:
```
python3 train_classifier.py route_to_folder
```
Using the same file organization as before

### Replicating the results of the competition
The following command produces a file submission.csv identical to our submission to the challenge.
```
python3 main.py route_to_test_images
```
If you replace route_to_test_images with the route to the images of testing downloaded from kaggle it will produce the csv file.
