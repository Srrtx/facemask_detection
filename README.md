# Real-Time Face Mask Detection

This project implements a real-time face mask detection system using a trained deep learning model and computer vision techniques.

## Dataset
# Face Mask Detection COVID-19 Dataset for training Face Mask Classifier
The dataset has a total of 14535 images. It can be primarily used to train face mask classifiers.
It consists of 3 classes.
a) Incorrect mask b)With mask C) without mask.
shiekhburhan/face-mask-dataset

## Features

- Real-time face detection using MediaPipe
- Mask classification (with_mask, without_mask, incorrect_mask)
- Bounding boxes with color-coded predictions
- Confidence scores for each prediction
- Screenshot functionality

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the trained model file `face_mask.h5` in the project directory.

## Usage

Run the real-time detection:
```bash
python realtime_mask_detection.py
```

## Controls

- **'q'**: Quit the application
- **'s'**: Save a screenshot of the current frame

## Model Classes

The model predicts three classes:
- **with_mask** (Green): Person is wearing a mask correctly
- **without_mask** (Red): Person is not wearing a mask
- **incorrect_mask** (Orange): Person is wearing a mask incorrectly

## Requirements

- Python 3.7+
- Webcam or camera device
- Trained model file (`face_mask.h5`)

## Troubleshooting

1. **Camera not working**: Make sure your camera is not being used by another application
2. **Model not found**: Ensure `face_mask.h5` is in the same directory as the script
3. **Performance issues**: Try reducing the camera resolution or frame rate

