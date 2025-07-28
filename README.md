# Real-Time Drowsiness Detection

## Overview

This project implements real-time drowsiness detection by combining two deep learning models:

- A **ResNet50** model for eye state classification (open or closed)
- A **YOLOv8** model for yawning detection

The models work together with live video feed from a webcam to monitor signs of fatigue such as prolonged eye closure and/or frequent yawning. When such signs are detected, the system triggers a real-time status alert.

## Motivation

Driver drowsiness is a significant contributor to accidents worldwide. Detecting early signs of fatigue—like closing eyes or yawning—can help mitigate risks by prompting appropriate interventions (e.g., handing over control in semi-autonomous driving)

## Datasets

- Eye state classification:
Dataset: [Drowsiness Detection Dataset](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset)
Contains labeled images of open and closed eyes.

- Yawning detection:
Dataset: [YawDD - Yawning Detection Dataset](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset)
Contains videos recorded inside vehicles.

## Models

### ResNet50 Eye State Classifier

- Framework: TensorFlow/Keras
- Base model: ResNet50 pretrained on ImageNet
- Variants in this project:
  1. Base: global average pooling + dense layer with sigmoid activations
  2. Finetuned: same as base + the network is re-trained from its 150th layer onward
- The eye state classifier is integrated with MediaPipe for eye region extraction from face landmarks.

### YOLOv8 Yawning Detector

- Framework: Ultralytics YOLOv8
- Architecture: YOLOv8 nano (`yolov8n.pt`)
- Training: 150 epochs with data augmentation and stratified dataset splits

The trained models can be downloaded from [HuggingFace](https://huggingface.co/selmagrosse/drowsiness_detection/tree/main).

## Additional Resources

- For full evaluation details and results, please refer to the Kaggle scripts available in the `notebooks/` directory: [Eye Classification Notebook](https://www.kaggle.com/code/selmagrosse/drowsiness-detection-eye-classification) and [Yawning Detection Notebook] (https://www.kaggle.com/code/selmagrosse/yawn-detection-yolo/edit/run/252056062)

## Setup

### Requirements

1. Create a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```
2. Install the required packages:
   
```bash
pip install -r requirements.txt
```

3. Download the trained models and store them in the `models/` subfolder. 

## Usage

To start real-time drowsiness detection using a webcam:

```bash
python src/realtime/realtime_drowsiness_detection.py
```
