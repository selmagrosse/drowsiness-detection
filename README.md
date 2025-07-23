# Real-Time Drowsiness Detection

## Overview

This project implements real-time drowsiness detection by combining two deep learning models:

- **ResNet50** model for the classification of eye states - open or closed
- **YOLOv8** model for yawning detection

The model runs in real-time, and the camera can be used to monitor signs of drowsiness and tiredness, such as prolonged yawning and/or eye closure. If signs of drowsiness or tiredness are detected, an appropriate status alert is triggered.

## Motivation

Driver drowsiness is a significant contributor to accidents worldwide. Monitoring states of the driver and detecting states in which the driver is not fully alert can be useful to trigger the handover process to the vehicle.


## Models

### ResNet50 Eye State Classifier

- Architecture: ResNet50 fine-tuned for binary classification (eye open vs. closed)
- Framework: TensorFlow/Keras
- Integration with MediaPipe for robust eye region extraction from face landmarks


### YOLOv8 Yawning Detector

- Architecture: YOLOv8 nano (`yolov8n.pt`) fine-tuned on a custom yawning dataset
- Training: 150 epochs with data augmentation and stratified dataset splits
<!-- - Performance: Reduced false positives, reasonable precision-recall balance -->

<!-- ## Setup

### Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt -->
