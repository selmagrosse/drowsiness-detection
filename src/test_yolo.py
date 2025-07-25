from ultralytics import YOLO

# Load your trained model
model = YOLO("models/yolo/best.pt")

# Run inference on your test images
results = model.predict(source="data/processed/YawDD//test-sample", conf=0.5, iou=0.3, save=True)