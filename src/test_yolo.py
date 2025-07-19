from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train4/weights/best.pt")

# Run inference on your test images
results = model.predict(source="data/processed/YawDD//test-sample", save=True)