import cv2 as cv
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("runs/detect/train4/weights/best.pt")

# Open webcam (0 = default camera)
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, conf=0.25)

    # Draw the results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv.imshow("YOLOv8 Real-Time Detection", annotated_frame)
    print(results[0].boxes)

    # Press 'q' to quit
    if cv.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()




