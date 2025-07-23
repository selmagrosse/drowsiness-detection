import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from ultralytics import YOLO
import time

# Load the trained ResNet50 model ('finetune' version)
resnet_model = load_model("models/finetune.h5")
# Load YOLO model
yolo_model = YOLO("runs/detect/train2/weights/best.pt")

# Time thresholds in seconds
DROWSY_DURATION = 2.5   # seconds
TIRED_DURATION = 1.0

# Mean and std of ImageNet will be used to normalize the images
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

# Preprocess the image
def preprocess(img):
    img = cv.resize(img, (224, 224))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img / 255.0
    img = (img - mean) / std
    # Expand to (1, 224, 224, 3) size for the Keras model
    return np.expand_dims(img, axis=0)

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, 
                                  max_num_faces=1, 
                                  refine_landmarks=True, 
                                  min_detection_confidence=0.5, 
                                  min_tracking_confidence=0.5)

# Mediapipe eye landmarks [left corner, right corner]
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

# Time tracking
eyes_closed_start_time = None
yawn_start_time = None

# Start the webcam (to open default camera, pass 0)
capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Cannot open camera.")
    exit()

while True:
    ret, frame = capture.read()

    # If frame is read correctly, ret will be True
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    now = time.time()
    eyes_closed = False
    yawn_detected = False

    # Yawn detection with YOLO
    yolo_results = yolo_model.predict(source=frame, conf=0.5, iou=0.3, imgsz=640, verbose=False)
    yawn_detected = len(yolo_results[0].boxes) > 0
    # Draw the results on the frame
    annotated_frame = yolo_results[0].plot()

    # Open/closed eye detection and classification with MediaPipe + ResNet
    h, w, _ = frame.shape
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mediapipe_results = face_mesh.process(rgb_frame)

    if mediapipe_results.multi_face_landmarks:
        for face_landmarks in mediapipe_results.multi_face_landmarks:
            for eye_indices, label in zip([LEFT_EYE, RIGHT_EYE], ["Left", "Right"]):
                x1 = int(face_landmarks.landmark[eye_indices[0]].x * w)
                y1 = int(face_landmarks.landmark[eye_indices[0]].y * h)
                x2 = int(face_landmarks.landmark[eye_indices[1]].x * w)
                y2 = int(face_landmarks.landmark[eye_indices[1]].y * h)

                # Add padding
                pad = 20
                x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
                x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)

                eye_img = frame[y1p:y2p, x1p:x2p]
                if eye_img.size == 0:
                    continue
                try:
                    eye_img = preprocess(eye_img)
                    pred = resnet_model.predict(eye_img)
                    # label = "Closed" if pred > 0.5 else "Open"
                    eyes_closed = pred > 0.5

                    # Draw ractangle around the eyes and add label
                    cv.rectangle(annotated_frame, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
                    cv.putText(annotated_frame, f"{label}", (x1p, y1p - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    print("Error processing eye region:", e)

    if eyes_closed:
        if eyes_closed_start_time is None:
            eyes_closed_start_time = now
        eyes_closed_duration = now - eyes_closed_start_time
    else:
        eyes_closed_start_time = None
        eyes_closed_duration = 0
    if yawn_detected:
        if yawn_start_time is None:
            yawn_start_time = now
        yawn_duration = now - yawn_start_time
    else:
        yawn_start_time = None
        yawn_duration = 0

    # Combine predictions
    if eyes_closed_duration > DROWSY_DURATION and yawn_duration > DROWSY_DURATION:
        status = "Yawning and eyes closed: DROWSY"
        color = (0, 0, 255)
    elif yawn_duration > TIRED_DURATION:
        status = "Yawning: TIRED"
        color = (0, 165, 255)
    elif eyes_closed_duration > TIRED_DURATION:
        status = "Eyes closed: TIRED"
        color = (0, 165, 255)
    else:
        status = "ALERT"
        color = (0, 255, 0)

    # Display the resulting frame
    cv.putText(annotated_frame, f"Status: {status}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv.imshow('Drowsiness Detection', annotated_frame)
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture
capture.release()
cv.destroyAllWindows()





