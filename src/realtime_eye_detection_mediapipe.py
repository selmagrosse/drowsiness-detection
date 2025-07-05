import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained ResNet50 model ('finetune' version)
model = load_model("models/finetune.h5")

# Mean and std of ImageNet will be used to normalize the images
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

# Load Haar cascade classifiers
face_cascade = cv.CascadeClassifier("models/haar_cascade_classifiers/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("models/haar_cascade_classifiers/haarcascade_eye.xml")

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
    h, w, _ = frame.shape
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
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
                    pred = model.predict(eye_img)
                    label = "Closed" if pred > 0.5 else "Open"

                    # Draw ractangle around the eyes and add label
                    cv.rectangle(frame, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
                    cv.putText(frame, f"{label}", (x1p, y1p - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    print("Error processing eye region:", e)

    # Display the resulting frame
    cv.imshow('Fatigure Detection', frame)
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture
capture.release()
cv.destroyAllWindows()





