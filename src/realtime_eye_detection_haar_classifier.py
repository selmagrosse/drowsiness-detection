import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained ResNet50 model ('finetuned' version)
model = load_model("models/finetuned.h5")

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
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(grayscale)

    for (x, y, w, h) in face:
        face_roi = frame[y:y+h, x:x+w]
        # Draw rectangle around the face
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        eyes = eye_cascade.detectMultiScale(grayscale[y:y+h, x:x+w])

        for (ex, ey, ew, eh) in eyes:
            eye_img = face_roi[ey:ey+eh, ex:ex+ew]
            try:
                eye_img = preprocess(eye_img)
                pred = model.predict(eye_img)
                label = "Closed" if pred > 0.5 else "Open"

                # Draw ractangle around the eyes and add label
                cv.rectangle(face_roi, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                cv.putText(face_roi, label, (ex, ey - 10),  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                print("Error processing eye region:", e)

    # Display the resulting frame
    cv.imshow('Fatigure Detection', frame)
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture
capture.release()
cv.destroyAllWindows()





