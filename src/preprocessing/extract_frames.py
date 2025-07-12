import os
import cv2 as cv
import uuid

source_dir = "data/raw/YawDD"
processed_dir = "data/processed/YawDD"

# Extract every 10th frame
frame_count = 10

# Traverse all videos
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.avi'):
            video_path = os.path.join(root, file)
            cap = cv.VideoCapture(video_path)
            count = 0
            frame_id = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if count & frame_count == 0:
                    filename = f"{os.path.splitext(file)[0]}_{frame_id}.jpg"
                    cv.imwrite(os.path.join(processed_dir, filename), frame)
                    frame_id += 1
                count += 1
            cap.release()

