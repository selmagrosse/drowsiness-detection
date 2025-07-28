import os
import numpy as np
import cv2 as cv

# Process raw data from the DDD Kaggle dataset

# Path of the directory with raw images
raw_dir = 'data/raw/DDD-kaggle/'
folders = ['Open_Eyes', 'Closed_Eyes']

# Path of the directory where processed images will be stored
processed_dir = 'data/processed/DDD-kaggle/'

# Mean and std of ImageNet will be used to normalize the images
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

images = []
labels = []

def process_image(img_path):
    img = cv.imread(img_path)
    if img is None:
        print(f"Error loading {img_path}.")
        return None
    # Resize image to 224 x 224 as this image size is expected by ResNet or MobileNet which will be used for image classification
    img = cv.resize(img, (224, 224))
    # cv2 uses the BGR color format, so we convert it to the RGB format
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # Normalize the image values to range [0, 1]
    img = img / 255.0
    img = (img - mean) / std
    return img


# Go through all the images in the dataset and create labels
for label, folder in enumerate(folders):
    folder_path = os.path.join(raw_dir, folder)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        img = process_image(file_path)
        if img is not None:
            images.append(img)
            labels.append(label)

# Convert images and labels to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Save processed images and labels
np.save(os.path.join(processed_dir, "images.npy"), images)
np.save(os.path.join(processed_dir, "labels.npy"), labels)


    