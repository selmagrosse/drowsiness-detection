import os
import numpy as np
from sklearn.model_selection import train_test_split

processed_dir = 'data/processed/DDD-kaggle/'

# Load the processed dataset
images = np.load(os.path.join(processed_dir, "images.npy"))
labels = np.load(os.path.join(processed_dir, "labels.npy"))

'''Split the dataset into train+val (80%) and test datasets (20%). The data is automatically shuffled. 
stratify=labels: the generated splits gave the same proprotion of labels as given by parameter "labels".'''
images_train_val, images_test, labels_train_val, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

# Split the train_val dataset into train (75%) and val datasets (25%)
images_train, images_val, labels_train, labels_val = train_test_split(images_train_val, labels_train_val, test_size=0.25, random_state=42, stratify=labels_train_val)

print(f"Train set size: {images_train.shape[0]}")
print(f"Validation set size: {images_val.shape[0]}")
print(f"Test set size: {images_test.shape[0]}")

# Save split data and labels
np.save(os.path.join(processed_dir, "train_images.npy"), images_train)
np.save(os.path.join(processed_dir, "train_labels.npy"), labels_train)
np.save(os.path.join(processed_dir, "val_images.npy"), images_val)
np.save(os.path.join(processed_dir, "val_labels.npy"), labels_val)
np.save(os.path.join(processed_dir, "test_images.npy"), images_test)
np.save(os.path.join(processed_dir, "test_labels.npy"), labels_test)