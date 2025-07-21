import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
images_dir = "data/processed/YawDD/images-sample"
labels_dir = "data/processed/YawDD/labels-sample"
output_dir = "data/processed/YawDD"

def labeld_or_empty(image_list, labels_dir):
    '''For each image in image_list check its corresponding label'''
    present = []
    for img in image_list:
        label_path = os.path.join(labels_dir, os.path.splitext(img)[0] + ".txt")
        with open(label_path, "r") as f:
            label = f.read().strip()
            present.append(1 if label else 0)

    return present

# Get all image filenames
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
# Get presence of positive labels in the entire dataset
all_images_labels = labeld_or_empty(image_files, labels_dir)
print(f"All images: Total={len(all_images_labels)}, Positives={sum(all_images_labels)}")

# Stratified split
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42, stratify=all_images_labels)
# Get presence of positive labels in the train and val dataset
train_images_labels = labeld_or_empty(train_files, labels_dir)
val_images_labels = labeld_or_empty(val_files, labels_dir)
print(f"Train images: Total={len(train_images_labels)}, Positives={sum(train_images_labels)}")
print(f"Validation images: Total={len(val_images_labels)}, Positives={sum(val_images_labels)}")

train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)


# Copy image and label
def copy_data(files, split):
    img_out = os.path.join(output_dir, "images", split)
    lbl_out = os.path.join(output_dir, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for img in files:
        base = os.path.splitext(img)[0]
        shutil.copy(os.path.join(images_dir, img), os.path.join(img_out, img))
        label_path = os.path.join(labels_dir, base + ".txt")
        if not os.path.exists(label_path):
            with open(label_path, "w") as f:
                pass
        shutil.copy(label_path, os.path.join(lbl_out, base + ".txt"))

# Copy train and val files
copy_data(train_files, "train")
copy_data(val_files, "val")

