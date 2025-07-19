import os 
import random
import shutil

source_dir = "data/processed/YawDD"
output_dir = "data/processed/YawDD/test-sample"
train_val_dir = "data/processed/YawDD/images-sample"

existing_files = set(os.listdir(train_val_dir))

dataset_size = 100
# Get additional images that haven't been sampled yet
all_imgs = [img for img in os.listdir(source_dir) if img.endswith(".jpg")]
unsampled_imgs = [img for img in all_imgs if img not in existing_files]

sampled_imgs = random.sample(unsampled_imgs, dataset_size)

# Copy sampled images to the test-sample subfolder
for img in sampled_imgs:
    shutil.copy(os.path.join(source_dir, img), os.path.join(output_dir, img))