import os 
import random
import shutil

source_dir = "data/processed/YawDD"
output_dir = "data/processed/YawDD/to-label"

dataset_size = 500
# Get all images found in the source_dir
all_imgs = [img for img in os.listdir(source_dir) if img.endswith(".jpg")]
# Randomly sample 500 images
sampled_imgs = random.sample(all_imgs, dataset_size)

# Copy sampled images in the output_dir
for img in sampled_imgs:
    shutil.copy(os.path.join(source_dir, img), os.path.join(output_dir, img))