import os 
import random
import shutil

source_dir = "data/processed/YawDD"
output_dir = "data/processed/YawDD/to-label"

dataset_size = 300
all_imgs = [img for img in os.listdir(source_dir) if img.endswith(".jpg")]
sampled_imgs = random.sample(all_imgs, dataset_size)

for img in sampled_imgs:
    shutil.copy(os.path.join(source_dir, img), os.path.join(output_dir, img))