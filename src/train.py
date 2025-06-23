import numpy as np
import datetime
from models.resnet_model import get_resnet50_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC, F1Score
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

# Load train dataset
X_train = np.load("data/processed/DDD-kaggle/train_images.npy")
y_train = np.load("data/processed/DDD-kaggle/train_labels.npy")
# Reshape y_train (num_train_examples,) to have the same shape as model predictions in tf (num_train_examples, 1)
y_train = y_train.reshape(-1, 1)

# Load validation dataset
X_val = np.load("data/processed/DDD-kaggle/val_images.npy")
y_val = np.load("data/processed/DDD-kaggle/val_labels.npy")
# Reshape y_val (num_val_examples,) to have the same shape as model predictions in tf (num_val_examples, 1)
y_val = y_val.reshape(-1, 1)

# Load ResNet50 model
model = get_resnet50_model()

# Configure model settings for training
model.compile(optimizer=Adam(learning_rate=1e-4), loss=BinaryCrossentropy(), metrics=[Accuracy(), Precision(), Recall(), AUC(), F1Score()])

# Callbacks
checkpoint = ModelCheckpoint("models/resnet50_modified.h5", save_best_only=True)
log_dir = "logs/fit/" #+ datetime.datetime.now().to_str()
tensorboard = TensorBoard(log_dir=log_dir)

# Train modified ResNet50
history = model.fit(X_train, y_train, validation_data=[X_val, y_val], batch_size=32, epochs=10, callbacks=[checkpoint, tensorboard])