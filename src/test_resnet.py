import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Load test dataset
X_test = np.load("data/processed/DDD-kaggle/test_images.npy")
y_test = np.load("data/processed/DDD-kaggle/test_labels.npy")
# Reshape y_test (num_test_examples,) to have the same shape as model predictions in tf (num_test_examples, 1)
y_test = y_test.reshape(-1, 1)

# Load the updated resnet model
model = load_model("models/resnet/finetuned.h5")

# Evaluate on the test dataset
results = model.evaluate(X_test, y_test, batch_size=32, verbose=1)

# Predict labels
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Open", "Closed"]))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))