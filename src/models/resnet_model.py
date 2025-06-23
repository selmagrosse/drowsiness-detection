from tensorflow.keras.applications import ResNet50
from tensorflow.keras import models, layers, Sequential

def get_resnet50_model():
    # Load the ResNet50 model pretrained on ImageNet
    base_model = ResNet50(include_top=False)

    # Freeze all layers
    base_model.trainable = False

    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation="sigmoid")
    ])

    return model