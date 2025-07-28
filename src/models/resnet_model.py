from tensorflow.keras.applications import ResNet50
from tensorflow.keras import models, layers, Sequential

def get_resnet50_model(variant='base'):
    # Load the ResNet50 model pretrained on ImageNet
    base_model = ResNet50(include_top=False, input_shape=(224, 224, 3))
    train_from = 150

    if variant == 'finetuned':
        base_model.trainable = True
        # Freeze all layers before the 150th layer
        for layer in base_model.layers[:train_from]:
            layer.trainable = False
    else:
        # Freeze all layers
        base_model.trainable = False
    
    model_append = [layers.GlobalAveragePooling2D()]
    model_append.append(layers.Dense(1, activation='sigmoid'))

    model = Sequential([base_model] + model_append)

    return model