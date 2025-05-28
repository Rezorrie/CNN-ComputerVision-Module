from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

def build_efficientnet_transfer(input_shape=(224, 224, 3), num_classes=10):
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=base.input, outputs=outputs)
