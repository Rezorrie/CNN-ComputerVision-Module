import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import VGG16

def build_color_hist_cnn(input_shape=(224, 224, 3), hist_bins=32, num_classes=10):
    # CNN backbone
    base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = layers.Flatten()(base.output)

    # Histogram input
    hist_input = Input(shape=(hist_bins * 3,), name='hist_input')

    # Fusion
    fusion = layers.Concatenate()([x, hist_input])
    dense = layers.Dense(256, activation='relu')(fusion)
    outputs = layers.Dense(num_classes, activation='softmax')(dense)

    model = models.Model(inputs=[base.input, hist_input], outputs=outputs)
    base.trainable = False
    return model
