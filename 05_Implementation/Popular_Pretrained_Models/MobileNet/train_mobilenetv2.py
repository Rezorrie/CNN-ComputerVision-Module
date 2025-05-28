import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_mobilenetv2 import build_mobilenetv2_transfer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=8)
    args = parser.parse_args()

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(args.data_dir, target_size=(224,224),
                                            batch_size=args.batch_size, subset='training')
    val_gen   = datagen.flow_from_directory(args.data_dir, target_size=(224,224),
                                            batch_size=args.batch_size, subset='validation')

    model = build_mobilenetv2_transfer()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=args.epochs)
    model.save('mobilenetv2_transfer.h5')
