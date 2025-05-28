import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_efficientnet import build_efficientnet_transfer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(args.data_dir, target_size=(224,224),
                                            batch_size=args.batch_size, subset='training')
    val_gen   = datagen.flow_from_directory(args.data_dir, target_size=(224,224),
                                            batch_size=args.batch_size, subset='validation')

    model = build_efficientnet_transfer()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=args.epochs)
    model.save('efficientnet_transfer.h5')
