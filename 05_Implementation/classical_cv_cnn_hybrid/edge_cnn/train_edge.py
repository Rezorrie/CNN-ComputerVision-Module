import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from model_edge import build_edge_cnn


def load_data_with_edges(data_dir, img_size=(28, 28)):
    X, y = [], []
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        for fname in os.listdir(label_dir):
            path = os.path.join(label_dir, fname)
            img = load_img(path, color_mode='grayscale', target_size=img_size)
            arr = img_to_array(img)
            edges = cv2.Canny(arr.astype('uint8'), 100, 200)
            edges = edges.reshape((*img_size, 1))
            fused = np.concatenate([arr/255.0, edges/255.0], axis=-1)
            X.append(fused)
            y.append(int(label))
    X = np.array(X)
    y = to_categorical(y)
    return X, y

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output_model', default='edge_cnn.h5')
    args = parser.parse_args()

    X, y = load_data_with_edges(args.data_dir)
    model = build_edge_cnn(input_shape=X.shape[1:], num_classes=y.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.2)
    model.save(args.output_model)
