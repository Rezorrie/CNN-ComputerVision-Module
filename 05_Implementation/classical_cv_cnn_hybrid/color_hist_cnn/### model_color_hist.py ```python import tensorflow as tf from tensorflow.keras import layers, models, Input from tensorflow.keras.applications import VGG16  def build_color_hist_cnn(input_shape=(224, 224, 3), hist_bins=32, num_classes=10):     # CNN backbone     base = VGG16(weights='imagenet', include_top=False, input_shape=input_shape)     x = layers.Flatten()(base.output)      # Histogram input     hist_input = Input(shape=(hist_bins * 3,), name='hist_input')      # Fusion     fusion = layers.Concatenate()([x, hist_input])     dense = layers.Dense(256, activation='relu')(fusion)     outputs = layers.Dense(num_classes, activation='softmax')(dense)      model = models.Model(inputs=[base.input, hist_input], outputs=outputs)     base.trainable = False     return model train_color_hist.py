import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from model_color_hist import build_color_hist_cnn

def extract_color_hist(image, bins=32):
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
    return np.array(hist_features)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--output_model', default='color_hist_cnn.h5')
    args = parser.parse_args()

    X_img, X_hist, y = [], [], []
    for label in sorted(os.listdir(args.data_dir)):
        for fname in os.listdir(os.path.join(args.data_dir, label)):
            path = os.path.join(args.data_dir, label, fname)
            img = load_img(path, target_size=(224, 224))
            arr = img_to_array(img)
            X_img.append(arr / 255.0)
            X_hist.append(extract_color_hist(arr.astype('uint8')))
            y.append(int(label))
    X_img = np.array(X_img)
    X_hist = np.array(X_hist)
    y = to_categorical(y)

    model = build_color_hist_cnn(input_shape=X_img.shape[1:], hist_bins=X_hist.shape[1]//3, num_classes=y.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([X_img, X_hist], y, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.2)
    model.save(args.output_model)
