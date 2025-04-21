# Apa itu Preprocessing?

**Preprocessing** adalah suatu metode untuk melakukan transformasi atau manipulasi pada data yang akan dijadikan input data pada model. Supaya, proses training model bisa berjalan dengan sebaik mungkin.
Untuk model CNN, input data yang berupa gambar akan ditransformasi dan manipulasi untuk memastikan hasil training model memberikan output sebaik mungkin.

---
# Pentingnya Preprocessing
- **Standardization** : Memastikan input data uniform
- **Feature Enhancement** : Meredam noise tanpa menghilangkan informasi penting pada fitur
- **Efisiensi Komputasi** : Preprocessing dapat menghilangkan data yang tidak penting sehingga mengurangi jumlah data yang diproses
- **Efisiensi Learning** : Pattern pada data menjadi lebih mudah untuk dipelajari oleh model
- **Generalisasi** : Harapannya, model juga bisa memiliki performa yang bagus di data yang baru

---
# Jenis - Jenis Preprocessing pada CNN

### 1. Image Resizing 
Beberapa model CNN memerlukan data dengan size tertentu untuk diolahnya. Resizing memastikan data yang akan diolah CNN sesuai dengan ukuran input data yang diterima model

```Python
# --------------------------------------------------------------------------------------
# PyTorch
# --------------------------------------------------------------------------------------
import torch
from torchvision import transforms
from PIL import Image

resize_transform = transforms.Resize((224, 224))  # Resize kek 224 x 224 (Ukuran Umum)
image = Image.open('sample_image.jpg')
resized_image = resize_transform(image)

# --------------------------------------------------------------------------------------
# TensorFlow
# --------------------------------------------------------------------------------------
import tensorflow as tf

image_path = 'path/to/image'
target_size = (224, 224)

img = tf.io.read_file(image_path) # Baca Gambar
img = tf.image.decode_jpeg(img, channels=3)

img = tf.image.resize(img, target_size) # Resize


# --------------------------------------------------------------------------------------
# OpenCV
# --------------------------------------------------------------------------------------
import cv2
import numpy as np

img = cv2.imread(image_path) # Baca Gambar
    
img_resized = cv2.resize(img, target_size) # Resize
```

### 2. Image Scaling
Melakukan normalisasi terhadap pixel value pada gambar sehingga membantu stabilitas dan konvergensi pada proses training

```Python
# ======================================================================================
# Dari [0, 255] →menjadi [0, 1] 
# ======================================================================================

# PyTorch
normalize_transform = transforms.Compose([
    transforms.ToTensor(),  # Mengubah range ke 0 hingga 1

# TensorFlow
img = tf.io.read_file(image_path) # Baca Gambar
img = tf.image.decode_jpeg(img, channels=3)
img = tf.cast(img, tf.float32) / 255.0

# OpenCV
img = cv2.imread(image_path)
img = img.astype(np.float32) / 255.0

# ======================================================================================
# Mean/Std Normalization (ImageNet statistics)
# ======================================================================================

# PyTorch
imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # ImageNet mean
    std=[0.229, 0.224, 0.225]    # ImageNet std
)

# Combine resize and normalize in a pipeline
preprocessing_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),          # Scales to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# TensorFlow mean/std normalization
def normalize_imagenet_tf(image):
    # Convert to float32
    image = tf.cast(image, tf.float32)
    # Scale to [0, 1]
    image = image / 255.0
    # Normalize with ImageNet stats
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image
```

### 3. Grayscale Conversion
Melakukan konversi gambar dari RGB menjadi grayscale dapat meningkatkan efisiensi komputasi dan efisiensi learning model. Sebab, konversi ke grayscale mampu mengurangi dimensi gambar RGB yaitu 3 menjadi 1.
Misal, gambar RGB dengan resolusi 224 x 224 akan tertera sebagai (224, 224, 3) sedangkan resolusi yang sama dengan gambar grayscale menjadi (224, 224, 1)

```Python
# --------------------------------------------------------------------------------------
# PyTorch
# --------------------------------------------------------------------------------------
grayscale_transform = transforms.Grayscale(num_output_channels=1)  # 1 channel output sebab grayscale

gray_image = grayscale_transform(image)

# Multiple Transformasi
mnist_preprocessing = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST-specific statistics
])
# --------------------------------------------------------------------------------------
# TensorFlow
# --------------------------------------------------------------------------------------
img = tf.io.read_file(image_path) # Baca Gambar
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.rgb_to_grayscale(img)

# --------------------------------------------------------------------------------------
# OpenCV grayscale conversion
# --------------------------------------------------------------------------------------
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ============================================================================================================================================================================
# Important: Adjusting CNN input layer for grayscale
# ============================================================================================================================================================================
"""
# --------------------------------------------------------------------------------------
# PyTorch
# --------------------------------------------------------------------------------------
model = nn.Sequential(
    nn.Conv2d(in_channels=1,  # 1 = grayscale, 3 = RGB
              out_channels=32, 
              kernel_size=3, 
              padding=1),
    # rest of the model...
)

# --------------------------------------------------------------------------------------
# TensorFlow/Keras:
# --------------------------------------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', 
                          input_shape=(height, width, 1)),  # 1 channel = grayscale
    # rest of the model...
])
"""
```

### 4. Histogram Equalization / CLAHE (Contrast Limited Adaptive Histogram Equalization)
Histogram Equalization adalah teknik yang digunakan untuk meningkatkan kecerahan pada gambar. Umum dipakai untuk gambar medis atau gambar yang memiliki tingkat kecerahan minim

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Histogram Equalization
# --------------------------------------------------------------------------------------

"""Mengaplikasikan basic histogram equalization"""
if len(image.shape) > 2 and image.shape[2] == 3: # Mengubah gambar ke grayscale apabila masih RGB
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray = image
    
equalized = cv2.equalizeHist(gray) # histogram equalization


"""Mengaplikasikan Contrast Limited Adaptive Histogram Equalization."""
if len(image.shape) > 2 and image.shape[2] == 3:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # Mengubah dari BGR ke LAB
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size) #CLAHE ke channel L
    cl = clahe.apply(l)
    
    merged = cv2.merge((cl, a, b)) # Gabungkan channelnya
    
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR) # Convert ke BGR
else:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    result = clahe.apply(image)

# --------------------------------------------------------------------------------------
# Visualisasi
# --------------------------------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original, cmap='gray' if len(original.shape) == 2 else None)
plt.title(title1)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(processed, cmap='gray' if len(processed.shape) == 2 else None)
plt.title(title2)
plt.axis('off')

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------
# Jadikan Function
# --------------------------------------------------------------------------------------

def preprocess_with_clahe(image_path, target_size=(224, 224)):
    # Read image
    img = cv2.imread(image_path)
    
    # Apply CLAHE
    img_clahe = apply_clahe(img)
    
    # Resize
    img_resized = cv2.resize(img_clahe, target_size)
    
    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return img_normalized
```

### 5. Data Tidak Seimbang
Data tidak seimbang adalah kondisi dimana data dengan label tertentu memiliki jumlah yang lebih sedikit daripada data dengan label lainnya. Hal tersebut dapat menjadikan model menjadi bias terhadap data yang memiliki label lebih banyak.

```Python
import numpy as np
from collections import Counter
from sklearn.utils import resample
from torchvision import transforms
import random

# 1. Analyze class distribution
def analyze_class_distribution(labels):
    """Analyze and print class distribution."""
    counter = Counter(labels)
    total = sum(counter.values())
    
    print("Class distribution:")
    for label, count in counter.items():
        print(f"  Class {label}: {count} samples ({count/total:.2%})")
    
    return counter

# 2. Undersampling majority classes
def undersample_majority_classes(images, labels, target_count=None):
    """Undersample majority classes to match the minority class count."""
    counter = Counter(labels)
    if target_count is None:
        target_count = min(counter.values())
    
    # Separate images and labels by class
    class_indices = {}
    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Undersample each class to target_count
    balanced_indices = []
    for label, indices in class_indices.items():
        if len(indices) > target_count:
            balanced_indices.extend(random.sample(indices, target_count))
        else:
            balanced_indices.extend(indices)
    
    balanced_images = [images[i] for i in balanced_indices]
    balanced_labels = [labels[i] for i in balanced_indices]
    
    return balanced_images, balanced_labels

# 3. Oversampling minority classes
def oversample_minority_classes(images, labels, target_count=None):
    """Oversample minority classes to match the majority class count."""
    counter = Counter(labels)
    if target_count is None:
        target_count = max(counter.values())
    
    # Separate images and labels by class
    class_data = {}
    for idx, label in enumerate(labels):
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(images[idx])
    
    # Oversample each class to target_count
    balanced_images = []
    balanced_labels = []
    
    for label, imgs in class_data.items():
        if len(imgs) < target_count:
            # Oversample with replacement
            oversampled = resample(imgs, 
                                  replace=True, 
                                  n_samples=target_count,
                                  random_state=42)
            balanced_images.extend(oversampled)
            balanced_labels.extend([label] * target_count)
        else:
            balanced_images.extend(imgs)
            balanced_labels.extend([label] * len(imgs))
    
    return balanced_images, balanced_labels

# 4. Enhanced augmentation for minority classes
def create_enhanced_augmentation(minority_classes):
    """Create stronger augmentation pipeline for minority classes."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 5. Class-aware batch sampling
class ClassAwareSampler:
    """PyTorch sampler that ensures each batch contains samples from all classes."""
    def __init__(self, labels, batch_size, classes_per_batch=None):
        self.labels = labels
        self.batch_size = batch_size
        self.classes_per_batch = classes_per_batch or len(set(labels))
        
        # Group indices by label
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        # Shuffle indices within each class
        for label in self.label_to_indices:
            random.shuffle(self.label_to_indices[label])
        
        self.used_indices = {label: 0 for label in self.label_to_indices}
        self.count = 0
        self.unique_labels = list(self.label_to_indices.keys())
    
    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count >= len(self.labels):
            raise StopIteration
        
        # Select classes for this batch
        batch_classes = random.sample(
            self.unique_labels, 
            min(self.classes_per_batch, len(self.unique_labels))
        )
        
        # Determine samples per class
        samples_per_class = self.batch_size // len(batch_classes)
        
        batch_indices = []
        for label in batch_classes:
            class_indices = self.label_to_indices[label]
            if self.used_indices[label] + samples_per_class >= len(class_indices):
                # Reshuffle if we're at the end
                random.shuffle(class_indices)
                self.used_indices[label] = 0
            
            # Get indices for this class
            start_idx = self.used_indices[label]
            end_idx = min(start_idx + samples_per_class, len(class_indices))
            batch_indices.extend(class_indices[start_idx:end_idx])
            
            # Update used indices
            self.used_indices[label] = end_idx
        
        self.count += len(batch_indices)
        return batch_indices
    
    def __len__(self):
        return len(self.labels) // self.batch_size
```

### 6. Data Splitting 
Dalam melakukan proses training, data akan dibagi menjadi data training, data testing, dan data validation.

```Python
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
# --------------------------------------------------------------------------------------
# Train Test Split
# --------------------------------------------------------------------------------------

"""Split data menjadi training, validation, dan test sets."""
# Split Pertama : Train dan Teset
X_train_val, X_test, y_train_val, y_test = train_test_split(
    images, labels, test_size=test_size, random_state=random_state, stratify=labels
)

# Split ke dua : Train dan Val
# Calculate validation size relative to the train_val set
relative_val_size = val_size / (1 - test_size)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=relative_val_size, 
    random_state=random_state, stratify=y_train_val
)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")

# --------------------------------------------------------------------------------------
# Stratified k-Fold Split
# --------------------------------------------------------------------------------------

"""stratified k-fold splits untuk cross-validation."""
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

folds = []
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
    print(f"Fold {fold_idx+1}:")
    print(f"  Training: {len(train_idx)} samples")
    print(f"  Validation: {len(val_idx)} samples")
    
    X_train, X_val = [images[i] for i in train_idx], [images[i] for i in val_idx]
    y_train, y_val = [labels[i] for i in train_idx], [labels[i] for i in val_idx]
    
    folds.append(((X_train, y_train), (X_val, y_val)))
```
