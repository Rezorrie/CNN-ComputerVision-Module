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
# ======================================================================================
# PyTorch
# ======================================================================================
import torch
from torchvision import transforms
from PIL import Image

resize_transform = transforms.Resize((224, 224))  # Resize kek 224 x 224 (Ukuran Umum)
image = Image.open('sample_image.jpg')
resized_image = resize_transform(image)

# =====================================================================================
# TensorFlow
# ======================================================================================
import tensorflow as tf

image_path = 'path/to/image'
target_size = (224, 224)

img = tf.io.read_file(image_path) # Baca Gambar
img = tf.image.decode_jpeg(img, channels=3)

img = tf.image.resize(img, target_size) # Resize


# ======================================================================================
# OpenCV
# ======================================================================================
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

### 
