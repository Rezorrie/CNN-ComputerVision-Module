## 1. **Definisi Function**
Mendefinisikan function untuk membuad custom model dari EfficientNet

```python
def build_efficientnet_transfer(input_shape=(224, 224, 3), num_classes=10):
```

**Parameter** :
* **input_shape** : Dimensi gambar input (width, height, channels). Pada code diberikan parameter default -> (224, 224, 3) -> 224 x 224 pixel, dengan 3 layer yang menandakan RGB
* **num_classes** : Banyaknya kelas klasifikasi. Pada code diberikan parameter default -> 10

## 2. **Base Model**
Mengimport ResNet50

```python
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
```

**Parameter** :
* **weights='imagenet'** : Menggunakan weight dari hasil pre-trained ImageNet
* **include_top=False** : Tidak mengimport layer klasifikasi
* **input_shape=input_shape** : Set dimensi input

---
```python
    base.trainable = False

```
untuk memastikan base layer tidak mengalami training (perubahan weight). Biasanya digunakan untk menjaga pre-trained weights dari model dan apabila hanya ingin melakukan training pada custom layer


## 3. **Custom Classification**
```python
    x = layers.GlobalAveragePooling2D()(base.output)
```
* **GlobalAveragePooling2D()** : mengurangi dimensi spasial features menjadi 1D vector.

```python
    x = layers.Dense(256, activation='relu')(x)
```
Menambahkan fully connected layer sebagai decoder
* **Dense()** : Fully connected layer dengan 256 neuron dan fungsi aktivasi relu

## 4. **Final Model**
```python
  return models.Model(inputs=base.input, outputs=outputs)
```
return model serta menggabungkan EfficienNet sebagai base model dan custom model
