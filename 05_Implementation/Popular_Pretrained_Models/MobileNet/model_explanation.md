## 1. **Definisi Function**
Mendefinisikan function untuk membuad custom model dari mobilenetV2

```python
def build_mobilenetv2_transfer(input_shape=(224, 224, 3), num_classes=10):
```

**Parameter** :
* **input_shape** : Dimensi gambar input (width, height, channels). Pada code diberikan parameter default -> (224, 224, 3)
* **num_classes** : Banyaknya kelas klasifikasi. Pada code diberikan parameter default -> 10

## 2. **Base Model**
Mengimport model mobilenet

```python
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
```

**Parameter** :
* **weights='imagenet'** : Menggunakan weight dari hasil pre-trained dataset ImageNet
* **include_top=False** : Tidak mengimport _fully-connected layer_. Biasanya digunakan apabila menambahkan layer custom
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
* **GlobalAveragePooling2D()** : mengurangi dimensi spasial features menjadi 1D vector. M

```python
    x = layers.Dense(128, activation='relu')(x)
```
Menambahkan fully connected layer sebagai decoder
* **Dense()** : Fully connected layer dengan 128 neuron dan fungsi aktivasi relu

## 4. **Final Model**
```python
    model = models.Model(inputs=base.input, outputs=outputs)
    return model
```
menambahkan layer output
* **inputs=base.input** : Meletakan ResNet50's untuk menerima input dan mengolahnya
* **outputs=outputs** : Menghubungkan custom classification layers
