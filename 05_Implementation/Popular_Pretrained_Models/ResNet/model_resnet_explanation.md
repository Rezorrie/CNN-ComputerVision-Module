## 1. **Definisi Function**
Mendefinisikan function untuk membuad custom model dari ResNet50 (50 Layer)

```python
def build_resnet_finetune(input_shape=(224, 224, 3), num_classes=10, freeze_base=True):
```

**Parameter** :
* **input_shape** : Dimensi gambar input (width, height, channels). Pada code diberikan parameter default -> (224, 224, 3)
* **num_classes** : Banyaknya kelas klasifikasi. Pada code diberikan parameter default -> 10
* **freeze_base** : Apakah ingin memastikan weight model ResNet50 berubah saat training atau tidak. Pada code diberikan parameter default -> True (transfer learning)

## 2. **Base Model**
Mengimport ResNet50

```python
base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
```

**Parameter** :
* **weights='imagenet'** : Menggunakan weight dari hasil pre-trained ImageNet
* **include_top=False** : Tidak mengimport layer klasifikasi
* **input_shape=input_shape** : Set dimensi input

---
```python
    if freeze_base:
        for layer in base.layers:
            layer.trainable = False
```
* **Condition** : Ketika `freeze_base=True`
* **Loop** : Berfungsi untuk melakukan iterasi semua layer ResNet50
* `layer.trainable=False` : Mencegah perubahan weight saat training -> Cocok untuk dataset kecil


## 3. **Custom Classification**
```python
    x = layers.GlobalAveragePooling2D()(base.output)
```
* **GlobalAveragePooling2D()** :

```python
    x = layers.Dense(256, activation='relu')(x)
```
Menambahkan fully connected layer sebagai decoder
* **Dense()** : Fully connected layer dengan 256 neuron dan fungsi aktivasi relu

## 4. **Final Model**
```python
    model = models.Model(inputs=base.input, outputs=outputs)
```
menambahkan layer output
* **inputs=base.input** : Meletakan ResNet50's untuk menerima input dan mengolahnya
* **outputs=outputs** : Menghubungkan custom classification layers
