# CNN Building Blocks: Komponen Utama Jaringan Syaraf Konvolusional

**CNN Building Blocks** adalah kumpulan lapisan neural network yang dirancang khusus untuk memproses data grid seperti gambar. Berikut penjelasan komponen utamanya:

---

## 1. Convolutional Layer
**Layer utama** yang menggunakan kernel/filter untuk ekstraksi fitur secara otomatis.

### Representasi Matematis
$$ Y_{i,j} = \sum_{m}\sum_{n} X_{i+m,j+n} \cdot W_{m,n} + b $$

### Komponen Penting:
| Istilah       | Deskripsi                                                                 |
|---------------|---------------------------------------------------------------------------|
| **Kernel**    | Matriks ekstraksi fitur (umumnya 3×3 atau 5×5)                           |
| **Stride**    | Jarak pergerakan kernel (biasanya 1 atau 2 pixel)                        |
| **Padding**   | Penambahan border untuk mengontrol dimensi output                         |
| **Channels**  | Jumlah filter (lebih banyak = lebih banyak fitur terdeteksi)             |

![Convolution Visualization](https://miro.medium.com/v2/resize:fit:1400/1*Zx-ZMLKab7VOCQTxdZ1OAw.gif)
```Python
# PyTorch
import torch
import torch.nn as nn

# Simple convolutional layer
conv_layer = nn.Conv2d(
    in_channels=3,       # Input channels, 3 berarti RGB
    out_channels=64,     
    kernel_size=3,       # Ukuran Filter (3x3)
    stride=1,            # Laju Pergeseran Filter
    padding=1            # Zero-padding pada pinggiran data
)

# TensorFlow
import tensorflow as tf

conv_layer_tf = tf.keras.layers.Conv2D(
    filters=64,          # Banyaknya Filter
    kernel_size=3,       # Ukuran filter (3x3)
    strides=1,           # Laju Pergeseran Filter
    padding='same',      # Padding mode, same = enable padding
    activation=None,     # Untuk activation function
    input_shape=(224, 224, 3)  # Data input 224 x 224 dengan RGB
)
```
---

## 2. Activation Function
Fungsi non-linear yang memperkenalkan kompleksitas pada model.

### Fungsi Aktivasi Populer:

#### ReLU (Rectified Linear Unit)
$$ f(x) = \max(0,x) $$
- **Keuntungan**: Komputasi efisien, mitigasi vanishing gradient
- **Penggunaan**: Lapisan tersembunyi (hidden layers)

#### Leaky ReLU
$$ f(x) = \begin{cases} 
x & \text{if } x > 0 \\ 
\alpha x & \text{if } x \leq 0 
\end{cases} \quad (\alpha \approx 0.01) $$
- **Keuntungan**: Mengatasi "dying ReLU" problem

#### Sigmoid
$$ \sigma(x) = \frac{1}{1+e^{-x}} $$
- **Penggunaan**: Lapisan output klasifikasi biner

#### Softmax
$$ \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} $$
- **Penggunaan**: Lapisan output klasifikasi multi-kelas


``` Python
# PyTorch 
relu_layer = nn.ReLU(inplace=True)  # inplace = Memodifikasi input secara langsung, dapat menghemat memori

conv_block_pt = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True)
)

# TensorFlow 
relu_layer_tf = tf.keras.layers.ReLU()

conv_layer_with_activation = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=3,
    strides=1,
    padding='same',
    activation='relu',  # Langsung di Conv2D
    input_shape=(224, 224, 3)
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```
---

## 3. Batch Normalization
Teknik stabilisasi distribusi aktivasi layer.

**Manfaat**:
- Mempercepat konvergensi
- Mengurangi ketergantungan pada inisialisasi weight
- Berfungsi sebagai regularizer ringan

$$ \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta $$

```Python
# PyTorch 
batch_norm = nn.BatchNorm2d(
    num_features=64,  # Harus sesuai dengan banyaknya channel di layer sebelumnya
    eps=1e-5,         # Konstanta ber value kecil untuk stabilitas
    momentum=0.1      
)

# Building Block nya
conv_bn_relu_block = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True)
)

# TensorFlow 
batch_norm_tf = tf.keras.layers.BatchNormalization(
    axis=-1,         
    momentum=0.99,    
    epsilon=1e-5      # Konstanta ber value kecil untuk stabilitas
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.BatchNormalization(),
])
```

---

## 4. Pooling Layers
Operasi downsampling untuk reduksi dimensi.

### Jenis Pooling:
| Tipe          | Operasi               | Keuntungan                     |
|---------------|-----------------------|--------------------------------|
| **Max**       | $\max(\text{window})$ | Mempertahankan fitur dominan   |
| **Average**   | $\text{mean}(\text{window})$ | Lebih smooth                |
| **Global**    | Pooling seluruh feature map | Untuk klasifikasi        |

```Python
# PyTorch 
max_pool = nn.MaxPool2d(
    kernel_size=2,    
    stride=2          
)

avg_pool = nn.AvgPool2d(
    kernel_size=2,
    stride=2
)

cnn_block_with_pool = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2)
)

# TensorFlow 
max_pool_tf = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),   
    strides=(2, 2),     
    padding='valid'     
)

avg_pool_tf = tf.keras.layers.AveragePooling2D(
    pool_size=(2, 2),
    strides=(2, 2)
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
])
```
---

## 5. Dropout
Teknik regularisasi dengan menonaktifkan neuron acak.

**Karakteristik**:
- Hanya aktif selama training
- Rate dropout tipikal: 0.2-0.5
- Mencegah co-adaptation neuron

$$ r_j \sim \text{Bernoulli}(p) $$
$$ \hat{y} = r \cdot y $$

```Python
# PyTorch 
dropout = nn.Dropout2d(p=0.5)  # 50% dropout rate

# Complete CNN block with dropout
complete_cnn_block = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout2d(p=0.25)  # 25% dropout rate
)

# TensorFlow 
dropout_tf = tf.keras.layers.Dropout(rate=0.5)  # 50% dropout rate
spatial_dropout_tf = tf.keras.layers.SpatialDropout2D(rate=0.5)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.25),
])
```
---

## Arsitektur CNN Umumnya
1. Pola umum: `Conv → ReLU → Pooling`
2. Gunakan BatchNorm setelah Conv sebelum aktivasi
3. Dropout setelah lapisan fully connected
4. Hindari penggunaan Sigmoid/Tanh untuk hidden layers
5. Pertimbangkan residual connections untuk jaringan dalam

