# Apa itu CNN Building Blocks?

**Neural Network Building Blocks** adalah kumpulan dari beragam layer Neural Network yang membangun satu kesatuan sistem yang bertujuan untuk memproses suatu data. Sehingga, **CNN Building Blocks** adalah satu kesatuan sistem yang bertujuan untuk memprocess data grid, seperti gambar.

---
# Komponen CNN Building Blocks

## 1. Convolutional Layer

**Convolutional Layer** adalah salah satu layer neural network yang memiliki kernel/filter yang berfungsi sebagai feature extractor. Dimana, kernel/filter tersebut akan dilatih dalam proses training.

### Secara Matematis

$$ Y_{i, j} = \sum \sum X_{i + m, j + n} W_{m, n} + b $$

### Istilah Umum
- **Kernel/Filters** : Matriks persegi yang akan bergerak menyusuri data untuk mencari pola atau ekstraksi fitur. Ukurannya dapat ditentukan namun umumnya 3x3 atau 5x5
- **Stride** : Laju perpindahan filter. Umumnya satu pixel atau dua pixel
- **Padding** : Tambahan dimensi pada data input untuk mengatur dimensi data setelah proses konvolusi
- **Number of Filter** : Banyaknya filter pada layer. Lebih banyak filter memungkinkan layer untuk mendeteksi lebih banyak fitur namun komputasi lebih berat.

---
## 2. Activation Function

**Activation Function** adalah fungsi matematis yang diaplikasikan dengan tujuan memberikan sifat non-linearitas pada model. Sebab, data di dunia nyata umumnya non-linear. Sehingga, mampu membuat model lebih baik dalam menghadapi data yang kompleks. 

### Beberapa Activation Function yang Umum
- **ReLU (Rectiied Linear Unit)** :
  - **Rumus** : ![ReLU Derivative](https://latex.codecogs.com/svg.image?f(x)=\begin{cases}x&\text{if}x%3E0\\0&\text{if}x\leq&space;0\end{cases})
  - **Use** : Digunakan secara umum di neural network

- **Leaky ReLU**
  - **Rumus** : ![Leaky ReLU](https://latex.codecogs.com/svg.image?f(x)=\begin{cases}x&\text{if}x%3E0\\\alpha&space;x&\text{if}x\leq&space;0\end{cases}\quad\text{where}\alpha\in(0,1),\text{e.g.,}0.01)

- **Parametric ReLU** :
    - **Rumus** :

- **Sigmoid**
    - **Rumus** :
    - **Use** : Binary Classification
 
- **Tanh** :
    - **Rumus** :
    - **Use** : Binary Classification

- **Swish**
    - **Rumus** :
    - **Use** : EfficientNet

---
## 3. Batch Normalization

**Batch Normalization Layer** adalah layer dalam struktur CNN yang bertujuan untuk melakukan normalisasi terhadap data yang diproses

### Peran Batch Normalization
- Untuk stabilisasi dan meningkatkan performa model
- Bertindak sebagai regularization sehingga mengurangi overfitting
- Mempermudah konvergensi losses

---
## 4. Pooling Layers

**Pooling Layers** adalah layer dalam CNN yang bertugas untuk mengurangi dimensi data tanpa menghilangkan informasi pada data. Sama seperti **Convolutional Layer**, Pooling Layer berupa matriks yang ukurannya dapat ditentukan sendiri. Selain itu, matriks tersebut akan bergerak menyusuri data secara menyeluruh

### Jenis Pooling Layers
- **Max Pooling Layer** : Bekerja dengan mengambil nilai paling besar dalam window-nya
- **Average Pooling Layer** : Mengambil nilai rerata di windownya

---
## 5. Dropout Layers

**Dropout Layers** adalah layer dalam CNN yang digunakan untuk mengurangi overfitting dengan cara mematikan beberapa neuron secara random dalam proses training.

### Peran Dropout layer
- Mencegah overfitting dengan mematikan beberapa neuron sementara
- Membuat model menjadi lebih robust dengan mencegah co-adaptation pada neuron
- Hanya aktif ketika proses training sehingga tidak mengganggu proses identifikasi 
