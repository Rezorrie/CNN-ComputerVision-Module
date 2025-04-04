# 📈 Machine Learning

## Definisi dan Konsep Dasar
**Machine Learning** adalah salah satu cabang dari AI yang memungkinkan sistem yang dibuat mampu mempelajari atau menganalisis suatu data tanpa harus diprogram manual oleh manusia. 
Berdasarkan bagaimana sistem tersebut bekerja, model Machine Learning dapat dikategorikan menjadi tiga, yaitu :

- **Supervised Learning**
- **Unsupervised Learning**
- **Reinforcement Learning**

---

# 🏷️ Supervised Learning

## Definisi dan Konsep Dasar
**Supervised Learning** adalah salah satu teknik untuk melatih model machine learning menggunakan dataset yang **sudah dilengkapi dengan labelnya**. 

## Jenis Jenis Supervised Learning
### 1️⃣ **Classification** : 
Model akan memprediksi suatu data kategorikal dan mengkategorikan input data tersebut ke suatu kelas.

📌 **Karakteristik** :
- Output berupa kategorisasi kelas
- Jumlah kelas paling sedikit adalah dua (binary) atau lebih (multi-class)
- Umumnya menggunakan fungsi sigmoid dan softmax

⚙️ **Contohnya** : 
- Klasifikasi gambar kucing dan anjing
- Klasifikasi buah 
- Klasifikasi email spamn

📊 **Algoritma** :
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Neural Network
  
### 2️⃣ **Regression** : 
Model akan mempelajari suatu data kontinyu dan memprediksi nilai numerik berdasarkan data input

📌 **Karakteristik** :
- Output berupa angka kontinyu 
- Umumnya direpresentasikan dengan *Line Plot*

⚙️ **Contohnya** : 
- Prediksi harga rumah 
- Prediksi suhu dan kelembapan udara untuk cuaca
- Prediksi saham

📊 **Algoritma** :
- Linear Regression
- Polinomial Regression
- Decision Tree
- Random Forest
- Support Vector Regression (SVR)
- Neural Network (LSTM)

---

# 🛑 Unsupervised Learning

## Definisi dan Konsep Dasar
**Unsupervised Learning** adalah salah satu teknik untuk melatih model machine learning menggunakan dataset yang **tidak dilengkapi dengan labelnya**. 


## Jenis Jenis Supervised Learning
### 1️⃣ **Clustering** : 
Model akan mengelompokkan data berdasarkan kemiripan fitur antar data

#### ⚙️ **Contohnya** : 
- Segmentasi pelanggan
- Pengelompokan berita

#### 📊 **Algoritma** :
- K-Means Clustering
- Hierarchial Clustering

### 2️⃣ **Dimension Reduction** : 
Model akan mengurangi jumlah fitur pada data namun dengan tetap mempertahankan informasi pada data

#### ⚙️ **Contohnya** : 
- Mengurangi jumlah kolom dataset dengan PCA
- Visualisasi data kompleks dengan t-SNE

#### 📊 **Algoritma** :
- Principal Component Analysis (PCA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)

---

# 🤖 Reinforcement Learning

## Definisi dan Konsep Dasar
**Reinforcement Learning** adalah salah satu teknik untuk melatih model machine learning tanpa menggunakan dataset. Namun, model Machine Learning akan belajar dengan konsep **reward and punishment**. Dimana, model akan mendapatkan reward apabila melakukan aksi yang benar dan punishment apabila sebaliknya. Model nantinya akan berusaha untuk mendapatkan cara untuk memaksimalkan reward. 

## Unsur - Unsur Reinforcement Learning
- **Agent** -> Sistem yang akan dilatih
- **Environment** - Dunia tempat agen melakukan training
- **State** - Keadaan sistem
- **Action** - Tindakan yang dapat dilakukan oleh sistem
- **Reward** - Hadiah apabila agen mendekati objektif
- **Punishment** - Hukuman apabila agen menjauhi objektif
- **Objektif** - Keadaan target yang diinginkan agen untuk dicapai
- **Policy** - Strategi agen dalam memilih aksi berdasarkan state-nya
- **Q-Value** - Ekspektasi total reward

## Prinsip Kerja
- Agent akan mengamati state yang ia jalani
- Agent akan menentukan suatu policy kemudian akan mengambil action berdasarkan policy tersebut
- Agent akan mendapatkan reward atau punishment berdasarkan action yang dia ambil relatif terhadap objektif
- Agent akan memperbarui policy
- Proses akan terus diulang hingga agent mampu mencapai reward maksimum
