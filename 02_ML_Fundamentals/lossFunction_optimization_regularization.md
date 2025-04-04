# 🔻 Loss Function

## Definisi dan Konsep Dasar
**Loss Function** adalah fungsi matematika yang digunakan untuk mengukur tingkat performa suatu model Machine Learning. 
Loss Function sering digunakan model Machine Learning dalam proses training, validasi, dan testing dimana model akan mengukur tingkat error antara output model dengan label sebenarnya.

## Jenis Jenis Loss Function


### 1️⃣ **Classification** : 
- **Binary Cross Entropy**
- **Categorical Cross Entropy**
- **Sparse Categorical Cross Entropy**

### 2️⃣ **Regression** : 
- **Mean Absolute Error**
- **Mean Square Error**

---

# 🛠️ Optimization

## Definisi dan Konsep Dasar
**Optimization** adalah proses mengubah nilai *weight* dan *bias* pada model Machine Learning supaya performa model Machine Learning menjadi lebih baik.
Proses *Optimization* bergantung erat dengan *Loss Function* dimana output *Loss Function* akan menjadi input proses *Optimization* dan harapannya nilai Loss menjadi lebih minimum.

## Algoritma Optimization

### 1️⃣ **Gradient Descent**
- **Batch Gradient Descent**
- **Stochastic Gradien Descent**
- **Mini Batch Gradient Descent**

### 2️⃣ **Lainnya**
- **Adam (Adaptive Moment Estimation)**
- **RMSProp (Root Mean Square Propagation)**
- **Adagrad (Adaptive Gradient Algorithm)**

---

# 🛡️ Regularization

## Definisi dan Konsep Dasar
**Regularization** adalah teknik untuk mencegah overfitting dengan cara memberikan *constraint* pada *Loss Function* sehingga model Machine Learning mampu me-generalisasi data

## Macam-Macam
- **L1 Regularization (Lasso)**
- **L2 Regularizatoin (Ridge)**
- **Dropout Layer**
- **Batch Normalization**
