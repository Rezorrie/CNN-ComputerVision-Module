## 1. **Main Execution**
```python
if __name__ == '__main__':
```
Memastikan code berjalan saat dieksekusi secara langsung (tidak saat diimport)

## 2. **Parsing Argumen**
```python
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
```
* **argparse** : Menangani argumen command-line
* **--data_dir** : Address dataset
* **--batch_size** : Jumlah gambar yang diproses setiap step
* **--epochs** : Siklus training

## 3. **Data Augmentation**
```python
    train_datagen = ImageDataGenerator(
        rescale=1./255,               
        rotation_range=20,            
        width_shift_range=0.2,         
        height_shift_range=0.2,        
        horizontal_flip=True,          
        validation_split=0.2           
    )
```
Setup data augmentasi

## 4. **Data Generator**
```python
    train_gen = train_datagen.flow_from_directory(
        args.data_dir,
        target_size=(224, 224),        
        batch_size=args.batch_size,
        subset='training'              
    )
    val_gen = train_datagen.flow_from_directory(
        args.data_dir,
        target_size=(224, 224),
        batch_size=args.batch_size,
        subset='validation'           
    )
```
1. Membagi training subset dan validation subset
2. Resize gambar menjadi 224 x 224 pixel
3. Menjadikan batch sesuai batch_size

## 5. **Kompilasi Model**
```python
    model.compile(
        optimizer='adam',                     
        loss='categorical_crossentropy',       
        metrics=['accuracy']                  
    )
```
* **optimizer='adam'** : Menjadikan adam sebagai optimizer untuk back propagation/gradient descent
* **loss='categorical_crossentropy'** : Loss function untuk kelas lebih dari dua
* **metrics=['accuracy']** : Metric yang dimonitor
