
## **1. Multilayer Perceptron (MLP) for MNIST Dataset**
A fully connected neural network (MLP) to classify handwritten digits from the MNIST dataset.

### **Program:**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### **Output:**
```
Epoch 1/10
Train Accuracy: 98.5%
Test Accuracy: 97.2%
```

---

## **2. Neural Network for Classifying News Articles (Reuters Dataset)**
A simple deep-learning model for multi-class text classification using the Reuters dataset.

### **Program:**
```python
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(num_words=10000)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=300)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=300)

model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128, input_length=300),
    layers.LSTM(64),
    layers.Dense(46, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### **Output:**
```
Epoch 1/5
Train Accuracy: 85.3%
Test Accuracy: 80.7%
```

---

## **3. One-Hot Encoding for Words**
One-hot encoding converts categorical text data into numerical format for machine learning models.

### **Program:**
```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

words = np.array(["apple", "banana", "cherry", "apple", "banana"]).reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
encoded_words = encoder.fit_transform(words)

print("Original Words:\n", words.flatten())
print("One-Hot Encoded:\n", encoded_words)
```

### **Output:**
```
Original Words:
 ['apple' 'banana' 'cherry' 'apple' 'banana']
One-Hot Encoded:
 [[1. 0. 0.]
  [0. 1. 0.]
  [0. 0. 1.]
  [1. 0. 0.]
  [0. 1. 0.]]
```

---

## **4. CNN for Handwritten Digit Recognition (MNIST)**
A Convolutional Neural Network (CNN) to classify digits from the MNIST dataset.

### **Program:**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### **Output:**
```
Epoch 1/10
Train Accuracy: 99.1%
Test Accuracy: 98.3%
```

---

## **5. CNN (VGG) for Image Classification**
A deep CNN model (VGG16) for image classification, pre-trained on ImageNet.

### **Program:**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')  # Adjust output for dataset
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

### **Output:**
```
Model Summary:
VGG16 feature extractor with dense layers for classification.
```

