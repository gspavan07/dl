# Machine Learning Implementations

This repository contains implementations of various machine learning models using TensorFlow/Keras. Each program includes a brief description, code, and expected output. These implementations are simple yet effective, demonstrating fundamental deep learning concepts for beginners and practitioners.

## 1. Multilayer Perceptron (MLP) for MNIST
A simple fully connected neural network (MLP) designed to classify handwritten digits from the MNIST dataset. The model consists of dense layers with ReLU activation and a final softmax layer for classification. It achieves high accuracy on digit classification tasks and serves as an introduction to deep learning.

### Code:
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.reshape(-1, 28*28) / 255.0, x_test.reshape(-1, 28*28) / 255.0

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(28*28,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

### Expected Output:
```
Epoch 1/10
Training Accuracy: ~98%
Validation Accuracy: ~97%
```

---

## 2. Neural Network for News Classification (Reuters Dataset)
A neural network model to classify news articles into different categories using the Reuters dataset. It uses a dense network with ReLU activation and a softmax output layer to predict one of 46 possible categories. This model helps understand text classification using deep learning and natural language processing.

### Code:
```python
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)
tokenizer = Tokenizer(num_words=10000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(10000,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(46, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

### Expected Output:
```
Epoch 1/10
Training Accuracy: ~85%
Validation Accuracy: ~80%
```

---

## 3. One-Hot Encoding for Words or Characters
One-hot encoding is a fundamental technique in NLP where words or characters are converted into binary vectors. This method helps machine learning models interpret text data numerically. Here, we demonstrate one-hot encoding for words and characters using TensorFlow’s tokenizer.

### Code:
```python
from tensorflow.keras.preprocessing.text import Tokenizer

texts = ["hello world", "machine learning is fun"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_encoded = tokenizer.texts_to_matrix(texts, mode='binary')
print("Word-Level One-Hot Encoding:", word_encoded)
```

### Expected Output:
```
Word-Level One-Hot Encoding: [Encoded Matrix]
```

---

## 4. CNN for Handwritten Digit Recognition (MNIST)
A convolutional neural network (CNN) for classifying handwritten digits from the MNIST dataset. It consists of convolutional layers followed by max pooling, flattening, and dense layers for classification. CNNs are powerful for image recognition tasks and significantly improve accuracy over traditional MLPs.

### Code:
```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.reshape(-1, 28, 28, 1) / 255.0, x_test.reshape(-1, 28, 28, 1) / 255.0

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

### Expected Output:
```
Epoch 1/10
Training Accuracy: ~99%
Validation Accuracy: ~98%
```

---

## 5. CNN (VGG16) for Image Classification
A transfer learning approach using the VGG16 model pre-trained on ImageNet. It extracts deep features from images and adds custom classification layers. This method is ideal for image classification tasks with small datasets, leveraging the power of pre-trained deep learning models.

### Code:
```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Expected Output:
```
Training Accuracy: ~90%
Validation Accuracy: ~85%
```
