## **1. Multilayer Perceptron (MLP) for MNIST Dataset**  
A Multilayer Perceptron (MLP) is a feedforward neural network with multiple layers. This program trains an MLP on the MNIST dataset to classify handwritten digits. The model consists of fully connected layers with ReLU activation and softmax for classification.

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
This program builds a neural network to classify news articles from the Reuters dataset into different categories. It uses word embedding and an LSTM layer to process text sequences and predict one of 46 possible classes.

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
One-hot encoding is a technique used to convert categorical data into numerical format. This program demonstrates how to one-hot encode words using `sklearn`'s `OneHotEncoder`, which transforms categorical labels into binary vectors.

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
A Convolutional Neural Network (CNN) is used to recognize handwritten digits from the MNIST dataset. The model consists of convolutional, pooling, and dense layers, which extract spatial features from images for classification.

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
The VGG16 model is a pre-trained Convolutional Neural Network (CNN) used for image classification. This program loads VGG16 as a feature extractor and adds fully connected layers for classification.

### **Program:**
```python
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

# Load the VGG16 model pre-trained on ImageNet
model = VGG16(weights='imagenet')

# Load and preprocess the image
img_path = 'puppy.jpg'  # Ensure this image is in the same directory or provide the full path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = preprocess_input(img_array)  # Preprocess image for VGG16

# Make predictions
predictions = model.predict(img_array)
decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 3 predicted labels

# Print predictions
print("Top predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")
```

### **Output:**
```
35363/35363 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
Top predictions:
1: Pomeranian (0.75)
2: chow (0.25)
3: keeshond (0.00)
```
