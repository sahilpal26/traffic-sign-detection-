# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import os
from PIL import Image

# === Define dataset path ===
dataset_path = r"C:\Users\admin\Desktop\projects\GTSRB dataset\Train"  # <- Update this if needed
classes = 43  # Number of classes in the dataset

# === Load and preprocess the dataset ===
data = []
labels = []

for i in range(classes):
    path = os.path.join(dataset_path, str(i))
    print(f"Loading images from: {path}")
    if not os.path.exists(path):
        print(f"❌ Folder not found: {path}")
        continue
    images = os.listdir(path)
    for img in images:
        try:
            image = Image.open(os.path.join(path, img))
            image = image.resize((30, 30))
            image = np.array(image)
            if image.shape == (30, 30, 3):  # Ensure image is RGB
                data.append(image)
                labels.append(i)
        except Exception as e:
            print(f"Error loading image {img}: {e}")

data = np.array(data)
labels = np.array(labels)

# === Split into training and testing datasets ===
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# === Normalize the data ===
X_train = X_train / 255.0
X_test = X_test / 255.0

# === One-hot encode the labels ===
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

# === Build the CNN model ===
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(30, 30, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(classes, activation='softmax')
])

# === Compile the model ===
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Train the model ===
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# === Evaluate the model ===
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")

# === Save the model ===
model.save("traffic_sign_recognition_model.h5")
print("📁 Model saved as traffic_sign_recognition_model.h5")

# === Plot training history ===
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.show()

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.show()
