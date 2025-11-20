
import numpy as np
import tensorflow as tf
from tensorflow import keras
datasets = keras.datasets
layers = keras.layers
models = keras.models
from sklearn import svm
import joblib

print("📌 Loading MNIST Dataset...")
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train_cnn = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_cnn = x_test.reshape(-1, 28, 28, 1) / 255.0

# ---------------- CNN MODEL ----------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("📌 Training CNN Model...")
model.fit(x_train_cnn, y_train, epochs=3, batch_size=128)

model.save("mnist_cnn_model.h5")
print("✅ CNN Model Saved!")

# ---------------- SVM MODEL ----------------
x_train_svm = x_train.reshape(len(x_train), -1) / 255.0
svm_model = svm.SVC(probability=True)
print("📌 Training SVM Model (5000 samples)...")
svm_model.fit(x_train_svm[:5000], y_train[:5000])

joblib.dump(svm_model, "mnist_svm_model.pkl")
print("✅ SVM Model Saved!")