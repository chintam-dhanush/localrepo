
#week8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# 1. Demonstrate Convolution Operations

image = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
], dtype=np.float32).reshape(1, 7, 7, 1)

vertical = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32).reshape(3, 3, 1, 1)

horizontal = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
], dtype=np.float32).reshape(3, 3, 1, 1)

sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32).reshape(3, 3, 1, 1)

v = tf.nn.conv2d(image, vertical, strides=1, padding="SAME")
h = tf.nn.conv2d(image, horizontal, strides=1, padding="SAME")
s = tf.nn.conv2d(image, sharpen, strides=1, padding="SAME")

plt.subplot(1, 4, 1)
plt.imshow(image[0, :, :, 0], cmap="gray")
plt.title("Original")

plt.subplot(1, 4, 2)
plt.imshow(v[0, :, :, 0], cmap="gray")
plt.title("Vertical")

plt.subplot(1, 4, 3)
plt.imshow(h[0, :, :, 0], cmap="gray")
plt.title("Horizontal")

plt.subplot(1, 4, 4)
plt.imshow(s[0, :, :, 0], cmap="gray")
plt.title("Sharpen")

plt.show()


# 2. Demonstrate Pooling Operations

feature_map = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=np.float32).reshape(1, 4, 4, 1)

max_pool = tf.nn.max_pool2d(
    feature_map, ksize=2, strides=2, padding="VALID"
)

avg_pool = tf.nn.avg_pool2d(
    feature_map, ksize=2, strides=2, padding="VALID"
)

global_avg = tf.reduce_mean(feature_map, axis=[1, 2])

print("Max Pooling:\n", max_pool[0, :, :, 0])
print("Average Pooling:\n", avg_pool[0, :, :, 0])
print("Global Average Pooling:\n", global_avg)


# 3. Build CNN

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.AveragePooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation="softmax")
])

model.summary()

# 4. Compile and Train CNN

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# 5. Evaluate CNN

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy * 100, 2), "%")


# 6. Display Training Performance

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.legend()

plt.show()
