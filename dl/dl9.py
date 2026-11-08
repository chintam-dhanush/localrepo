
#week9
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# 1. Load MNIST Dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# 2. Select Smaller Dataset

X_train, y_train = X_train[:10000], y_train[:10000]
X_test, y_test = X_test[:2000], y_test[:2000]


# 3. Normalize Data

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# 4. Add Channel Dimension

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


# 5. Define Residual Block

def residual_block(x, filters, stride=1):
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride,
                      padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding="same",
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride,
            padding="same", use_bias=False
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


# 6. Build ResNet Model

inputs = layers.Input(shape=(28, 28, 1))

x = layers.Conv2D(16, 3, padding="same",
                  use_bias=False)(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = residual_block(x, 16)
x = residual_block(x, 16)

x = residual_block(x, 32, stride=2)
x = residual_block(x, 32)

x = residual_block(x, 64, stride=2)
x = residual_block(x, 64)

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = models.Model(inputs, outputs)


# 7. Display Model

model.summary()


# 8. Compile Model

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# 9. Train ResNet

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)


# 10. Evaluate Model

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy * 100, 2), "%")

# 11. Make Predictions

predictions = model.predict(X_test, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)


# 12. Display Predictions

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap="gray")
    plt.title(f"True:{y_test[i]} Pred:{predicted_labels[i]}")
    plt.axis("off")

plt.show()


