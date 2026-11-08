
#week7
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

# 1. Load Fashion-MNIST Dataset

(X_train, y_train), (X_test, y_test) = \
    keras.datasets.fashion_mnist.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 2. Build Neural Network

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# 3. Normal / Full-Batch Gradient Descent

tf.keras.utils.set_random_seed(42)
model_normal = build_model()

start = time.time()

history_normal = model_normal.fit(
    X_train, y_train,
    epochs=20,
    batch_size=len(X_train),
    validation_split=0.1,
    verbose=0
)

normal_time = time.time() - start
normal_loss, normal_acc = model_normal.evaluate(
    X_test, y_test, verbose=0
)


# 4. Mini-Batch Gradient Descent

tf.keras.utils.set_random_seed(42)
model_minibatch = build_model()

batch_size = 32
start = time.time()

history_minibatch = model_minibatch.fit(
    X_train, y_train,
    epochs=20,
    batch_size=batch_size,
    validation_split=0.1,
    verbose=0
)

minibatch_time = time.time() - start
minibatch_loss, minibatch_acc = model_minibatch.evaluate(
    X_test, y_test, verbose=0
)


# 5. Calculate Weight Updates

normal_updates = 1
minibatch_updates = int(np.ceil(len(X_train) / batch_size))


# 6. Display Results

print("\nFinal Comparison")
print("Parameter       Normal      Mini-Batch")
print("Batch Size     ", len(X_train), "   ", batch_size)
print("Updates/Epoch  ", normal_updates, "           ",
      minibatch_updates)
print("Test Loss      ", round(normal_loss, 4), "    ",
      round(minibatch_loss, 4))
print("Test Accuracy  ", round(normal_acc * 100, 2), "%   ",
      round(minibatch_acc * 100, 2), "%")
print("Training Time  ", round(normal_time, 2), "    ",
      round(minibatch_time, 2))


# 7. Compare Training Accuracy

normal_train_acc = history_normal.history["accuracy"][-1]
minibatch_train_acc = history_minibatch.history["accuracy"][-1]

print("\nTraining Accuracy")
print("Normal:", round(normal_train_acc * 100, 2), "%")
print("Mini-Batch:", round(minibatch_train_acc * 100, 2), "%")

