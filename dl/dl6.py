
#week6
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Generate dataset

X, y=make_moons(n_samples=1500, noise=0.25, random_state=42)

X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# 2. Build DNN

def build_model(reg=None):
    model = keras.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dense(32, activation="relu", kernel_regularizer=reg),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# 3. Train and evaluate

methods = {
    "NONE": None,
    "L1": regularizers.L1(0.001),
    "L2": regularizers.L2(0.001)
}

results = {}

for name, reg in methods.items():
    model = build_model(reg)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=0
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    results[name] = (model, accuracy)

    print(name, "Accuracy:", round(accuracy * 100, 2), "%")

# 4. Display final weights

for name, (model, accuracy) in results.items():
    print("\n", name)

    for layer in model.layers:
        weights = layer.get_weights()

        if weights:
            print("Layer:", layer.name)
            print("Average weight:",
                  round(np.mean(np.abs(weights[0])), 4))


# 5. Accuracy comparison

print("\nFinal Results")

for name, (model, accuracy) in results.items():
    print(name, ":", round(accuracy * 100, 2), "%")
