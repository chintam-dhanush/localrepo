import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Generate dataset
# --------------------------------------------------
X, y = make_moons(n_samples=1500, noise=0.25, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------------------------------------------
# 2. Build DNN
# --------------------------------------------------
def build_model(reg=None):
    model = keras.Sequential([
        layers.Input(shape=(2,)),

        layers.Dense(
            64,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=reg
        ),

        layers.Dense(
            32,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=reg
        ),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# 3. Train and evaluate
# --------------------------------------------------
results = {}

methods = {
    "NONE": None,
    "L1": regularizers.L1(0.001),
    "L2": regularizers.L2(0.001)
}

for name, reg in methods.items():

    print("\n" + "=" * 55)
    print("TRAINING:", name)
    print("=" * 55)

    model = build_model(reg)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.20,
        epochs=300,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    loss, accuracy = model.evaluate(
        X_test, y_test, verbose=0
    )

    results[name] = (model, history, accuracy)

    print("Test Accuracy:", round(accuracy * 100, 2), "%")
    print("Training stopped at epoch:", len(history.history["loss"]))


# --------------------------------------------------
# 4. Display final weights
# --------------------------------------------------
for name, (model, history, accuracy) in results.items():

    print("\n" + "=" * 55)
    print("FINAL WEIGHTS -", name)
    print("=" * 55)

    for layer in model.layers:

        if len(layer.get_weights()) == 0:
            continue

        weights, bias = layer.get_weights()

        print("\nLayer:", layer.name)
        print("Weights shape:", weights.shape)
        print("Bias shape:", bias.shape)
        print("Average |Weight|:",
              np.mean(np.abs(weights)))


# --------------------------------------------------
# 5. Accuracy comparison
# --------------------------------------------------
print("\n" + "=" * 55)
print("FINAL RESULTS")
print("=" * 55)

for name, (_, _, accuracy) in results.items():
    print(f"{name:5s} : {accuracy * 100:.2f}%")