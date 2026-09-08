# ============================================================
# DNN vs LOGISTIC REGRESSION
# NONLINEAR CLASSIFICATION USING TWO-MOONS DATASET
# ============================================================

import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# ============================================================
# STEP 1: Generate Two-Moons Dataset
# ============================================================

X, y = make_moons(
    n_samples=1000,
    noise=0.20,
    random_state=42
)

print("Dataset Information")
print("-------------------")
print("Number of Samples :", X.shape[0])
print("Number of Features:", X.shape[1])


# ============================================================
# STEP 2: Split Dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# STEP 3: Standardize Data
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# ============================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================

print("\n" + "="*60)
print("LOGISTIC REGRESSION")
print("="*60)

lr_model = LogisticRegression()

lr_model.fit(
    X_train,
    y_train
)

lr_prediction = lr_model.predict(
    X_test
)


# ============================================================
# Logistic Regression Metrics
# ============================================================

lr_accuracy = accuracy_score(
    y_test,
    lr_prediction
)

lr_precision = precision_score(
    y_test,
    lr_prediction
)

lr_recall = recall_score(
    y_test,
    lr_prediction
)

lr_f1 = f1_score(
    y_test,
    lr_prediction
)


print("\nLogistic Regression Results")

print("Accuracy  :", round(lr_accuracy * 100, 2), "%")
print("Precision :", round(lr_precision * 100, 2), "%")
print("Recall    :", round(lr_recall * 100, 2), "%")
print("F1 Score  :", round(lr_f1 * 100, 2), "%")

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        lr_prediction
    )
)


# ============================================================
# MODEL 2: DEEP NEURAL NETWORK
# ============================================================

print("\n" + "="*60)
print("DEEP NEURAL NETWORK")
print("="*60)


dnn_model = Sequential([

    # Hidden Layer 1
    Dense(
        32,
        activation="relu",
        input_shape=(2,)
    ),

    # Hidden Layer 2
    Dense(
        16,
        activation="relu"
    ),

    # Hidden Layer 3
    Dense(
        8,
        activation="relu"
    ),

    # Output Layer
    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# Compile DNN
# ============================================================

dnn_model.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# Train DNN
# ============================================================

print("\nTraining DNN...")

history = dnn_model.fit(

    X_train,
    y_train,

    epochs=100,

    batch_size=32,

    validation_split=0.20,

    verbose=0
)

print("DNN Training Completed")


# ============================================================
# DNN Prediction
# ============================================================

probability = dnn_model.predict(
    X_test,
    verbose=0
)

dnn_prediction = (
    probability >= 0.5
).astype(int).flatten()


# ============================================================
# DNN Metrics
# ============================================================

dnn_accuracy = accuracy_score(
    y_test,
    dnn_prediction
)

dnn_precision = precision_score(
    y_test,
    dnn_prediction
)

dnn_recall = recall_score(
    y_test,
    dnn_prediction
)

dnn_f1 = f1_score(
    y_test,
    dnn_prediction
)


print("\nDeep Neural Network Results")

print("Accuracy  :", round(dnn_accuracy * 100, 2), "%")
print("Precision :", round(dnn_precision * 100, 2), "%")
print("Recall    :", round(dnn_recall * 100, 2), "%")
print("F1 Score  :", round(dnn_f1 * 100, 2), "%")

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        dnn_prediction
    )
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "="*70)

print(
    "LOGISTIC REGRESSION vs DEEP NEURAL NETWORK"
)

print("="*70)

print(
    "{:<20}{:<20}{:<20}".format(
        "Metric",
        "Logistic Regression",
        "Deep Neural Network"
    )
)

print("-"*70)

print(
    "{:<20}{:<20.2f}{:<20.2f}".format(
        "Accuracy (%)",
        lr_accuracy * 100,
        dnn_accuracy * 100
    )
)

print(
    "{:<20}{:<20.2f}{:<20.2f}".format(
        "Precision (%)",
        lr_precision * 100,
        dnn_precision * 100
    )
)

print(
    "{:<20}{:<20.2f}{:<20.2f}".format(
        "Recall (%)",
        lr_recall * 100,
        dnn_recall * 100
    )
)

print(
    "{:<20}{:<20.2f}{:<20.2f}".format(
        "F1 Score (%)",
        lr_f1 * 100,
        dnn_f1 * 100
    )
)

print("="*70)


# ============================================================
# DETERMINE BETTER MODEL
# ============================================================

if dnn_accuracy > lr_accuracy:

    print(
        "\nResult: DNN performed better "
        "than Logistic Regression."
    )

elif dnn_accuracy < lr_accuracy:

    print(
        "\nResult: Logistic Regression performed "
        "better than DNN."
    )

else:

    print(
        "\nResult: Both models achieved "
        "the same accuracy."
    )