#week1
import math

n = int(input("Enter number of inputs: "))

x = []
for i in range(n):
    x.append(float(input("Enter X" + str(i+1) + ": ")))

print("\n")
w = []
for i in range(n):
    w.append(float(input("Enter W" + str(i+1) + ": ")))

print("\n")
b = float(input("Enter Bias: "))

print("\n")
# Normalization
print("\nChoose Normalization Method")
print("0. No Normalization")
print("1. Min-Max (0 to 1)")
print("2. Min-Max (-1 to 1)")
print("3. Z-Score")
print("4. Decimal Scaling")

norm = int(input("Enter choice: "))
xn = x.copy()

if norm == 1:
    xmin = float(input("Enter minimum: "))
    xmax = float(input("Enter maximum: "))
    for i in range(n):
        xn[i] = (x[i] - xmin) / (xmax - xmin)

elif norm == 2:
    xmin = float(input("Enter minimum: "))
    xmax = float(input("Enter maximum: "))
    for i in range(n):
        xn[i] = 2 * (x[i] - xmin) / (xmax - xmin) - 1

elif norm == 3:
    mean = sum(x) / n
    std = (sum((i - mean) ** 2 for i in x) / n) ** 0.5
    for i in range(n):
        xn[i] = (x[i] - mean) / std if std != 0 else 0

elif norm == 4:
    j = int(input("Enter scaling factor: "))
    for i in range(n):
        xn[i] = x[i] / (10 ** j)

print("\n")
# Activation function
print("\nChoose Activation Function")
print("1. Linear")
print("2. Binary Step")
print("3. Sigmoid")
print("4. Tanh")
print("5. ReLU")
print("6. Leaky ReLU")
print("7. ELU")
print("8. Softplus")

choice = int(input("Enter choice: "))

z = sum(x[i] * w[i] for i in range(n)) + b
zn = sum(xn[i] * w[i] for i in range(n)) + b


def activation(z, choice):
    if choice == 1:
        return z
    elif choice == 2:
        return 1 if z >= 0 else 0
    elif choice == 3:
        return 1 / (1 + math.exp(-z))
    elif choice == 4:
        return math.tanh(z)
    elif choice == 5:
        return max(0, z)
    elif choice == 6:
        return z if z > 0 else 0.01 * z
    elif choice == 7:
        return z if z > 0 else math.exp(z) - 1
    elif choice == 8:
        return math.log(1 + math.exp(z))


a = activation(z, choice)
an = activation(zn, choice)

print("\n")
print("Original Inputs:", x)
print("Normalized Inputs:", xn)
print("Weighted Sum without normalization:", round(z, 6))
print("Output without normalization:", round(a, 6))
print("Weighted Sum with normalization:", round(zn, 6))
print("Output with normalization:", round(an, 6))
















#week2
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(y, y_pred):
    m = len(y)
    return np.sum((y - y_pred) ** 2) / (2 * m)


m = int(input("Enter Number of Training Samples : "))
n = int(input("Enter Number of Input Features : "))

X = []
y = []

print("\nEnter Training Data")

for i in range(m):
    print(f"\nSample {i+1}")
    row = []

    for j in range(n):
        value = float(input(f"Enter X{j+1} : "))
        row.append(value)

    target = int(input("Enter Target Output (0/1) : "))
    X.append(row)
    y.append(target)

X = np.array(X)
y = np.array(y)


print("\nEnter Initial Weights")

weights = []

for i in range(n):
    w = float(input(f"Weight W{i+1} : "))
    weights.append(w)

weights = np.array(weights)

bias = float(input("Enter Initial Bias : "))
learning_rate = float(input("Enter Learning Rate : "))
tolerance = float(input("Enter Saturation Tolerance (Example 0.000001): "))

max_iterations = 10000

previous_cost = float('inf')
iteration = 0

while True:

    iteration += 1

    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)

    cost = compute_cost(y, y_pred)

    error = (y_pred - y) * y_pred * (1 - y_pred)

    dw = (1/m) * np.dot(X.T, error)
    db = (1/m) * np.sum(error)

    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    print(iteration, cost, bias, weights)

    if abs(previous_cost - cost) < tolerance:
        break

    previous_cost = cost

    if iteration >= max_iterations:
        break


print("\nTraining Completed")
print("Weights =", weights)
print("Bias =", bias)
print("Cost =", cost)

predicted = (y_pred >= 0.5).astype(int)

print(predicted)



















#week3
import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def compute_cost(y, y_pred):
    m = len(y)
    epsilon = 1e-10

    cost = -(1/m) * np.sum(
        y * np.log(y_pred + epsilon) +
        (1-y) * np.log(1-y_pred + epsilon)
    )
    return cost


m = int(input("Enter Number of Training Samples : "))
n = int(input("Enter Number of Input Features : "))

X = []
y = []

print("\nEnter Training Data")

for i in range(m):
    print(f"\nSample {i+1}")
    row = []

    for j in range(n):
        value = float(input(f"Enter X{j+1} : "))
        row.append(value)

    target = int(input("Enter Target Output (0/1) : "))
    X.append(row)
    y.append(target)

X = np.array(X)
y = np.array(y)


print("\nEnter Initial Weights")

weights = []

for i in range(n):
    w = float(input(f"Weight W{i+1} : "))
    weights.append(w)

weights = np.array(weights)

bias = float(input("Enter Initial Bias : "))
learning_rate = float(input("Enter Learning Rate : "))
tolerance = float(input("Enter Saturation Tolerance (Example 0.000001): "))

max_iterations = 10000

previous_cost = float('inf')
iteration = 0

while True:

    iteration += 1

    z = np.dot(X, weights) + bias
    y_pred = tanh(z)

    cost = compute_cost(y, y_pred)

    error = (y_pred - y) * y_pred * (1 - y_pred)

    dw = (1/m) * np.dot(X.T, error)
    db = (1/m) * np.sum(error)

    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    print(iteration, cost, bias, weights)

    if abs(previous_cost - cost) < tolerance:
        break

    previous_cost = cost

    if iteration >= max_iterations:
        break


print("\nTraining Completed")
print("Weights =", weights)
print("Bias =", bias)
print("Cost =", cost)

predicted = (y_pred >= 0.5).astype(int)

print(predicted)
















#week 4
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def compute_loss(y, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)

    loss = -np.mean(
        y*np.log(y_pred) +
        (1-y)*np.log(1-y_pred)
    )

    return loss


m = int(input("Enter Number of Training Samples : "))
n = int(input("Enter Number of Input Features : "))

X = []
Y = []

print("\nEnter Training Samples\n")

for i in range(m):

    row = []

    print(f"Sample {i+1}")

    for j in range(n):
        row.append(float(input(f"X{j+1} : ")))

    target = int(input("Target (0/1) : "))

    X.append(row)
    Y.append(target)

X = np.array(X)
Y = np.array(Y).reshape(m,1)


hidden1 = int(input("\nNeurons in Hidden Layer-1 : "))
hidden2 = int(input("Neurons in Hidden Layer-2 : "))

learning_rate = float(input("Learning Rate : "))

tolerance = 1e-7
max_iterations = 10000


np.random.seed(0)

W1 = np.random.randn(n, hidden1) * 0.1
B1 = np.zeros((1, hidden1))

W2 = np.random.randn(hidden1, hidden2) * 0.1
B2 = np.zeros((1, hidden2))

W3 = np.random.randn(hidden2, 1) * 0.1
B3 = np.zeros((1,1))

previous_loss = float("inf")
iteration = 0

print("\nTraining Started\n")
print("{:<10}{:<15}".format("Iteration","Loss"))
print("-"*30)


while True:

    iteration += 1

    Z1 = np.dot(X,W1) + B1
    A1 = relu(Z1)

    Z2 = np.dot(A1,W2) + B2
    A2 = relu(Z2)

    Z3 = np.dot(A2,W3) + B3
    Y_pred = sigmoid(Z3)

    loss = compute_loss(Y,Y_pred)

    print("{:<10}{:<15.8f}".format(iteration,loss))

    if abs(previous_loss-loss) < tolerance:
        break

    previous_loss = loss


    dZ3 = Y_pred - Y

    dW3 = np.dot(A2.T,dZ3)/m
    dB3 = np.sum(dZ3,axis=0,keepdims=True)/m

    dA2 = np.dot(dZ3,W3.T)

    dZ2 = dA2 * relu_derivative(Z2)

    dW2 = np.dot(A1.T,dZ2)/m
    dB2 = np.sum(dZ2,axis=0,keepdims=True)/m

    dA1 = np.dot(dZ2,W2.T)

    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = np.dot(X.T,dZ1)/m
    dB1 = np.sum(dZ1,axis=0,keepdims=True)/m


    W3 -= learning_rate*dW3
    B3 -= learning_rate*dB3

    W2 -= learning_rate*dW2
    B2 -= learning_rate*dB2

    W1 -= learning_rate*dW1
    B1 -= learning_rate*dB1

    if iteration >= max_iterations:
        print("\nMaximum Iterations Reached")
        break


print("\nTraining Completed")

print("\nFinal Loss =",loss)

print("\nPrediction Results")

print("-"*55)

print("{:<10}{:<12}{:<15}{:<10}".format(
    "Sample",
    "Actual",
    "Probability",
    "Predicted"))

print("-"*55)

predicted = (Y_pred >= 0.5).astype(int)

for i in range(m):

    print("{:<10}{:<12}{:<15.6f}{:<10}".format(
        i+1,
        int(Y[i][0]),
        Y_pred[i][0],
        int(predicted[i][0])))

print("-"*55)













#week5
!pip install tensorflow

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Dataset
X, y = make_moons(n_samples=1000, noise=0.20, random_state=42)
print("Dataset Information")
print("-------------------")
print("Number of Samples :", X.shape[0])
print("Number of Features:", X.shape[1])

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function for metrics
def metrics(y_test, prediction):
    return (
        accuracy_score(y_test, prediction),
        precision_score(y_test, prediction),
        recall_score(y_test, prediction),
        f1_score(y_test, prediction)
    )

# Logistic Regression
print("\n" + "="*60)
print("LOGISTIC REGRESSION")
print("="*60)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc, lr_pre, lr_rec, lr_f1 = metrics(y_test, lr_pred)

print("\nLogistic Regression Results")
print("Accuracy  :", round(lr_acc*100, 2), "%")
print("Precision :", round(lr_pre*100, 2), "%")
print("Recall    :", round(lr_rec*100, 2), "%")
print("F1 Score  :", round(lr_f1*100, 2), "%")
print("\nConfusion Matrix")
print(confusion_matrix(y_test, lr_pred))


# DNN
print("\n" + "="*60)
print("DEEP NEURAL NETWORK")
print("="*60)

dnn = Sequential([
    Dense(32, activation="relu", input_shape=(2,)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

dnn.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining DNN...")
dnn.fit(X_train, y_train, epochs=100, batch_size=32,
        validation_split=0.20, verbose=0)
print("DNN Training Completed")

prob = dnn.predict(X_test, verbose=0)
dnn_pred = (prob >= 0.5).astype(int).flatten()
dnn_acc, dnn_pre, dnn_rec, dnn_f1 = metrics(y_test, dnn_pred)

print("\nDeep Neural Network Results")
print("Accuracy  :", round(dnn_acc*100, 2), "%")
print("Precision :", round(dnn_pre*100, 2), "%")
print("Recall    :", round(dnn_rec*100, 2), "%")
print("F1 Score  :", round(dnn_f1*100, 2), "%")
print("\nConfusion Matrix")
print(confusion_matrix(y_test, dnn_pred))



# Comparison
print("\n" + "="*70)
print("LOGISTIC REGRESSION vs DEEP NEURAL NETWORK")
print("="*70)

print("{:<20}{:<20}{:<20}".format(
    "Metric", "Logistic Regression", "Deep Neural Network"))

for name, lr_val, dnn_val in [
    ("Accuracy (%)", lr_acc*100, dnn_acc*100),
    ("Precision (%)", lr_pre*100, dnn_pre*100),
    ("Recall (%)", lr_rec*100, dnn_rec*100),
    ("F1 Score (%)", lr_f1*100, dnn_f1*100)
]:
    print("{:<20}{:<20.2f}{:<20.2f}".format(name, lr_val, dnn_val))

print("="*70)

if dnn_acc > lr_acc:
    print("\nResult: DNN performed better than Logistic Regression.")
elif dnn_acc < lr_acc:
    print("\nResult: Logistic Regression performed better than DNN.")
else:
    print("\nResult: Both models achieved the same accuracy.")


























# =====================================================================================










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




















#week10
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, SimpleRNN, Dense, Dropout

# 1. Set Parameters
VOCAB_SIZE = 10000
MAX_LENGTH = 100
BATCH_SIZE = 64
EPOCHS = 5


tf.keras.utils.set_random_seed(42)


# 2. Load IMDB Dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

X_train, y_train = X_train[:5000], y_train[:5000]
X_test, y_test = X_test[:1000], y_test[:1000]
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# 3. Pad Sequences
X_train = pad_sequences(X_train, maxlen=MAX_LENGTH, padding="post")
X_test = pad_sequences(X_test, maxlen=MAX_LENGTH, padding="post")

# 4. Build Bidirectional RNN
model = Sequential([
    Embedding(VOCAB_SIZE, 32, mask_zero=True),
    Bidirectional(SimpleRNN(32)),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])


# 5. Compile Model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. Train Model
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)


# 7. Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy * 100, 2), "%")


# 8. Predict Sentiment

predictions = model.predict(X_test[:10], verbose=0)

for i in range(10):
    pred = "Positive" if predictions[i][0] >= 0.5 else "Negative"
    actual = "Positive" if y_test[i] == 1 else "Negative"

    print("Review", i + 1, "| Predicted:", pred, "| Actual:", actual)


# 9. Plot Accuracy

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Sentiment Analysis")
plt.legend()
plt.show()





    