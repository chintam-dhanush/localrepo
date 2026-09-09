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