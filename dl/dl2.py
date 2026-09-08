import numpy as np

# ------------------------------------------
# Sigmoid Activation Function
# ------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ------------------------------------------
# Mean Square Error Cost Function
# ------------------------------------------
def compute_cost(y, y_pred):

    m = len(y)

    return np.sum((y - y_pred)**2) / (2*m)

# ------------------------------------------
# Read Training Data
# ------------------------------------------

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

# ------------------------------------------
# Initial Parameters
# ------------------------------------------

print("\nEnter Initial Weights")

weights = []

for i in range(n):
    w = float(input(f"Weight W{i+1} : "))
    weights.append(w)

weights = np.array(weights)

bias = float(input("Enter Initial Bias : "))

learning_rate = float(input("Enter Learning Rate : "))

tolerance = float(input("Enter Saturation Tolerance (Example 0.000001): "))

# Safety limit
max_iterations = 10000

# ------------------------------------------
# Gradient Descent
# ------------------------------------------

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