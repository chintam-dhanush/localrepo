import numpy as np

# -----------------------------------------
# Activation Functions
# -----------------------------------------

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# -----------------------------------------
# Binary Cross Entropy Loss
# -----------------------------------------

def compute_loss(y, y_pred):

    epsilon = 1e-10

    y_pred = np.clip(y_pred, epsilon, 1-epsilon)

    loss = -np.mean(
            y*np.log(y_pred) +
            (1-y)*np.log(1-y_pred)
            )

    return loss

# -----------------------------------------
# Read Dataset
# -----------------------------------------

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

# -----------------------------------------
# Network Architecture
# -----------------------------------------

hidden1 = int(input("\nNeurons in Hidden Layer-1 : "))
hidden2 = int(input("Neurons in Hidden Layer-2 : "))

learning_rate = float(input("Learning Rate : "))

tolerance = 1e-7

max_iterations = 10000

# -----------------------------------------
# Weight Initialization
# -----------------------------------------

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

# -----------------------------------------
# Training
# -----------------------------------------

while True:

    iteration += 1

    # Forward Propagation

    Z1 = np.dot(X,W1) + B1
    A1 = relu(Z1)

    Z2 = np.dot(A1,W2) + B2
    A2 = relu(Z2)

    Z3 = np.dot(A2,W3) + B3
    Y_pred = sigmoid(Z3)

    # Loss

    loss = compute_loss(Y,Y_pred)

    print("{:<10}{:<15.8f}".format(iteration,loss))

    # Saturation Check

    if abs(previous_loss-loss) < tolerance:
        break

    previous_loss = loss

    # -----------------------------------------
    # Backpropagation
    # -----------------------------------------

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

    # -----------------------------------------
    # Update Parameters
    # -----------------------------------------

    W3 -= learning_rate*dW3
    B3 -= learning_rate*dB3

    W2 -= learning_rate*dW2
    B2 -= learning_rate*dB2

    W1 -= learning_rate*dW1
    B1 -= learning_rate*dB1

    if iteration >= max_iterations:
        print("\nMaximum Iterations Reached")
        break

# -----------------------------------------
# Final Prediction
# -----------------------------------------

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