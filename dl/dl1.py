import math

print("="*60)
print(" SINGLE NEURON  PERCEPTRON ")
print("="*60)

# Number of inputs
n = int(input("\nEnter number of inputs: "))

# Read inputs
x = []
print("\nEnter Input Values:")
for i in range(n):
    x.append(float(input(f"X{i+1}: ")))

# Read weights
w = []
print("\nEnter Corresponding Weights:")
for i in range(n):
    w.append(float(input(f"W{i+1}: ")))

# Bias
b = float(input("\nEnter Bias: "))
# -------------------------------
# Normalization Selection
# -------------------------------

print("\nChoose Normalization Method")
print("0. No Normalization")
print("1. Min-Max Normalization (0 to 1)")
print("2. Min-Max Normalization (-1 to 1)")
print("3. Z-Score Standardization (Mean=0, Std=1)")
print("4. Decimal Scaling")

norm = int(input("\nEnter choice (0-4): "))

x_norm = x.copy()

if norm == 1:

    xmin = float(input("\nEnter Minimum Value : "))
    xmax = float(input("Enter Maximum Value : "))

    for i in range(n):
        x_norm[i] = (x[i]-xmin)/(xmax-xmin)

    norm_name = "Min-Max (0 to 1)"
    norm_range = "[0,1]"

elif norm == 2:

    xmin = float(input("\nEnter Minimum Value : "))
    xmax = float(input("Enter Maximum Value : "))

    for i in range(n):
        x_norm[i] = 2*((x[i]-xmin)/(xmax-xmin))-1

    norm_name = "Min-Max (-1 to 1)"
    norm_range = "[-1,1]"

elif norm == 3:

    mean = sum(x)/n

    std = (sum((i-mean)**2 for i in x)/n)**0.5

    for i in range(n):
        x_norm[i]=(x[i]-mean)/std if std!=0 else 0

    norm_name = "Z-Score Standardization"
    norm_range = "Mean=0, Std=1"

elif norm == 4:

    j=int(input("\nEnter Decimal Scaling Factor (j): "))

    for i in range(n):
        x_norm[i]=x[i]/(10**j)

    norm_name="Decimal Scaling"
    norm_range="Depends on j"

else:

    norm_name="No Normalization"
    norm_range="Original Values"

# Activation Function
print("\nChoose Activation Function")
print("1. Linear")
print("2. Binary Step")
print("3. Sigmoid")
print("4. Tanh")
print("5. ReLU")
print("6. Leaky ReLU")
print("7. ELU")
print("8. Softplus")

choice = int(input("\nEnter choice (1-8): "))

# -------------------------------
# Without Normalization
# -------------------------------

z = sum(x[i]*w[i] for i in range(n)) + b

# -------------------------------
# With Normalization
# -------------------------------

z_norm = sum(x_norm[i]*w[i] for i in range(n)) + b

def activation(value, choice):

    if choice==1:
        return value

    elif choice==2:
        return 1 if value>=0 else 0

    elif choice==3:
        return 1/(1+math.exp(-value))

    elif choice==4:
        return math.tanh(value)

    elif choice==5:
        return max(0,value)

    elif choice==6:
        alpha=0.01
        return value if value>0 else alpha*value

    elif choice==7:
        alpha=1
        return value if value>0 else alpha*(math.exp(value)-1)

    elif choice==8:
        return math.log(1+math.exp(value))


a = activation(z,choice)
a_norm = activation(z_norm,choice)

print("\n")
print("="*70)
print("                RESULTS")
print("="*70)

print("\nOriginal Inputs")

for i in range(n):
    print(f"X{i+1} = {x[i]}")

print("\nNormalized Inputs")

for i in range(n):
    print(f"X{i+1} = {x_norm[i]:.6f}")

print("\nNormalization Method :",norm_name)
print("Output Range         :",norm_range)

print("\nWeights")

for i in range(n):
    print(f"W{i+1} = {w[i]}")

print("\nBias =",b)

print("\n")
print("-"*70)
print("WITHOUT NORMALIZATION")
print("-"*70)

print("Weighted Sum (Z) =",round(z,6))
print("Activation Output =",round(a ,6))

print("\n")
print("-"*70)
print("WITH NORMALIZATION")
print("-"*70)

print("Weighted Sum (Z) =",round(z_norm,6))
print("Activation Output =",round(a_norm,6))

print("\n")
print("="*70)