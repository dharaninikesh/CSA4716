import numpy as np

# Inputs (batch of 2 samples, 3 features each)
X = np.array([[0.5, -0.2, 0.1], 
              [0.9, 1.2, -0.4]])

# Weights and Bias
W = np.array([0.4, 0.7, -0.2])
b = -0.1

# Forward Propagation: Z = XW + b
Z = np.dot(X, W) + b

# Step Activation Function (Perceptron rule)
A = np.where(Z >= 0, 1, 0)

print("Pre-activation (Z):", Z)
print("Perceptron Output (A):", A)
