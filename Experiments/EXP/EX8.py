import numpy as np

# Sigmoid activation and its derivative
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

# Data: XOR problem inputs and targets
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Network Architecture (2 inputs, 3 hidden neurons, 1 output)
np.random.seed(42)
W1 = np.random.randn(2, 3)
W2 = np.random.randn(3, 1)

# 1 Step of Training (Forward & Backpropagation)
# Forward Pass
h_input = np.dot(X, W1)
h_out = sigmoid(h_input)
o_input = np.dot(h_out, W2)
o_out = sigmoid(o_input)

# Backpropagation
error = y - o_out
d_o_out = error * sigmoid_derivative(o_out)

error_h = d_o_out.dot(W2.T)
d_h_out = error_h * sigmoid_derivative(h_out)

# Weight Updates
W2 += h_out.T.dot(d_o_out) * 0.1
W1 += X.T.dot(d_h_out) * 0.1

print("Predicted Output:\n", o_out)
print("Loss (MSE):", np.mean(error**2))
