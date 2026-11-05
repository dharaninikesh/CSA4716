from sklearn.neural_network import MLPClassifier
import numpy as np

# Toy dataset (XOR gate classification)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Multi-Layer Perceptron: 8 hidden neurons with L-BFGS for flawless convergence on small datasets
mlp = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', solver='lbfgs', max_iter=2000, random_state=42)
mlp.fit(X, y)

print("Predictions:", mlp.predict(X))
print("Accuracy:", mlp.score(X, y))
