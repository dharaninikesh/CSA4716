import numpy as np

# f(w) = w^2, grad = 2w
lr = 0.1

# 1. Gradient Descent / SGD (on constant gradient)
w = 10.0
for _ in range(5): w -= lr * (2 * w)
print(f"GD/SGD final w: {w:.4f}")

# 2. Momentum
w, v = 10.0, 0.0
beta = 0.9
for _ in range(5):
    v = beta * v + lr * (2 * w)
    w -= v
print(f"Momentum final w: {w:.4f}")

# 3. Adam
w, m, v = 10.0, 0.0, 0.0
beta1, beta2, eps = 0.9, 0.999, 1e-8
for t in range(1, 6):
    g = 2 * w
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    w -= lr * m_hat / (np.sqrt(v_hat) + eps)
print(f"Adam final w: {w:.4f}")
