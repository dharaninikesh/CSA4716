import numpy as np
import matplotlib.pyplot as plt

# Generate noisy quadratic data
np.random.seed(0)
x = np.linspace(-3, 3, 10)
y = x**2 + np.random.normal(0, 1, 10)

# Fit 3 models: Underfit (degree 1), Good fit (degree 2), Overfit (degree 9)
x_plot = np.linspace(-3, 3, 100)
plt.scatter(x, y, color='red', label='Data')

for deg, label in zip([1, 2, 9], ['Underfit (d=1)', 'Good Fit (d=2)', 'Overfit (d=9)']):
    coefs = np.polyfit(x, y, deg)
    plt.plot(x_plot, np.polyval(coefs, x_plot), label=label)

plt.ylim(-2, 12)
plt.legend()
plt.show()
