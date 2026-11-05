
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Probability Distribution (Plot Normal Distribution)
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, 0, 1), label='Normal PDF')
plt.title("Normal Distribution")
plt.legend()
plt.show()

# 2. Bayes' Theorem Calculation
# P(A|B) = (P(B|A) * P(A)) / P(B)
p_a = 0.01      # Prior probability of disease
p_b_given_a = 0.95  # True positive rate (sensitivity)
p_b_given_not_a = 0.05 # False positive rate

p_b = (p_b_given_a * p_a) + (p_b_given_not_a * (1 - p_a))  # Total probability of positive test
p_a_given_b = (p_b_given_a * p_a) / p_b                   # Posterior probability

print(f"Posterior Probability P(Disease|Positive Test): {p_a_given_b:.4f}")
