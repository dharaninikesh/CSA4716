import numpy as np
from scipy.stats import entropy

# Define two probability distributions
p = np.array([0.4, 0.6])
q = np.array([0.5, 0.5])

# 1. Entropy of P
ent = entropy(p)

# 2. Cross-Entropy
cross_ent = entropy(p) + entropy(p, q)

# 3. KL Divergence D_KL(P || Q)
kl_div = entropy(p, q)

print(f"Entropy of P: {ent:.4f}")
print(f"Cross-Entropy: {cross_ent:.4f}")
print(f"KL Divergence: {kl_div:.4f}")
