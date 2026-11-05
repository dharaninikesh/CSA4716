import numpy as np

A = np.array([[1, 2], [2, 3], [3, 4]])

C = np.cov(A.T)
print("Covariance:\n", C)

v, e = np.linalg.eig(C)

print("Eigenvalues:", v)
print("Eigenvectors:\n", e)
