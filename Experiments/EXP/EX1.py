import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])

print("Addition:", a+b)
print("Subtraction:", a-b)
print("Dot Product:", np.dot(a,b))

print("Matrix:", a.reshape(1,3))
print("Transpose:", a.T)
