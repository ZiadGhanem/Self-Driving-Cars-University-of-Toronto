import numpy as np
import matplotlib.pyplot as plt

I = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
V = np.array([1.23, 1.38, 2.06, 2.47, 3.17])

m = I.shape[0]
n = 1

# Covariance Matrix
P_k = np.array([[4, 10.0], [0, 0.2]])

# Initialize parameter estimate
x_k = np.array([4.0, 0])

R_k = 0.0225

for k in range(m):
    H_k = np.array([[I[k], 1.0]])
    K_k = np.dot(P_k, np.dot(H_k.T, np.linalg.inv(np.dot(H_k, np.dot(P_k, H_k.T) + R_k))))
    x_k = x_k + np.dot(K_k, (V[k] - np.dot(H_k, x_k)))
    P_k = np.dot((np.eye(n+1, n+1) - np.dot(K_k, H_k)), P_k)

    print(x_k)

