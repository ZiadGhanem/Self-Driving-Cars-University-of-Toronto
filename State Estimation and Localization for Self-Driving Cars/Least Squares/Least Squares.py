import numpy as np
import matplotlib.pyplot as plt

I = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
V = np.array([1.23, 1.38, 2.06, 2.47, 3.17])

m = I.shape[0]
n = 1

H = np.ones((m, n+1))
H[:, 0] = I

x_ls = np.dot(np.linalg.inv(np.dot(H.T, H)), np.dot(H.T, V))

I_line = np.arange(0, 0.8, 0.1)
V_line = x_ls[0]*I_line + x_ls[1]
plt.scatter(I, V)
plt.plot(I_line, V_line)
plt.plot()
plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.show()