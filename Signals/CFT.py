# Author: Max Martinez Ruts
# Creation: 2019
# Title: Discrete Fourier Transform

import numpy as np
import matplotlib.pyplot as plt
import math


T = 2
fs = 20
fc1 = 4

N = T * fs
print(N)

t = np.linspace(-T/2, T/2, N)
f  = np.arange(-fs/2, fs/2, fs/N)


# Define x
x = np.cos(2*math.pi*fc1*t)

plt.plot(t, x,'ro' )
plt.show()

while True:
    # Comparison Fourier Transform cointinuous

    X = np.zeros(N, dtype=complex)
    for k in range(len(X)):
        for n in range(N):
            X[k] += x[n] * np.exp(-1j * 2 * math.pi * n * T / N * f[k])
    plt.plot(f, np.real(X))
    plt.show()

