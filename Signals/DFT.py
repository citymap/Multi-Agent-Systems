# Author: Max Martinez Ruts
# Creation: 2019
# Title: Discrete Fourier Transform

import numpy as np
import matplotlib.pyplot as plt
import math

K = 1000
T = 2
fs = 13
fc1 = 4
fc2 = 7

N = T * fs
print(N)

t = np.linspace(0, T, K)
t_d = np.arange(0,T, T/N)

f  = np.arange(-fs/2, fs/2, fs/N)
# fs =

# Define x
x = np.cos(2*math.pi*fc1*t)+np.cos(2*math.pi*fc2*t)
x_d = np.cos(2*math.pi*fc1*t_d)+np.cos(2*math.pi*fc2*t_d)


plt.plot(t, x,)
plt.plot(t_d, x_d ,'go')

plt.show()

X =  np.zeros(N, dtype=complex)
for k in range(len(X)):
    for n in range(N):
        X[k] += x_d[n] * np.exp(-1j * 2 * math.pi * n * T / N * f[k])
# plt.plot(f,np.real(X), 'go')
plt.plot(f,np.imag(X), 'bo')
plt.plot(f,np.abs(X), 'rx')
plt.show()

x_r =  np.zeros(N, dtype=complex)
for n in range(N):
    for k in range(N):
        x_r[n] +=X[k]*np.exp(1j*2*math.pi*n*T/N*f[k])
x_r *= 1/N
print(x_r)
plt.plot(t,x)
plt.plot(t_d,x_r, 'ro')
plt.show()

# Comparison Fourier Transform cointinuous

# f_c = np.arange(-fs / 2, fs / 2, fs / K)
# X = np.zeros(K, dtype=complex)
# for k in range(len(X)):
#     for n in range(K):
#         X[k] += x[n] * np.exp(-1j * 2 * math.pi * n * T / K * f_c[k])
# plt.plot(f_c, np.abs(X))
# plt.show()

