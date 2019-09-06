import numpy as np
import matplotlib.pyplot as plt
import math

K = 1000
T0 = 1
A = 1
w0 = 2* math.pi/ T0

t = np.linspace(-T0/2, T0/2, 1000)

# Define x
x = np.zeros((len(t)))
x = np.where(t < -0.25, -t, x)
x = np.where(t > -0.25, t, x)
x = np.where(t > 0.25, -t, x)

plt.plot(t, x)
plt.show()

for j in range(3, 2001):

    # Declare coefficients
    a =  np.zeros((j))
    b = np.zeros((j))

    # Determine a0
    a[0] = np.mean(x)

    # Define f
    f = np.linspace(0,0,1000)
    f += a[0] * np.cos(0 * w0 * t)

    for m in range(1,j):
        # Determin coefficients
        a[m] = np.sum(x*np.cos(m*w0*t))/1000*2/T0
        b[m] = np.sum(x*np.sin(m*w0*t))/1000*2/T0

        # Add sine  / cosine contribution
        f += a[m]*np.cos(m*w0*t)
        f += b[m]*np.sin(m*w0*t)

    plt.plot(t,x)
    plt.plot(t,f)
    plt.show()


