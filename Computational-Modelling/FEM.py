import numpy as np
import matplotlib.pyplot as plt
import math

size = 100000
start = 70000

n = 4
sep =int((size-start)/(n-1))

x =np.linspace(-start/(size-start),1, size)
ramp = np.linspace(0,0, size)
for i in range(size):
    ramp[i] = max(min(1,x[i]*(n-1)),0)
# plt.plot(x,ramp)
# plt.show()
ramp = np.roll(ramp,-sep)
ramp[start:] = np.ones(size-start)

ts = []

# PDE: Uxx + 2Ux + U = 3x^2
# BCs: U(-1) = -18, U(1) = -6
# Domain: -1 < x < 1

xs = np.linspace(-1,1,1000)

# Weighting functions
t0 = math.sqrt(2)/2*np.ones(1000)
t1 = math.sqrt(6)/2*xs
t2 = math.sqrt(90)/4*(2*xs**2-1)
t3 = math.sqrt(1190)/34*(4*xs**3-3*xs)
ts = [t0,t1,t2,t3]


for i in range(n):
    # ts.append(np.roll(ramp, sep*i)[start:] - np.roll(ramp, sep*(i+1))[start:])
    plt.plot(xs, ts[i])
plt.show()


# Derivatives of W (Wx)
dts = []
for i in range(n):
    dts.append(np.gradient(ts[i],xs))
    plt.plot(xs, dts[i])
plt.show()


K = np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        K[i,j] = np.trapz(ts[i]*(2*dts[j]+ts[j]),x=xs)-np.trapz(dts[i]*dts[j],x=xs)

F = np.zeros(n)
for i in range(0,n):

    F[i] = np.trapz(3*xs**2*ts[i], x=xs)-(ts[i][-1]*-6)+(ts[i][0]*-18)

A = np.linalg.solve(K,F)

print(K)
print(F)
print(A)

u = np.zeros(1000)

for i in range(len(A)):
    u+= A[i] * ts[i]

plt.plot(xs,u)
plt.show()

dudx =  np.gradient(u,xs)
print('dudx(-1)= ',dudx[0], ' dudx(1)= ', dudx[-1])
dudxx = np.gradient(dudx,xs)

plt.plot(xs,(dudxx+2*dudx+u))
plt.plot(xs,3*xs**2)

plt.show()