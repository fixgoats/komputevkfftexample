import matplotlib.pyplot as plt
import numpy as np


def sqnorm(x):
    return x.real * x.real + x.imag * x.imag


hbar = 6.582e-1
samples = 1024
dt = 10 / samples
kmax = np.pi / dt
dk = 2 * kmax / samples
a = np.loadtxt("build/testsign.csv", dtype=complex)
fwdspect = sqnorm(np.fft.fftshift(a))
fwdidx = np.argmax(fwdspect)
print(f"energy resolution is: {dk}")
print(kmax * (2 * fwdidx / samples - 1))
kaxis = np.arange(-kmax, kmax, dk)
fig, ax = plt.subplots()
b = sqnorm(np.fft.fftshift(a))
d = sqnorm(a)
c = np.argmax(b)
e = np.argmax(d)
print(kmax * (2 * c / samples - 1))
print(2 * kmax * e / samples)
ax.plot(kaxis, fwdspect)
ax.plot(kaxis, b)
fig.savefig("apparentenergies.pdf")
