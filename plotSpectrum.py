import matplotlib.pyplot as plt
import numpy as np


def sqnorm(x):
    return x.real * x.real + x.imag * x.imag


samples = 1024
a = np.loadtxt("build/testsign.csv", dtype=complex)
dt = 10 / samples
emax = 6.582e-1 / dt
de = 2 * emax / samples
print(f"energy resolution is: {de}")
eaxis = np.arange(-emax, emax, de)
fig, ax = plt.subplots()
b = sqnorm(np.fft.fftshift(a))
d = sqnorm(a)
c = np.argmax(b)
e = np.argmax(d)
print(emax * (2 * c / samples - 1))
print(2 * emax * e / samples)
ax.plot(eaxis, a.real * a.real + a.imag * a.imag)
ax.plot(eaxis, b)
fig.savefig("apparentenergies.pdf")
