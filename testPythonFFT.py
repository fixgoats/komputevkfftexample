import numpy as np
import matplotlib.pyplot as plt

hbar = 6.582e-1
E = 24.5
tstart = 0
tend = 10
samples = 1024
dt = (tend - tstart) / samples
ts = np.arange(tstart, tend, dt)
testarray = np.exp(-1j * (E * ts / hbar))

backwardtransform = np.fft.ifft(testarray)
forwardtransform = np.fft.fft(testarray)


def sqnorm(x):
    return x.real * x.real + x.imag * x.imag


emax = hbar / dt
de = 2 * emax / samples
eaxis = np.arange(-emax, emax, de)
backspect = sqnorm(np.fft.fftshift(backwardtransform))
plt.plot(eaxis, backspect, label="back")
backidx = np.argmax(backspect)
fwdspect = sqnorm(np.fft.fftshift(forwardtransform))
plt.plot(eaxis, fwdspect, label="forward")
print(emax * (2 * backidx / samples - 1))
forwardidx = np.argmax(fwdspect)
print(emax * (2 * forwardidx / samples - 1))
print(backspect[backidx])
plt.legend()
plt.savefig("pythonfft.pdf")
