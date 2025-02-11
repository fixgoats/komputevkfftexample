import matplotlib.pyplot as plt
import numpy as np

hbar = 6.582e-1
E = 24.5
omega = 14
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


# emax = hbar / dt
kmax = hbar * np.pi / dt
dk = 2 * kmax / samples
print(f"k space resolution is: {dk}")
eaxis = np.arange(-kmax, kmax, dk)
backspect = sqnorm(np.fft.fftshift(backwardtransform))
plt.plot(eaxis, backspect, label="back")
backidx = np.argmax(backspect)
fwdspect = sqnorm(np.fft.fftshift(forwardtransform))
plt.plot(eaxis, fwdspect, label="forward")
print(kmax * (2 * backidx / samples - 1))
forwardidx = np.argmax(fwdspect)
print(kmax * (2 * forwardidx / samples - 1))
print(backspect[backidx])
plt.legend()
plt.savefig("pythonfft.pdf")
