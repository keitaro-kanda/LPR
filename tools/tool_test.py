import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Filter specifications
lowcut = 250e6
highcut = 750e6
sample_interval = 0.3125e-9  # Sample interval
fs = 1/sample_interval  # Sampling frequency

# Design the Butterworth band-pass filter
order = 4
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist

b, a = signal.butter(order, [low, high], btype='band')

# Frequency response
w, h = signal.freqz(b, a, worN=8000)
frequencies = (w / np.pi) * nyquist

# Plot the frequency response
plt.figure(figsize=(10, 6))
plt.plot(frequencies / 1e6, abs(h), 'b')
plt.title('Band-pass Filter Frequency Response')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Gain')
plt.axvline(lowcut / 1e6, color='k', linestyle='--')
plt.axvline(highcut / 1e6, color='k', linestyle='--')
plt.axvline(150, color='r', linestyle='--')
plt.axvline(850, color='r', linestyle='--')
plt.ylim(0, 1.2)
plt.grid()
plt.show()
