import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
import argparse
from tqdm import tqdm
import mpl_toolkits.axes_grid1 as axgrid1



print('Loading data...')
data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Processed_data/4_Gain_function/4_gain_function.txt'
data = np.loadtxt(data_path, delimiter=' ')
print('Data shape:', data.shape)


#* Parameters
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]






sig = data[:, 500]
tm = np.arange(0, len(sig)*sample_interval, sample_interval)

f, t, Sxx = signal.spectrogram(sig, 1/sample_interval, nperseg=1024, noverlap=512, nfft=1024, mode='magnitude')
t = t / 1e-9 # [ns]
f = f / 1e6 # [MHz]
Sxx = 10 * np.log10(Sxx/np.amax(Sxx))

#* Plot the spectrogram
fig, ax = plt.subplots(figsize = (10, 8))
im = ax.imshow(Sxx, aspect='auto',
                extent=[t[0], t[-1], f[0], f[-1]],
                cmap='jet', origin='lower',
                vmin=-30, vmax=0)
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [s]')
ax.set_title('Spectrogram')
#ax.set_ylim(0, 2000)

cbar = fig.colorbar(im, ax=ax)

plt.show()
