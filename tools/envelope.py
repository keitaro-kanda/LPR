"""
This code make hall B-scan plot from resampled ECHO data.
If you want to make B-scan plot of each sequence, you can use resampling.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
import argparse
from natsort import natsorted
from scipy import signal
import matplotlib.colors as colors

data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Processed_data/4_Gain_function/4_gain_function.txt'
#* Define output folder path
output_dir = os.path.join(os.path.dirname(data_path), 'envelope')
os.makedirs(output_dir, exist_ok=True)

print('Loading data...')
print('   ')
Bscan_data = np.loadtxt(data_path, delimiter=' ')
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]

skip_sec = 0
skip = int(skip_sec/sample_interval)
Bscan_data = Bscan_data[skip:, :]
print('Shape of B-scan after skipping', Bscan_data.shape)



#* Calculate envelove
def envelope(data):
    #* Calculate the envelope of the data
    envelope = np.abs(signal.hilbert(data, axis=0))
    return envelope


envelope_data = envelope(Bscan_data)
envelope_data = 10 * np.log10(envelope_data)

#* plot
font_large = 20
font_medium = 18
font_small = 16

plt.figure(figsize=(18, 6), tight_layout=True)
plt.imshow(envelope_data, aspect='auto', cmap='jet',
        extent=[0, envelope_data.shape[1]*trace_interval, envelope_data.shape[0]*sample_interval*1e9, 0],
        vmin=-30, vmax=0
        )
plt.xlabel('Distance [m]', fontsize=font_large)
plt.ylabel('Time [ns]', fontsize=font_large)
plt.tick_params(axis='both', which='major', labelsize=font_medium)

#* Colorbar
delvider = axgrid1.make_axes_locatable(plt.gca())
cax = delvider.append_axes('right', size='3%', pad=0.1)
plt.colorbar(cax=cax, orientation = 'vertical').set_label('Amplitude', fontsize=font_large)
cax.tick_params(labelsize=font_small)

plt.savefig(output_dir + '/envelope.png', dpi=120)
plt.savefig(output_dir + '/envelope.pdf', format='pdf', dpi=600)
plt.show()