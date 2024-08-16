"""
This code make hall B-scan plot from resampled ECHO data.
If you want to make B-scan plot of each sequence, you can use resampling.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm
from scipy import signal

data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Processed_data/4_Gain_function/4_Bscan_gain.txt'
#* Define output folder path
output_dir = os.path.join(os.path.dirname(data_path), 'Hilbert')
os.makedirs(output_dir, exist_ok=True)

print('Loading data...')
print('   ')
Bscan_data = np.loadtxt(data_path, delimiter=' ')
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]



#* Referrence: https://labo-code.com/python/hilbert-transform/
#*Hilbert transform
jsgn = np.sign(np.fft.fftfreq(len(Bscan_data), sample_interval))*1.0j # 1/(pi*t)のフーリエ変換のマイナス

hsig = np.zeros(Bscan_data.shape)
for i in tqdm(range(Bscan_data.shape[1]), desc='Hilbert transform'):
    hsig[:, i] = np.fft.ifft(-jsgn*np.fft.fft(Bscan_data[:, i])) # フーリエ空間で積を取りフーリ逆変換 (畳み込み積分の計算)

#* Calculate the envelope
envelope_data = np.sqrt(Bscan_data**2 + hsig**2)

#* Calculate the instantaneous phase
inst_phase = np.arctan2(np.real(hsig), Bscan_data)

#* Calculate the instantaneous frequency
inst_freq = np.zeros(inst_phase.shape)
for i in tqdm(range(inst_phase.shape[1]), desc='Instantaneous frequency'):
    inst_freq[:, i] = np.gradient(np.unwrap(inst_phase[:, i])) / sample_interval/(2.0*np.pi) / 1e6 # 瞬時周波数

"""
#* Calculate envelove
def envelope(data):
    #* Calculate the envelope of the data
    envelope = np.abs(signal.hilbert(data, axis=0))
    return envelope
envelope_data = envelope(Bscan_data)


#* Calculate the instantaneous frequency
inst_phase = np.zeros((Bscan_data.shape[0], Bscan_data.shape[1]))
for i in tqdm(range(Bscan_data.shape[1])):
    inst_phase[:, i] = np.angle(signal.hilbert(envelope_data[:, i]))
inst_freq = np.zeros((Bscan_data.shape[0], Bscan_data.shape[1]-1))
for i in range(inst_phase.shape[1]-1):
    inst_freq[:, i] = np.gradient(np.unwrap(inst_phase[:, i])) / sample_interval / (2*np.pi) / 1e6 # [MHz]
"""


#* Mask the inst_phase
inst_phase_mask = np.zeros(inst_phase.shape)
for i in range(inst_phase.shape[1]):
    inst_phase_mask[:, i] = inst_phase[:, i] * envelope_data[:, i]


envelope_data = 10 * np.log10(envelope_data/np.amax(envelope_data))
#* Plot
plot_list = [envelope_data, inst_phase, inst_phase_mask, inst_freq]
plot_name = ['envelope', 'inst_phase', 'inst_phase_mask', 'inst_freq']
cmaps = ['jet', 'seismic', 'seismic', 'jet']
vmin = [np.amin(envelope_data)/2, -np.pi, -np.amax(np.abs(inst_phase_mask))/5, 250]
vmax = [0, np.pi, np.amax(np.abs(inst_phase_mask))/5, 750]
cbar_label = ['Envelope', 'Instantaneous phase', 'Masked instantaneous phase', 'Instantaneous frequency [MHz]']

font_large = 20
font_medium = 18
font_small = 16

for i, plot in enumerate(plot_list):
    fig, ax = plt.subplots(figsize=(18, 6))
    plt.imshow(plot,
                extent=[0, plot.shape[1] * trace_interval, plot.shape[0] * sample_interval / 1e-9, 0],
                interpolation='nearest', aspect='auto', cmap=cmaps[i],
                vmin=vmin[i], vmax=vmax[i])
    ax.set_xlabel('x [m]', fontsize=20)
    ax.set_ylabel('Time [ns]', fontsize=20)
    ax.tick_params(labelsize=18)
    ax.grid(True, linestyle='-.', linewidth=0.5)

    delvider = axgrid1.make_axes_locatable(ax)
    cax = delvider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(cax=cax)
    cbar.set_label(cbar_label[i], size=20)
    cbar.ax.tick_params(labelsize=16)

    plt.savefig(os.path.join(output_dir, plot_name[i] + '.png'), format='png', dpi=120)
    plt.savefig(os.path.join(output_dir, plot_name[i] + '.pdf'), format='pdf', dpi=600)
    plt.show()