import json
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
import argparse
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline
import json


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='matched_filter.py',
    description='Apply matched filter to the data',
    epilog='End of help message',
    usage='python tools/fk_migration.py [function_type]',
)
parser.add_argument('function_type', choices=['calc', 'plot'], help='Choose the function type')
args = parser.parse_args()


#* Data path
if args.function_type == 'calc':
    data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Processed_Data/4_Gain_function/4_Bscan_gain.txt'
    #* Define output folder path
    output_dir = os.path.join(os.path.dirname(data_path), 'Matched_filter')
    os.makedirs(output_dir, exist_ok=True)
elif args.function_type == 'plot':
    data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Processed_Data/4_Gain_function/Matched_filter/matched_filter.txt'
    #* Define output folder path
    output_dir = os.path.dirname(data_path)


#* Load data
print('Loading data...')
data = np.loadtxt(data_path, delimiter=' ')


#* Parameters
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]



#* Define the function to calculate the gaussian wave
def gaussian(t_array, mu, sigma):
        a = 1 / (sigma * np.sqrt(2 * np.pi))
        b = np.exp(-0.5 * (((t_array - mu) / sigma) ** 2))
        return a * b


#* Define the function to calculate the matched filter
def maeched_filter(Ascan):
    t = np.arange(0, 10e-9, sample_interval)
    reference_sig = gaussian(t, 2.8e-9, 0.3e-9)
    reference_sig = np.concatenate([np.flip(reference_sig), np.zeros(len(Ascan) - len(reference_sig))])
    fft_refer = np.fft.fft(np.conj(reference_sig))
    fft_data = np.fft.fft(Ascan)
    conv = np.fft.ifft(fft_refer * fft_data)
    return np.real(conv)



#* Apply the matched filter to the data
data_matched = np.zeros(data.shape)
for i in tqdm(range(data.shape[1]), desc='Calculating matched filter'):
    data_matched[:, i] = maeched_filter(data[:, i])

data_matched = np.abs(data_matched)
np.savetxt(os.path.join(output_dir, 'matched_filter.txt'), data_matched, delimiter=',')



data_matched = 10 * np.log10(data_matched / np.max(data_matched))


#* Plot
print('Plotting...')
plt.figure(figsize=(18, 6), facecolor='w', edgecolor='w')
im = plt.imshow(data_matched, cmap='jet', aspect='auto',
                extent=[0, data_matched.shape[1] * trace_interval,
                data_matched.shape[0] * sample_interval, 0],
                vmin=-35, vmax=0
                )

plt.xlabel('x [m]', fontsize=20)
plt.ylabel('Time [m]', fontsize=20)
plt.tick_params(labelsize=18)


delvider = axgrid1.make_axes_locatable(plt.gca())
cax = delvider.append_axes('right', size='5%', pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Amplitude [dB]', fontsize=20)
cbar.ax.tick_params(labelsize=18)


plt.savefig(os.path.join(output_dir, 'matched_filter.png'), format='png', dpi=120)
plt.savefig(os.path.join(output_dir, 'matched_filter.pdf'), format='pdf', dpi=600)
plt.show()