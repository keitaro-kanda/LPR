import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from bandpass import processing_filtering
from remove_background import processing_background_removal
from time_zero_correction import proccessing_time_zero_correction
import argparse
from tqdm import tqdm

#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='run_data_processing.py',
    description='Run bandpass filtering, time-zero correction, and background removal on resampled Bscan data',
    epilog='End of help message',
    usage='python tools/resampling.py [path_type]',
)
parser.add_argument('path_type', choices = ['local', 'SSD'], help='Choose the path type')
args = parser.parse_args()


#* Define data folder path
if args.path_type == 'local' or args.path_type == 'test':
    data_path = 'LPR_2B/Resampled_ECHO/Bscan.txt'
elif args.path_type == 'SSD':
    data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Resampled_ECHO/Bscan.txt'
print('Data path:', data_path)


#* load raw Bscan data
#data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Raw_Bscan/Bscan.txt'
Raw_Bscan = np.loadtxt(data_path, delimiter=' ')
print("Bscan shape:", Raw_Bscan.shape)
output_dir = os.path.join(os.path.dirname(os.path.dirname(data_path)), 'Processed_Bscan')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print('Output dir:', output_dir)



#* process bandpass filter
bandpass_filtering = processing_filtering(Raw_Bscan)
sample_interval = bandpass_filtering.sample_interval

filtered_Bscan = np.zeros(Raw_Bscan.shape)
for i in tqdm(range(Raw_Bscan.shape[1]), desc='Applying bandpass filter'):
            filtered_Bscan[:, i] = bandpass_filtering.bandpass_filter(Raw_Bscan[:, i], 250e6, 750e6, 5)
print('Finished bandpass filtering')
(filtered_Bscan.shape)



#* process time zero correction
time_zero_correction = proccessing_time_zero_correction(filtered_Bscan)
peak_time = time_zero_correction.find_peak_time()
aligned_Bscan = time_zero_correction.align_peak_time()
#aligned_Bscan = time_zero_correction.zero_corrections()
print('Finished time-zero correction')
print(aligned_Bscan.shape)



#* process background removal
background_removal = processing_background_removal(aligned_Bscan)
background_removed_Bscan = background_removal.subtract_background()
print('Finished background removal')
print(background_removed_Bscan.shape)


#* Process gain function
t = np.expand_dims(np.linspace(0, background_removed_Bscan.shape[0] *sample_interval, background_removed_Bscan.shape[0]), axis=1)
gained_Bscan = background_removed_Bscan * t ** 1.7
#gained_Bscan = gained_Bscan / np.max(gained_Bscan)


plot_data = [Raw_Bscan, filtered_Bscan, aligned_Bscan, background_removed_Bscan, gained_Bscan]
title = ['Raw B-scan', 'Bandpass filtered B-scan', 'Time-zero corrected B-scan', 'Background removed B-scan', 'Gained B-scan']


sample_interval_ns = sample_interval * 1e9
#* plot 4 figures
fig, ax = plt.subplots(len(plot_data), 1, figsize=(18, 20), tight_layout=True, sharex=True)
fontsize_large = 20
fontsize_medium = 18
fontsize_small = 16

for i in range(len(plot_data)):
    imshow = ax[i].imshow(plot_data[i], aspect='auto', cmap='seismic',
                extent=[0, plot_data[i].shape[1], plot_data[i].shape[0]*sample_interval_ns, 0],
                vmin=-15, vmax=15
                )
    ax[i].tick_params(axis='both', which='major', labelsize=fontsize_small)
    ax[i].set_title(title[i], fontsize=fontsize_large)

    if i == len(plot_data) - 1:
        ax[i].set_xlabel('Trace number', fontsize=fontsize_medium)
    #ax[i].set_ylim(50, 0)


#fig.supxlabel('Trace number', fontsize=fontsize_medium)
fig.supylabel('Time (ns)', fontsize=fontsize_medium)


#* plot colorbar
delvider = axgrid1.make_axes_locatable(ax[len(plot_data)-1])
cax = delvider.append_axes('bottom', size='5%', pad=1)
plt.colorbar(imshow, cax=cax, orientation = 'horizontal').set_label('Amplitude', fontsize=fontsize_medium)
cax.tick_params(labelsize=fontsize_small)


#* save plot
plt.savefig(output_dir + '/Processed_Bscan.png')

plt.show()