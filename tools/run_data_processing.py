import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from bandpass import processing_filtering
from remove_background import processing_background_removal
from time_zero_correction import proccessing_time_zero_correction
import argparse

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


#* process time zero correction
time_zero_correction = proccessing_time_zero_correction(Raw_Bscan)
peak_time = time_zero_correction.find_peak_time()
aligned_Bscan = time_zero_correction.align_peak_time()
print('Finished time-zero correction')
print(aligned_Bscan.shape)


#* process bandpass filter
bandpass_filtering = processing_filtering(aligned_Bscan)
sample_interval = bandpass_filtering.sample_interval
filtered_Bscan = bandpass_filtering.apply_bandpass_filter()
print('Finished bandpass filtering')
print(filtered_Bscan.shape)



#* process background removal
background_removal = processing_background_removal(filtered_Bscan)
background_removed_Bscan = background_removal.subtract_background()
print('Finished background removal')
print(background_removed_Bscan.shape)


plot_data = [Raw_Bscan, aligned_Bscan, filtered_Bscan, background_removed_Bscan]


sample_interval_ns = sample_interval * 1e9
#* plot 4 figures
fig, ax = plt.subplots(len(plot_data), 1, figsize=(12, 20), tight_layout=True, sharex=True)
fontsize_large = 20
fontsize_medium = 18
fontsize_small = 16

for i in range(len(plot_data)):
    imshow = ax[i].imshow(plot_data[i], aspect='auto', cmap='seismic',
                extent=[0, plot_data[i].shape[1], plot_data[i].shape[0]*sample_interval_ns, 0],
                vmin=-15, vmax=15
                )
    ax[i].tick_params(axis='both', which='major', labelsize=fontsize_small)
    ax[i].set_title(['Raw Bscan', 'Time-zero correction', 'Bandpass filtering', 'Background removal'][i], fontsize=fontsize_large)

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