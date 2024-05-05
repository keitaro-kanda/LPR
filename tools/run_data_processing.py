import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import mpl_toolkits.axes_grid1 as axgrid1
from bandpass import processing_filtering
from remove_background import processing_background_removal


#* load raw Bscan data
Bscan_data = np.loadtxt('Ascans/LPR_2B_echo_data.txt', skiprows=1, delimiter=' ')


#* process bandpass filter
bandpass_filtering = processing_filtering(Bscan_data)
sample_interval = bandpass_filtering.sample_interval
filtered_Bscan = bandpass_filtering.apply_bandpass_filter()
print(sample_interval)
print(filtered_Bscan.shape)
print(type(filtered_Bscan))


#* process background removal
background_removal = processing_background_removal(filtered_Bscan)
background_removed_Bscan = background_removal.subtract_background()
print(background_removed_Bscan.shape)



sample_interval_ns = sample_interval * 1e9
#* plot 3 figures
fig, ax = plt.subplots(3, 1, figsize=(15, 17), tight_layout=True, sharex=True)
fontsize_large = 20
fontsize_medium = 18
fontsize_small = 16

plot_data = [Bscan_data, filtered_Bscan, background_removed_Bscan]
for i in range(3):
    imshow = ax[i].imshow(plot_data[i], aspect='auto', cmap='seismic',
                extent=[0, plot_data[i].shape[1], plot_data[i].shape[0]*sample_interval_ns, 0],
                vmin=-100, vmax=100
                )
    ax[i].tick_params(axis='both', which='major', labelsize=fontsize_small)
    ax[i].set_title(['Raw Bscan', 'Bandpass filtering', 'Background removal'][i], fontsize=fontsize_large)

    if i == 2:
        ax[i].set_xlabel('Trace number', fontsize=fontsize_medium)


#fig.supxlabel('Trace number', fontsize=fontsize_medium)
fig.supylabel('Time (ns)', fontsize=fontsize_medium)

#* plot colorbar
delvider = axgrid1.make_axes_locatable(ax[2])
cax = delvider.append_axes('bottom', size='5%', pad=1)
plt.colorbar(imshow, cax=cax, orientation = 'horizontal').set_label('Amplitude', fontsize=fontsize_medium)
cax.tick_params(labelsize=fontsize_small)

# save plot
output_dir = os.path.join('Bscan_plots' + '/' + 'LPR_2B')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plot_name = 'LPR_2B' + '_processed.png'
plt.savefig(output_dir + '/' + plot_name)
plt.show()