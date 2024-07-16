import numpy as np
import matplotlib.pyplot as plt
import cv2
import mpl_toolkits.axes_grid1 as axgrid1
import os
from matplotlib.colors import LogNorm

data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Processed_Bscan/txt/gained_Bscan.txt'
output_dir = '/Volumes/SSD_kanda/LPR/LPR_2B/test/sobel'
os.makedirs(output_dir, exist_ok=True)

Bscan_data = np.loadtxt(data_path, delimiter=' ')
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]


skip_sec = 0
skip = int(skip_sec/sample_interval)
Bscan_data = Bscan_data[skip:, :]
print('Shape of B-scan after skipping', Bscan_data.shape)


#* Compute the derivative of the data in x direction with Sobel with a kernel size of 5
#* This should eliminate the horizonatal lines
sobelx = cv2.Sobel(Bscan_data, cv2.CV_64F, 1, 0, ksize=3)
sobelx = np.abs(sobelx)
sobely = cv2.Sobel(Bscan_data, cv2.CV_64F, 0, 1, ksize=3)
sobely = np.abs(sobely)


sobelx[sobelx == 0] = 1e-15
sobely[sobely == 0] = 1e-15

sobel_combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)


data_list = [sobelx, sobely, sobel_combined]
title_list = ['Sobel X', 'Sobel Y', 'Sobel Combined']



#* Plot
fig, ax = plt.subplots(3, 1, figsize=(18, 18), tight_layout=True, sharex=True)
font_large = 20
font_medium = 18
font_small = 16

for i, data in enumerate(data_list):
    """
    if np.amin(data) == 0:
        log_norm = LogNorm(vmin=0.1, vmax=np.amax(data)/1e5)
    else:
        log_norm = LogNorm(vmin=1e-2, vmax=np.amax(data)/1e5)
    """
    log_norm = LogNorm(vmin=np.amin(data), vmax=np.amax(data))
    data = 10 * np.log10(data/np.amax(data))
    im = ax[i].imshow(data,
                #norm=log_norm,
                aspect='auto', cmap='viridis',
                extent=[0, data.shape[1]*trace_interval, data.shape[0]*sample_interval*1e9, skip_sec*1e9],
                vmin=-30, vmax=0
                )
    ax[i].set_title(title_list[i], fontsize=font_large)
    ax[i].tick_params(axis='both', which='major', labelsize=font_small)


    #* plot colorbar in log scale
    delvider = axgrid1.make_axes_locatable(ax[i])
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(im, cax=cax).set_label('Amplitude [dB]', fontsize=font_medium)
    cax.tick_params(labelsize=font_small)

fig.supxlabel('Distance', fontsize=font_medium)
fig.supylabel('Time [ns]', fontsize=font_medium)

plt.savefig(output_dir + '/sobel.png')
plt.show()