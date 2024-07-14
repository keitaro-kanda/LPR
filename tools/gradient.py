import numpy as np
import matplotlib.pyplot as plt
import cv2
import mpl_toolkits.axes_grid1 as axgrid1
from matplotlib.colors import LogNorm
from tqdm import tqdm
import os


data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Processed_Bscan/txt/gained_Bscan.txt'
output_dir = '/Volumes/SSD_kanda/LPR/LPR_2B/test/gradient'
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

def grad_diviation():
    # Calculate the gradient of the data
    # calc f(x)/f(x-1) for x direction
    gradx = np.zeros_like(Bscan_data[1:, 1:])
    grady = np.zeros_like(Bscan_data[1:, 1:])

    #* Gradient for X
    for i in tqdm(range(gradx.shape[1]), desc='Gradient X'):
        for j in range(gradx.shape[0]):
            if Bscan_data[j, i] == 0:
                gradx[j, i] = 0
            else:
                gradx[j, i] = np.abs(Bscan_data[j+1, i] / Bscan_data[j, i])

    #* Gradient for Y
    for i in tqdm(range(grady.shape[0]), desc='Gradient Y'):
        for j in range(grady.shape[1]):
            if Bscan_data[i, j] == 0:
                grady[i, j] = 0
            else:
                grady[i, j] = np.abs(Bscan_data[i, j+1] / Bscan_data[i, j])

    return gradx, grady




def grad_traditional():
    gradx = np.abs(np.gradient(Bscan_data, axis=1))
    grady = np.abs(np.gradient(Bscan_data, axis=0))

    return gradx, grady


#* Calculate gradient
np_grad = True
if np_grad:
    gradx, grady = grad_traditional()
else:
    gradx, grady = grad_diviation()

gradx[gradx == 0] = 1e-15
grady[grady == 0] = 1e-15



grad_combined = np.sqrt(gradx**2 + grady**2)


data_list = [gradx, grady, grad_combined]
title_list = ['Gradient X', 'Gradient Y', 'Gradient Combined']

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
                vmin=-35, vmax=0
                )
    ax[i].set_title(title_list[i], fontsize=font_large)
    ax[i].tick_params(axis='both', which='major', labelsize=font_small)


    #* plot colorbar in log scale
    delvider = axgrid1.make_axes_locatable(ax[i])
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(im, cax=cax).set_label('Amplitude [dB]', fontsize=font_medium)
    cax.tick_params(labelsize=font_small)

fig.supxlabel('Trace number', fontsize=font_medium)
fig.supylabel('Time [ns]', fontsize=font_medium)

if np_grad:
    plt.savefig(output_dir + '/gradient_traditional.png')
else:
    plt.savefig(output_dir + '/gradient_diviation.png')
plt.show()