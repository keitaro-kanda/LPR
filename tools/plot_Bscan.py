import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm


Ascans_file_path = 'LPR_2B/ECHO/ECHO_0315.txt'
#ECHO_dir = 'LPR_2B/ECHO'
ECHO_dir = '/Volumes/SSD_kanda/LPR/LPR_2B/ECHO'
print('Data dir: ', ECHO_dir)


"""
#* load data
Ascans = np.loadtxt(Ascans_file_path, skiprows=1, delimiter=' ')
print("Ascans shape:", Ascans.shape)
"""
sample_interval = 0.312500  # [ns]



ECHO_for_plot = []
for ECHO_data in tqdm(os.listdir(ECHO_dir)):
    #* Load only .txt files
    if not ECHO_data.endswith('.txt'):
        continue
    if ECHO_data.startswith('._'):
        continue

    ECHO_data_path = os.path.join(ECHO_dir, ECHO_data)
    data = np.loadtxt(ECHO_data_path, skiprows=1, delimiter=' ')
    ECHO_for_plot.append(data)

ECHO_for_plot = np.concatenate(ECHO_for_plot, axis=1)
print("B-scan shape after concatenation:", ECHO_for_plot.shape)


output_dir = os.path.join(os.path.dirname(ECHO_dir), 'Raw_Bscan')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
np.savetxt(output_dir + '/Bscan.txt', ECHO_for_plot)


#* plot
fig = plt.figure(figsize=(20, 7), tight_layout=True)
ax = fig.add_subplot(111)
plt.imshow(ECHO_for_plot, aspect='auto', cmap='seismic',
            extent=[0, ECHO_for_plot.shape[1], ECHO_for_plot.shape[0]*sample_interval, 0],
            vmin=-50, vmax=50
            )
ax.set_xlabel('Trace number', fontsize=18)
ax.set_ylabel('Time (ns)', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)

plt.title(ECHO_dir, fontsize=20)


#* plot colorbar
delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax).set_label('Amplitude', fontsize=18)
cax.tick_params(labelsize=16)

#* save plot
plt.savefig(output_dir + '/Bscan.png', dpi=300)
plt.show()