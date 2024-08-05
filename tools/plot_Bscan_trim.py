import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
from tqdm import tqdm



#* Parse command line arguments




#* Define the data path
data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Processed_data/txt/4_gained_Bscan.txt'
output_dir = os.path.join(os.path.dirname(data_path), 'Trimmed_Plot_gained')
os.makedirs(output_dir, exist_ok=True)
output_dir_png = os.path.join(output_dir, 'png')
os.makedirs(output_dir_png, exist_ok=True)
output_dir_pdf = os.path.join(output_dir, 'pdf')
os.makedirs(output_dir_pdf, exist_ok=True)



#* Load data
print('Loading data...')
Bscan_data = np.loadtxt(data_path, delimiter=' ')
Bscan_data = Bscan_data / np.amax(Bscan_data)  # Normalize the data
print('Data shape:', Bscan_data.shape)
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]



#* Define the function to plot
def plot(data, x1, x2, y1, y2):
    plt.figure(figsize=(12, 8), tight_layout=True)
    plt.imshow(data, aspect='auto', cmap='gray',
                extent=[x1, x2, y2, y1],
                vmin=-0.1, vmax=0.1
                )
    plt.xlabel('Distance [m]', fontsize=20)
    plt.ylabel('Time [ns]', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.grid(which='both', axis='both', linestyle='-.')

    #* Colorbar
    delvider = axgrid1.make_axes_locatable(plt.gca())
    cax = delvider.append_axes('right', size='3%', pad=0.1)
    cbar = plt.colorbar(cax=cax, orientation = 'vertical')
    cbar.set_label('Normalized Amplitude', fontsize=20)
    cax.tick_params(labelsize=18)

    plt.savefig(os.path.join(output_dir_png, f'x{x1}_y{y1}.png'))
    plt.savefig(os.path.join(output_dir_pdf, f'x{x1}_y{y1}.pdf'), format='pdf', dpi=300)
    #plt.show()
    plt.close()



#* Trim the data
x_first_list = np.arange(0, Bscan_data.shape[1] * trace_interval, 100) # [m]
y_first_list = np.arange(0, Bscan_data.shape[0] * sample_interval * 1e9, 100) # [ns]

for x_first in tqdm(x_first_list):
    for y_first in y_first_list:
        x_last = min(Bscan_data.shape[1] * trace_interval, x_first + 100)
        y_last = min(Bscan_data.shape[0] * sample_interval * 1e9, y_first + 100)

        #* Trim the data
        x_first_idx = int(x_first / trace_interval)
        x_last_idx = int(x_last / trace_interval)
        y_first_idx = int(y_first / sample_interval / 1e9)
        y_last_idx = int(y_last / sample_interval / 1e9)
        trimmed_data = Bscan_data[y_first_idx:y_last_idx, x_first_idx:x_last_idx]

        plot(trimmed_data, x_first, x_last, y_first, y_last)