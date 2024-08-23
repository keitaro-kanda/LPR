import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
from tqdm import tqdm
import argparse



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='plot_Bscan.py',
    description='Plot trimmed B-scan',
    epilog='End of help message',
    usage='python tools/plot_Bscan.py [data_path]]',
)
parser.add_argument('data_path', help='Path to the txt file of thedata')
args = parser.parse_args()



#* Define the data path
data_path = args.data_path
output_dir = os.path.join(os.path.dirname(data_path), 'Trimmed_plot')
os.makedirs(output_dir, exist_ok=True)
output_dir_png = os.path.join(output_dir, 'png')
os.makedirs(output_dir_png, exist_ok=True)
output_dir_pdf = os.path.join(output_dir, 'pdf')
os.makedirs(output_dir_pdf, exist_ok=True)



#* Load data
print('Loading data...')
data = np.loadtxt(data_path, delimiter=' ')
cbar_max = np.amax(np.abs(data)) / 5
cbar_min = -cbar_max
#data = 10 * np.log10(data/np.amax(np.abs(data)))  # Normalize the data
print('Data shape:', data.shape)
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]



#* Define the function to plot
def plot(data_array, x1, x2, y1, y2):
    plt.figure(figsize=(12, 8), tight_layout=True)
    plt.imshow(data_array, aspect='auto', cmap='seismic',
                extent=[x1, x2, y2, y1],
                vmin=cbar_min, vmax=cbar_max
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

    plt.savefig(os.path.join(output_dir_png, f'x{x1}_y{y1}.png'), format='png', dpi=120)
    plt.savefig(os.path.join(output_dir_pdf, f'x{x1}_y{y1}.pdf'), format='pdf', dpi=600)
    plt.close()



#* Trim the data
x_first_list = np.arange(0, data.shape[1] * trace_interval, 100) # [m]
y_first_list = np.arange(0, data.shape[0] * sample_interval * 1e9, 100) # [ns]

for x_first in tqdm(x_first_list):
    for y_first in y_first_list:
        x_last = min(data.shape[1] * trace_interval, x_first + 100)
        y_last = min(data.shape[0] * sample_interval * 1e9, y_first + 100)

        #* Trim the data
        x_first_idx = int(x_first / trace_interval)
        x_last_idx = int(x_last / trace_interval)
        y_first_idx = int(y_first / sample_interval / 1e9)
        y_last_idx = int(y_last / sample_interval / 1e9)
        trimmed_data = data[y_first_idx:y_last_idx, x_first_idx:x_last_idx]

        plot(trimmed_data, x_first, x_last, y_first, y_last)