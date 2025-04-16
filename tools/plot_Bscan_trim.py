import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
from tqdm import tqdm
import argparse
import gc



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='plot_Bscan.py',
    description='Plot trimmed B-scan',
    epilog='End of help message',
    usage='python tools/plot_Bscan.py [data_path] [data_type]',
)
parser.add_argument('data_path', help='Path to the txt file of thedata')
parser.add_argument('data_type', choices=['Bscan', 'pulse_compression', 'fk_migration'], help='Type of the data')
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
print('Data shape:', data.shape)



#* Set parameters
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]



#* Plot parameters
class PlotParams:
    def __init__(self, data_type, data_array):
        self.data_type = data_type
        self.data_array = data_array
        self._params = self._compute_params()

    def _compute_params(self):
        if self.data_type == 'Bscan':
            cmap = 'viridis'
            vmin = -np.amax(np.abs(self.data_array)) / 10
            vmax = np.amax(np.abs(self.data_array)) / 10
            xlabel = 'Distance [m]'
            ylabel = 'Time [ns]'
            cbar_label = 'Amplitude'
        elif self.data_type == 'pulse_compression':
            cmap = 'viridis'
            cbar_max = np.amax(np.abs(self.data_array)) / 5
            vmin = -cbar_max
            vmax = cbar_max
            xlabel = 'Distance [m]'
            ylabel = 'Time [ns]'
            cbar_label = 'Amplitude'
        elif self.data_type == 'fk_migration':
            cmap = 'viridis'
            vmin = 0
            vmax = np.amax(np.abs(self.data_array)) / 5
            xlabel = 'Distance [m]'
            ylabel = 'z [m]'
            cbar_label = 'Amplitude [dB]'
        else:
            raise ValueError('Invalid data type')

        return cmap, vmin, vmax, xlabel, ylabel, cbar_label

    def get_params(self):
        return self._params


#* Define the function to plot
def plot(data_array, x1, x2, y1, y2, data_type):
    #* Get plot parameters
    plot_params = PlotParams(data_type, data) # Input 'data' into in2, not 'data_array'
    cmap, vmin, vmax, xlabel, ylabel, cbar_label = plot_params.get_params()

    #* Plot
    plt.figure(figsize=(18, 8), tight_layout=True)
    plt.imshow(data_array, aspect='auto', cmap=cmap,
                extent=[x1, x2, y2, y1],
                vmin=vmin, vmax=vmax)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(labelsize=18)
    plt.grid(which='both', axis='both', linestyle='-.')

    #* Colorbar
    delvider = axgrid1.make_axes_locatable(plt.gca())
    cax = delvider.append_axes('right', size='3%', pad=0.1)
    cbar = plt.colorbar(cax=cax, orientation='vertical')
    cbar.set_label(cbar_label, fontsize=20)
    cax.tick_params(labelsize=18)

    #* Save the plot
    plt.savefig(os.path.join(output_dir_png, f'x{x1}_y{y1}.png'), format='png', dpi=120)
    plt.savefig(os.path.join(output_dir_pdf, f'x{x1}_y{y1}.pdf'), format='pdf', dpi=600)
    plt.close()
    gc.collect()  # Garbage collection to avoid memory error



#* Trim the data

#* Define the first point of the trimmed data
if args.data_type == 'fk_migration':
    v = 299792458 / np.sqrt(3.4)  # [m/s]
    x_first_list = np.arange(0, data.shape[1] * trace_interval, 100) # [m]
    y_first_list = np.arange(0, data.shape[0] * sample_interval * v, 100e-9 * v / 2) # [m]
else:
    x_first_list = np.arange(0, data.shape[1] * trace_interval, 100) # [m]
    y_first_list = np.arange(0, data.shape[0] * sample_interval * 1e9, 100) # [ns]


for x_first in tqdm(x_first_list):
    for y_first in y_first_list:
        #* Define the last point of the trimmed data
        if args.data_type == 'fk_migration':
            x_last = min(data.shape[1] * trace_interval, x_first + 100)
            y_last = min(data.shape[0] * sample_interval * v, y_first + 100e-9 * v /2)
        else:
            x_last = min(data.shape[1] * trace_interval, x_first + 100) # [m]
            y_last = min(data.shape[0] * sample_interval * 1e9, y_first + 100) # [ns]

        #* Define trimming area in index
        x_first_idx = int(x_first / trace_interval) # [index]
        x_last_idx = int(x_last / trace_interval) # [index]
        if args.data_type == 'fk_migration':
            y_first_idx = int(y_first / (sample_interval * v)) # [index]
            y_last_idx = int(y_last / (sample_interval * v)) # [index]
        else:
            y_first_idx = int(y_first / sample_interval / 1e9) # [index]
            y_last_idx = int(y_last / sample_interval / 1e9) # [index]

        #* Trim the data
        trimmed_data = data[y_first_idx:y_last_idx, x_first_idx:x_last_idx]

        #* Plot the trimmed data
        plot(trimmed_data, x_first, x_last, y_first, y_last, args.data_type)