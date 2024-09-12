import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
import argparse
from scipy import signal
from tqdm import tqdm
import gc


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='detect_peak.py',
    description='Detect echo peak from B-scan data and plot',
    epilog='End of help message',
    usage='python tools/detect_peak.py [data_path]',
)
parser.add_argument('data_path', help='Path to the txt file of data.')
args = parser.parse_args()



#* Define the data path
data_path = args.data_path
output_dir = os.path.join(os.path.dirname(data_path), 'detect_peak')
os.makedirs(output_dir, exist_ok=True)

output_dir_trim_png = os.path.join(output_dir, 'trim/png')
os.makedirs(output_dir_trim_png, exist_ok=True)
output_dir_trim_pdf = os.path.join(output_dir, 'trim/pdf')
os.makedirs(output_dir_trim_pdf, exist_ok=True)


#* Parameters
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]



#* Load data
print('Loading data...')
data = np.loadtxt(data_path, delimiter=' ')
print('Data shape:', data.shape)



#* Calculate the envelope
envelope = np.abs(signal.hilbert(data, axis=0))



#* Detect peak from the B-scan data
scatter_x_idx = []
scatter_time_idx = []
scatter_value = []

#data_trim = data[int(50e-9/sample_interval):, :]
#envelope_trim = envelope[int(50e-9/sample_interval):, :]
backgrounds = np.mean(np.abs(data[int(50e-9/sample_interval):, :]), axis=0)
thresholds = backgrounds * 3


#* Detect the peak
for i in tqdm(range(data.shape[1]), desc='Detecting peaks'):
    above_threshold_indices = np.where(envelope[:, i] > thresholds[i])[0]

    if len(above_threshold_indices) > 0:
        # Find the start and end of each group of indices above the threshold
        split_points = np.split(above_threshold_indices, np.where(np.diff(above_threshold_indices) != 1)[0] + 1)

        for group in split_points:
            start, end = group[0], group[-1] + 1
            peak_idx_in_group = np.argmax(np.abs(envelope[start:end, i])) + start

            scatter_x_idx.append(i*trace_interval) # [m]
            scatter_time_idx.append(peak_idx_in_group * sample_interval * 1e9) # [ns]
            scatter_value.append(data[peak_idx_in_group, i])
            #if data[peak_idx_in_group, i] > 0:
            #    scatter_value.append(1)
            #else:
            #    scatter_value.append(-1)


scatter_x_idx = np.array(scatter_x_idx)
scatter_time_idx = np.array(scatter_time_idx)
scatter_value = np.array(scatter_value)

scatters = np.vstack((scatter_x_idx, scatter_time_idx, scatter_value)).T
print('Scatter shape:', scatters.shape)


#* Calculate dB
#envelope[envelope == 0] = 1e-10
#envelope_db = 10 * np.log10(envelope/np.amax(envelope))



#* Define the function to plot
def plot(Bscan_data, scatter_data, x1, x2, y1, y2):
    fig = plt.figure(figsize=(18, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    im = ax.imshow(Bscan_data, aspect='auto', cmap='gray',
                    extent=[x1, x2, y2, y1],
                    vmin=-3000, vmax=3000
                    )
    scatter = ax.scatter(scatter_data[:, 0], scatter_data[:, 1], # +50 to compensate the trim
                        c=scatter_data[:, 2], cmap='bwr', s=1,
                        #vmin = -scatter_max/5, vmax = scatter_max/5
                        vmin = -3000, vmax = 3000)


    #* Set labels
    ax.set_xlabel('Distance [m]', fontsize=20)
    ax.set_ylabel('Time [ns]', fontsize=20)
    ax.tick_params(labelsize=16)

    ax.grid(which='both', axis='both', linestyle='-.', color='white')


    delvider = axgrid1.make_axes_locatable(ax)
    cax_im = delvider.append_axes('right', size='3%', pad=0.1)
    cbar_im = plt.colorbar(im, cax=cax_im, orientation = 'vertical')
    cbar_im.set_label('Amplitude', fontsize=18)
    cbar_im.ax.tick_params(labelsize=16)

    cax_scatter = delvider.append_axes('right', size='3%', pad=1.5)
    cbar_scatter = plt.colorbar(scatter, cax=cax_scatter, orientation = 'vertical')
    cbar_scatter.set_label('Peak amplitude', fontsize=18)
    cbar_scatter.ax.tick_params(labelsize=16)


    filename_base = f'/x{x1}_y{y1}'
    fig.savefig(output_dir_trim_png + filename_base + '.png', format='png', dpi=120)
    fig.savefig(output_dir_trim_pdf + filename_base + '.pdf', format='pdf', dpi=300)

    plt.close()
    gc.collect()  # Garbage collection to avoid memory error

    return plt




#* Plot
print(' ')
print('Plotting...')
x_first_list = np.arange(0, data.shape[1] * trace_interval, 100) # [m]
y_first_list = np.arange(0, data.shape[0] * sample_interval * 1e9, 100) # [ns]


for x_first in tqdm(x_first_list, desc='Plot trimmed B-scan and scatter'):
    x_last = min(data.shape[1] * trace_interval, x_first + 100) # [m]
    #scatter_trim = scatters[(scatters[:, 0] >= x_first) & (scatters[:, 0] < x_last)]
    for y_first in y_first_list:
        #* Trim the data
        y_last = min(data.shape[0] * sample_interval * 1e9, y_first + 100) # [ns]

        #* Trim the data
        data_trim = data[int(y_first/sample_interval/1e9):int(y_last/sample_interval/1e9),
                                    int(x_first/trace_interval):int(x_last/trace_interval)]
        scatter_trim = scatters[(scatters[:, 0] >= x_first) & (scatters[:, 0] < x_last)
                                    & (scatters[:, 1] >= y_first) & (scatters[:, 1] < y_last)]
        #scatter_trim = scatter_trim[(scatter_trim[:, 1] >= y_first) & (scatter_trim[:, 1] < y_last)]
        plot(data_trim, scatter_trim, x_first, x_last, y_first, y_last)


        """
        fig = plt.figure(figsize=(18, 12), tight_layout=True)
        ax = fig.add_subplot(111)

        im = ax.imshow(data_trim, aspect='auto', cmap='gray',
                        extent=[x_first, x_last, y_last, y_first],
                        vmin=-3000, vmax=3000
                        )
        scatter = ax.scatter(scatter_x_idx*trace_interval, scatter_time_idx*sample_interval*1e9 + 50, # +50 to compensate the trim
                            c=scatter_value, cmap='bwr', s=1,
                            #vmin = -scatter_max/5, vmax = scatter_max/5
                            vmin = -3000, vmax = 3000)

        ax.set_xlim(x_first, x_last)
        ax.set_ylim(y_last, y_first)
        #* Set labels
        ax.set_xlabel('Distance [m]', fontsize=20)
        ax.set_ylabel('Time [ns]', fontsize=20)
        ax.tick_params(labelsize=16)

        ax.grid(which='both', axis='both', linestyle='-.', color='gray')


        delvider = axgrid1.make_axes_locatable(ax)
        cax_im = delvider.append_axes('right', size='3%', pad=0.1)
        cbar_im = plt.colorbar(im, cax=cax_im, orientation = 'vertical')
        cbar_im.set_label('Envelope', fontsize=18)
        cbar_im.ax.tick_params(labelsize=16)

        cax_scatter = delvider.append_axes('right', size='3%', pad=1)
        cbar_scatter = plt.colorbar(scatter, cax=cax_scatter, orientation = 'vertical')
        cbar_scatter.set_label('Peak amplitude', fontsize=18)
        cbar_scatter.ax.tick_params(labelsize=16)


        filename_base = f'/x{x_first}_y{y_first}'
        fig.savefig(output_dir_trim_png + filename_base + '.png', format='png', dpi=120)
        fig.savefig(output_dir_trim_pdf + filename_base + '.pdf', format='pdf', dpi=300)

        plt.close()
        gc.collect()  # Garbage collection to avoid memory error
        """
