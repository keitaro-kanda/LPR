import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
import argparse
from scipy import signal
from tqdm import tqdm


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='plot_Ascan.py',
    description='Plot A-scan from B-scan data',
    epilog='End of help message',
    usage='python tools/fk_migration.py [data_path] [x] [t_first] [t_last] [-auto]',
)
parser.add_argument('data_path', help='Path to the txt file of data.')
parser.add_argument('x', type=float, help='x position [m] of the A-scan')
parser.add_argument('t_first', type=float, help='Start time of the A-scan [ns]')
parser.add_argument('t_last', type=float, help='Last time of the A-scan [ns]')
parser.add_argument('-auto', action='store_true', help='Select plot area automatically from setting list')
args = parser.parse_args()



#* Define the data path
data_path = args.data_path
output_dir = os.path.join(os.path.dirname(data_path), 'A_scan')
os.makedirs(output_dir, exist_ok=True)



#* Parameters
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]



#* Load data
print('Loading data...')
Bscan = np.loadtxt(data_path, delimiter=' ')
print('Data shape:', Bscan.shape)



#* Detect the peak
envelope = np.abs(signal.hilbert(Bscan, axis=0))

scatter_x_idx = []
scatter_time_idx = []
scatter_value = []

backgrounds = np.mean(np.abs(Bscan[int(50e-9/sample_interval):, :]), axis=0)
thresholds = backgrounds * 3

for i in tqdm(range(Bscan.shape[1]), desc='Detecting peaks'):
    above_threshold_indices = np.where(envelope[:, i] > thresholds[i])[0]

    if len(above_threshold_indices) > 0:
        # Find the start and end of each group of indices above the threshold
        split_points = np.split(above_threshold_indices, np.where(np.diff(above_threshold_indices) != 1)[0] + 1)

        for group in split_points:
            start, end = group[0], group[-1] + 1
            peak_idx_in_group = np.argmax(np.abs(envelope[start:end, i])) + start

            scatter_x_idx.append(i) # index
            scatter_time_idx.append(peak_idx_in_group) # index
            scatter_value.append(Bscan[peak_idx_in_group, i])

scatters = np.vstack((scatter_x_idx, scatter_time_idx, scatter_value)).T
print('Scatter shape:', scatters.shape)



#* Define function to extract A-scan data from B-scan data
def make_plot_data(Bscan_data, x, t_first, t_last):
    x_idx = int(x / trace_interval)
    t_first_idx = int(t_first * 1e-9 / sample_interval)
    t_last_idx = int(t_last * 1e-9 / sample_interval)
    Ascan = Bscan_data[t_first_idx:t_last_idx, x_idx]

    t_array = np.arange(t_first_idx, t_last_idx) * sample_interval / 1e-9

    #* Calculate the envelope
    envelope = np.abs(signal.hilbert(Ascan))

    #* Calculate the background
    background = np.mean(np.abs(Bscan[int(50e-9/sample_interval):, x_idx]))

    #* Detect the peak in the envelope
    threshold = background * 3
    peak_idx = []
    peak_value = []

    i = 0
    while i < len(envelope):
        if envelope[i] > threshold:
            start = i
            while i < len(envelope) and envelope[i] > threshold:
                i += 1
            end = i
            peak_idx.append(np.argmax(np.abs(Ascan[start:end])) + start)
            peak_value.append(Ascan[peak_idx[-1]])
        i += 1




    #* Closeup B-scan and peak data around the A-scan
    x_first_idx = x_idx - int(15/trace_interval) # index
    x_last_idx = x_idx + int(15/trace_interval) # index
    Bscan_trim = Bscan[t_first_idx:t_last_idx, x_first_idx:x_last_idx]
    scatter_trim = scatters[(scatters[:, 0] >= x_first_idx) & (scatters[:, 0] < x_last_idx)
                                    & (scatters[:, 1] >= t_first_idx) & (scatters[:, 1] < t_last_idx)]

    return Ascan, envelope, background, t_array, peak_idx, peak_value, Bscan_trim, scatter_trim, x_first_idx, x_last_idx



#* Plot
def plot(Ascan_data, t_array, envelope_data, background_value, t_first, t_last, peak_idx, peak_value, Bscan_data, scatter_data, x, x_first_idx, x_last_idx):
    fig, ax = plt.subplots(1, 2, figsize=(18, 8), sharey=True, tight_layout=True)

    #* Plot A-scan
    ax[0].plot(Ascan_data, t_array, color='black', label='Signal')
    ax[0].plot(envelope_data, t_array, color='blue', linestyle='-.', label='Envelope')
    ax[0].vlines(background_value, t_first, t_last, color='gray', linestyle='--', label='Background')
    ax[0].vlines(-background_value, t_first, t_last, color='gray', linestyle='--')
    ax[0].scatter(peak_value, t_array[peak_idx], color='r')

    ax[0].set_xlabel('Amplitude', fontsize=20)
    ax[0].grid(True)
    ax[0].tick_params(labelsize=18)
    ax[0].set_xlim(-np.amax(np.abs(Ascan_data))*1.2, np.amax(np.abs(Ascan_data))*1.2)
    ax[0].legend(fontsize=16)



    #* Plot B-scan and detected peaks
    im = ax[1].imshow(Bscan_data,
                    aspect='auto', cmap='gray',
                    vmin=-5000, vmax=5000,
                    extent=[x_first_idx*trace_interval, x_last_idx*trace_interval, t_last, t_first])
    scatter = ax[1].scatter(scatter_data[:, 0]*trace_interval, scatter_data[:, 1]*sample_interval*1e9,
                        c=scatter_data[:, 2], cmap='bwr', s=5, alpha=0.5,
                        vmin = -3000, vmax = 3000)
    ax[1].set_xlabel('Distance [m]', fontsize=20)
    ax[1].tick_params(labelsize=18)

    ax[1].vlines(x, t_first, t_last, color='k', linestyle='-.')

    ax[1].grid(which='both', axis='both', linestyle='-.', color='white')


    delvider = axgrid1.make_axes_locatable(ax[1])
    cax_im = delvider.append_axes('right', size='3%', pad=0.1)
    cbar_im = plt.colorbar(im, cax=cax_im, orientation = 'vertical')
    cbar_im.set_label('Amplitude', fontsize=18)
    cbar_im.ax.tick_params(labelsize=16)

    cax_scatter = delvider.append_axes('right', size='3%', pad=1.5)
    cbar_scatter = plt.colorbar(scatter, cax=cax_scatter, orientation = 'vertical')
    cbar_scatter.set_label('Peak amplitude', fontsize=18)
    cbar_scatter.ax.tick_params(labelsize=16)

    fig.supylabel('Time [ns]', fontsize=20)


    #* Save the plot
    plt.savefig(os.path.join(output_dir_fig, f'{x}_t{t_first}_{t_last}.png'), dpi=120)
    plt.savefig(os.path.join(output_dir_fig, f'{x}_t{t_first}_{t_last}.pdf'), dpi=300)

    if args.auto:
        plt.close()
    else:
        plt.show()


if args.auto:
    plot_list = np.loadtxt(os.path.join(output_dir, 'plot_list.txt'), delimiter=' ')
    for i in tqdm(range(plot_list.shape[0]), desc='Plot A-scan'):
        #* Set the output directory
        output_dir_fig = os.path.join(output_dir, f'x={int(plot_list[i, 0]/100)*100}_{int(plot_list[i, 0]/100)*100+100}')
        os.makedirs(output_dir_fig, exist_ok=True)

        Ascan, envelope, background, t_array, peak_idx, peak_value, Bscan_trim, scatter_trim, x_first_idx, x_last_idx = make_plot_data(Bscan, plot_list[i, 0], plot_list[i, 1], plot_list[i, 2])
        plot(Ascan, t_array, envelope, background, plot_list[i, 1], plot_list[i, 2], peak_idx, peak_value, Bscan_trim, scatter_trim, plot_list[i, 0], x_first_idx, x_last_idx)

else:
    #* Make directory to save the plot
    output_dir_fig = os.path.join(output_dir, f'x={int(args.x/100)*100}_{int(args.x/100)*100+100}')
    os.makedirs(output_dir_fig, exist_ok=True)


    #* Save x, t_first, t_last to txt file
    plot_params = [args.x, args.t_first, args.t_last]
    plot_params = np.array(plot_params).reshape(1, 3)


    #* Make plot
    Ascan, envelope, background, t_array, peak_idx, peak_value, Bscan_trim, scatter_trim, x_first_idx, x_last_idx = make_plot_data(Bscan, args.x, args.t_first, args.t_last)
    plot(Ascan, t_array, envelope, background, args.t_first, args.t_last,peak_idx, peak_value,  Bscan_trim, scatter_trim, args.x, x_first_idx, x_last_idx)


    #* Save the plot parameters
    if not os.path.exists(os.path.join(output_dir, 'plot_list.txt')):
        np.savetxt(os.path.join(output_dir, 'plot_list.txt'), plot_params)
        print('Plot list saved at', os.path.join(output_dir, 'plot_list.txt'))
    else:
        plot_list = np.loadtxt(os.path.join(output_dir, 'plot_list.txt'))
        #* もし同じパラメータがあればスキップ，なければ追加して保存
        if not np.any(np.all(plot_list == plot_params, axis=1)):
            plot_list = np.vstack([plot_list, plot_params])
            plot_list = plot_list[plot_list[:, 0].argsort()]
            np.savetxt(os.path.join(output_dir, 'plot_list.txt'), plot_list, fmt='%.1f', delimiter=' ')
            print('Plot list saved at', os.path.join(output_dir, 'plot_list.txt'))
        else:
            print('The plot list already has the same parameters. Skip saving.')