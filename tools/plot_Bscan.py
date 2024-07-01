"""
This code make hall B-scan plot from resampled ECHO data.
If you want to make B-scan plot of each sequence, you can use resampling.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
import argparse
from natsort import natsorted


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='plot_Bscan.py',
    description='Plot B-scan from ECHO data',
    epilog='End of help message',
    usage='python tools/plot_Bscan.py [path_type] [function type]',
)
parser.add_argument('path_type', choices = ['local', 'SSD'], help='Choose the path type')
parser.add_argument('function_type', choices = ['load', 'plot'], help='Choose the function type')
args = parser.parse_args()



#* Define data folder path
if args.path_type == 'local':
    ECHO_dir = 'LPR_2B/Resampled_ECHO/txt'
elif args.path_type == 'SSD':
    ECHO_dir = '/Volumes/SSD_kanda/LPR/LPR_2B/Resampled_ECHO/txt'
else:
    raise ValueError('Invalid path type')


"""
#* load data
Ascans = np.loadtxt(Ascans_file_path, skiprows=1, delimiter=' ')
print("Ascans shape:", Ascans.shape)
"""
sample_interval = 0.312500  # [ns]


#* Define output folder path
output_dir = os.path.join(os.path.dirname(ECHO_dir))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def load_resampled_data():
    ECHO_for_plot = np.array([])
    for ECHO_data in tqdm(natsorted(os.listdir(ECHO_dir))):
        #* Load only .txt files
        if not ECHO_data.endswith('.txt'):
            continue
        if ECHO_data.startswith('._'):
            continue

        sequcence_id = ECHO_data.split('_')[-1].split('.')[0]

        ECHO_data_path = os.path.join(ECHO_dir, ECHO_data)
        data = np.loadtxt(ECHO_data_path, delimiter=' ', skiprows=1)

        if ECHO_for_plot.size == 0:
            ECHO_for_plot = data
        else:
            ECHO_for_plot = np.concatenate([ECHO_for_plot, data], axis=1)

    np.savetxt(output_dir + '/Bscan.txt', ECHO_for_plot)
    print("B-scan saved at", output_dir + '/Bscan.txt')
    print("B-scan shape:", ECHO_for_plot.shape)
    return ECHO_for_plot


#* plot
font_lartge = 20
font_medium = 18
font_small = 16

def single_plot(plot_data):
    plt.figure(figsize=(12, 6), tight_layout=True)
    plt.imshow(plot_data, aspect='auto', cmap='seismic',
                #extent=[0, plot_data.shape[1]*3.75*1e-2, plot_data.shape[0]*sample_interval, 0],
                extent=[0, plot_data.shape[1], plot_data.shape[0]*sample_interval, 0],
                vmin=-15, vmax=15
                )
    plt.xlabel('Record Count', fontsize=font_lartge)
    plt.ylabel('Time [ns]', fontsize=font_lartge)
    plt.colorbar().set_label('Amplitude', fontsize=font_lartge)

    plt.savefig(output_dir + '/Bscan.png')
    plt.show()

    return plt

if args.function_type == 'load':
    resampled_data = load_resampled_data()
    single_plot(resampled_data)
elif args.function_type == 'plot':
    resampled_data = np.loadtxt(os.path.dirname(ECHO_dir) + '/Bscan.txt')
    single_plot(resampled_data)
else:
    raise ValueError('Invalid function type')