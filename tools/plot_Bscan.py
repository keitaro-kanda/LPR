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
from scipy import signal


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='plot_Bscan.py',
    description='Plot B-scan from ECHO data',
    epilog='End of help message',
    usage='python tools/plot_Bscan.py [path_type] [function type] [-envelope]',
)
parser.add_argument('path_type', choices = ['local', 'SSD', 'other'], help='Choose the path type')
parser.add_argument('function_type', choices = ['load', 'plot'], help='Choose the function type')
parser.add_argument('-envelope', action='store_true', help='Plot B-scan with envelope')
args = parser.parse_args()



#* Define data folder path
if args.path_type == 'local':
    ECHO_dir = 'LPR_2B/Resampled_ECHO/txt'
elif args.path_type == 'SSD':
    ECHO_dir = '/Volumes/SSD_kanda/LPR/LPR_2B/Resampled_ECHO/txt'
elif args.path_type == 'other': # Set pass manually
    data_path= '/Volumes/SSD_kanda/LPR/LPR_2B/Processed_Bscan/txt/2_aligned_Bscan.txt'
else:
    raise ValueError('Invalid path type')


"""
#* load data
Ascans = np.loadtxt(Ascans_file_path, skiprows=1, delimiter=' ')
print("Ascans shape:", Ascans.shape)
"""
sample_interval = 0.312500  # [ns]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]


#* Define output folder path
if args.path_type == 'other':
    output_dir = os.path.join(os.path.dirname(os.path.dirname(data_path)), 'test')
elif args.path_type == 'local' or 'SSD':
    output_dir = os.path.join(os.path.dirname(ECHO_dir))
else:
    raise ValueError('Invalid path type')
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
        data = np.loadtxt(ECHO_data_path, delimiter=' ', skiprows=0)


        if ECHO_for_plot.size == 0:
            ECHO_for_plot = data
        else:
            ECHO_for_plot = np.concatenate([ECHO_for_plot, data], axis=1)

    np.savetxt(output_dir + '/Bscan.txt', ECHO_for_plot)
    print("B-scan saved at", output_dir + '/Bscan.txt')
    return ECHO_for_plot


#* Calculate envelove
def envelope(data):
    #* Calculate the envelope of the data
    envelope = np.abs(signal.hilbert(data, axis=0))
    return envelope


#* plot
font_large = 20
font_medium = 18
font_small = 16

def single_plot(plot_data):
    plt.figure(figsize=(18, 6), tight_layout=True)
    if args.envelope:
        plt.imshow(plot_data, aspect='auto', cmap='jet',
                extent=[0, plot_data.shape[1]*trace_interval, plot_data.shape[0]*sample_interval, 0],
                vmin=0, vmax=50
                )
    else:
        plt.imshow(plot_data, aspect='auto', cmap='seismic',
                    extent=[0, plot_data.shape[1]*trace_interval, plot_data.shape[0]*sample_interval, 0],
                    vmin=-30, vmax=30
                    )
    plt.xlabel('Distance [m]', fontsize=font_large)
    plt.ylabel('Time [ns]', fontsize=font_large)
    plt.tick_params(axis='both', which='major', labelsize=font_medium)

    #* Colorbar
    delvider = axgrid1.make_axes_locatable(plt.gca())
    cax = delvider.append_axes('right', size='3%', pad=0.1)
    plt.colorbar(cax=cax, orientation = 'vertical').set_label('Amplitude', fontsize=font_large)
    cax.tick_params(labelsize=font_small)

    if args.envelope:
        plt.savefig(output_dir + '/Bscan_envelope.png')
        plt.savefig(output_dir + '/Bscan_envelope.pdf', format='pdf', dpi=300)
    else:
        plt.savefig(output_dir + '/Bscan.png')
        plt.savefig(output_dir + '/Bscan.pdf', format='pdf', dpi=300)
    plt.show()

    return plt



#* Main
if args.path_type == 'other':
    resampled_data = np.loadtxt(data_path)
    print("B-scan shape:", resampled_data.shape)
    if args.envelope:
        print('Calculating envelope...')
        resampled_data = envelope(resampled_data)
    single_plot(resampled_data)
else:
    if args.function_type == 'load':
        resampled_data = load_resampled_data()
        print("B-scan shape:", resampled_data.shape)
        if args.envelope:
            print('Calculating envelope...')
            resampled_data = envelope(resampled_data)
        single_plot(resampled_data)
    elif args.function_type == 'plot':
        print('Function type is plot')
        print(' ')

        resampled_data = np.loadtxt(os.path.dirname(ECHO_dir) + '/Bscan.txt')
        print('Finished loading B-scan data')
        print("B-scan shape:", resampled_data.shape)
        print('')

        if args.envelope:
            print('Calculating envelope...')
            resampled_data = envelope(resampled_data)
        print('Now plotting...')

        single_plot(resampled_data)
    else:
        raise ValueError('Invalid function type')