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
    usage='python tools/plot_Bscan.py [path_type]',
)
parser.add_argument('path_type', choices = ['local', 'SSD'], help='Choose the path type')
args = parser.parse_args()



#* Define data folder path
if args.path_type == 'local':
    ECHO_dir = 'LPR_2B/ECHO'
elif args.path_type == 'SSD':
    ECHO_dir = '/Volumes/SSD_kanda/LPR/LPR_2B/ECHO'
else:
    raise ValueError('Invalid path type')


"""
#* load data
Ascans = np.loadtxt(Ascans_file_path, skiprows=1, delimiter=' ')
print("Ascans shape:", Ascans.shape)
"""
sample_interval = 0.312500  # [ns]


#Velocity = []
#ECHO = []
output_dir = os.path.join(os.path.dirname(ECHO_dir), 'Raw_Bscan')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for ECHO_data in tqdm(natsorted(os.listdir(ECHO_dir))):
    #* Load only .txt files
    if not ECHO_data.endswith('.txt'):
        continue
    if ECHO_data.startswith('._'):
        continue

    sequcence_id = ECHO_data.split('_')[-1].split('.')[0]
    Velocity = []
    XPOSITION = []
    YPOSITION = []
    ZPOSITION = []
    ECHO = []

    ECHO_data_path = os.path.join(ECHO_dir, ECHO_data)
    data = np.loadtxt(ECHO_data_path, delimiter=' ')


    Velocity = data[1, :]
    XPOSITION = data[2, :]
    YPOSITION = data[3, :]
    ZPOSITION = data[4, :]
    ECHO = data[5:, :]

    #ECHO_for_plot = np.concatenate(ECHO, axis=1)
    #print("B-scan shape after concatenation:", ECHO_for_plot.shape)

    #np.savetxt(output_dir + '/Bscan.txt', ECHO_for_plot)


    #* plot
    font_lartge = 20
    font_medium = 18
    font_small = 16


    fig = plt.figure(figsize=(20, 10), tight_layout=True)
    gs = GridSpec(3, 1, height_ratios=[4, 1, 1])

    gs_echo = plt.subplot(gs[0])
    gs_echo.imshow(ECHO, aspect='auto', cmap='seismic',
                extent=[0, ECHO.shape[1], ECHO.shape[0]*sample_interval, 0],
                vmin=-50, vmax=50
                )
    #axes[0].set_xlabel('Trace number', fontsize=18)
    gs_echo.set_ylabel('Time [ns]', fontsize=font_lartge)
    gs_echo.tick_params(axis='both', which='major', labelsize=font_medium)


    #* plot colorbar
    """
    delvider = axgrid1.make_axes_locatable(axes[0])
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(cax=cax).set_label('Amplitude', fontsize=18)
    cax.tick_params(labelsize=16)
    """

    #* plot velocity
    gs_V = plt.subplot(gs[1], sharex=gs_echo)
    gs_V.plot(Velocity*100)
    gs_V.set_ylabel('Velocity \n [cm/s]', fontsize=font_lartge)
    gs_V.set_ylim(0, 7)
    gs_V.axhline(y=5.5, color='red', linestyle='--')
    gs_V.tick_params(axis='both', which='major', labelsize=font_medium)

    #* plot position
    gs_posi = plt.subplot(gs[2], sharex=gs_echo)
    gs_posi.plot(XPOSITION, '-', label='X position')
    gs_posi.plot(YPOSITION, '--', label='Y position')
    gs_posi.plot(ZPOSITION, '-.', label='Z position')
    gs_posi.set_ylabel('Position \n [m]', fontsize=font_lartge)
    gs_posi.legend(loc='lower right', fontsize=font_medium)
    gs_posi.tick_params(axis='both', which='major', labelsize=font_medium)


    fig.supxlabel('Record number', fontsize=font_lartge)
    fig.suptitle('Sequence ID: ' + str(sequcence_id), fontsize=font_lartge)


    #* save plot
    plt.savefig(output_dir + '/Bscan_' + str(sequcence_id + '.png'), dpi=300)
    plt.close()