import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from natsort import natsorted
import argparse


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='plot_position.py',
    description='Plot position of observation site from ECHO data',
    epilog='End of help message',
    usage='python tools/plot_position.py [path_type] [function type]',
)
parser.add_argument('path_type', choices = ['local', 'SSD'], help='Choose the path type')
parser.add_argument('function_type', choices = ['load', 'plot', 'both'], help='Choose the function type')
args = parser.parse_args()


#* Define data folder path
if args.path_type == 'local':
    data_folder_path = 'LPR_2B/ECHO'
    position_folder_path = 'LPR_2B/Position'
elif args.path_type == 'SSD':
    data_folder_path = '/Volumes/SSD_kanda/LPR/LPR_2B/ECHO'
    position_folder_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Position'
output_dir = os.path.join(os.path.dirname(data_folder_path), 'Position')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



def load_positions():
    for ECHO_data in tqdm(natsorted(os.listdir(data_folder_path))):
        record_count = []
        VELOCITY = []
        XPOSITION = []
        YPOSITION = []
        ZPOSITION = []
        reference_X = []
        reference_Y = []
        reference_Z = []
        distance = []
        #* Load only .txt files
        if not ECHO_data.endswith('.txt'):
            continue
        if ECHO_data.startswith('._'):
            continue

        ECHO_data_path = os.path.join(data_folder_path, ECHO_data)
        data = np.loadtxt(ECHO_data_path, delimiter=' ')

        #* Load record_count, XPOSITION, YPOSITION, ZPOSITION
        for i in range(data.shape[1]):

            record_count.append(i + 1)
            VELOCITY.append(data[1, i])
            XPOSITION.append(data[2, i])
            YPOSITION.append(data[3, i])
            ZPOSITION.append(data[4, i])
            reference_X.append(data[5, i])
            reference_Y.append(data[6, i])
            reference_Z.append(data[7, i])

            distance.append(distance[-1] + np.sqrt((XPOSITION[-1] - XPOSITION[-2])**2 + (YPOSITION[-1] - YPOSITION[-2])**2) if distance else 0)

        #* Save record_count, XPOSITION, YPOSITION, ZPOSITION as 4xN array
        positions = np.array([record_count, VELOCITY, XPOSITION, YPOSITION, ZPOSITION, distance, reference_X, reference_Y, reference_Z])

        #* sort by record_count
        positions = positions[:, np.argsort(positions[0])]

        #* Save positions with header
        header = 'record_number Velocity X Y Z distance Refernce_X Reference_Y Reference_z'
        sequence_id = ECHO_data.split('_')[-1].split('.')[0]
        np.savetxt(os.path.join(output_dir, 'position_' + str(sequence_id) + '.txt'), positions.T, delimiter=' ', header=header, comments='')



def plot():
    positions4plot = np.array([])  # Initialize as numpy array
    total_distance4plot = np.array([])  # Initialize as numpy array
    total_x = np.array([])  # Initialize as numpy array
    total_y = np.array([])  # Initialize as numpy array
    record_num = []
    total_record_num = []
    sequence_id = []


    #* Load positions
    for positions_file in natsorted(os.listdir(position_folder_path)):
        if not positions_file.endswith('.txt'):
            continue
        if positions_file.startswith('._'):
            continue
        full_path = os.path.join(position_folder_path, positions_file)
        positions = np.loadtxt(full_path, delimiter=' ', skiprows=1) # postions file contains header

        sequence_id.append(positions_file.split('_')[-1].split('.')[0])
        record_num.append(positions.shape[0])
        total_record_num.append(total_record_num[-1] + positions.shape[0] if total_record_num else positions.shape[0])


        #* Load position data
        if positions4plot.size > 0:
            positions4plot = np.concatenate((positions4plot, positions))
        else:
            positions4plot = positions


        #* calculate total distance
        if total_distance4plot.size > 0:
            last_total_distance = total_distance4plot[-1]
            total_distance4plot = np.concatenate((total_distance4plot, last_total_distance + positions[:, 5]))
        else:
            total_distance4plot = positions[:, 5]

        #* calculate total position
        if total_x.size > 0:
            last_total_x = total_x[-1]
            last_total_y = total_y[-1]
            total_x = np.concatenate((total_x, last_total_x + positions[:, 2]))
            total_y = np.concatenate((total_y, last_total_y + positions[:, 3]))
        else:
            total_x = positions[:, 2]
            total_y = positions[:, 3]

    #* Extract data
    Velocity = positions4plot[:, 1]
    X = positions4plot[:, 2]
    Y = positions4plot[:, 3]
    Z = positions4plot[:, 4]
    Reference_X = positions4plot[:, 6]
    Reference_Y = positions4plot[:, 7]
    Reference_Z = positions4plot[:, 8]


    #* plot
    fontsize_large = 20
    fontsize_medium = 18


    fig, axes = plt.subplots(4, 1, figsize=(25, 20), tight_layout=True, sharex=True)

    #* plot velocity
    axes[0].scatter(np.arange(len(Velocity)), Velocity, color='black')
    axes[0].set_ylabel('Velocity [cm/s]', fontsize=fontsize_medium)
    axes[0].axhline(y=5.5, color='red', linestyle='--')
    axes[0].grid()

    #* plot X, Y, Z
    axes[1].plot(X, linestyle='-', label='X (North-South)')
    axes[1].plot(Y, linestyle='--', label='Y (East-West)')
    axes[1].plot(Z, linestyle='-.', label='Z [m]')
    axes[1].set_ylabel('Rover posi [m]', fontsize=fontsize_medium)
    axes[1].legend(loc='upper right')
    axes[1].grid()

    #* plot reference X, Y, Z
    axes[2].plot(np.abs(Reference_X), linestyle='-', label='Reference_X')
    axes[2].plot(np.abs(Reference_Y), linestyle='--', label='Reference_Y')
    axes[2].plot(np.abs(Reference_Z), linestyle='-.', label='Reference_Z')
    axes[2].set_ylabel('Reference point posi [m]', fontsize=fontsize_medium)
    axes[2].set_yscale('log')
    axes[2].legend(loc='upper right')
    axes[2].grid()

    #* plot distance
    axes[3].plot(total_distance4plot, color='black')
    axes[3].set_ylabel('Total distance [m]', fontsize=fontsize_medium)
    axes[3].grid()

    axes[3].set_xticks(total_record_num[::2], sequence_id[::2], rotation=90)

    #* set xticks
    fig.supxlabel('Record number', fontsize=fontsize_large)

    plt.savefig(os.path.join(position_folder_path, 'plot_position.png'))
    plt.show()


    #* Plot track of CE-4
    fig = plt.figure(figsize=(25, 20), tight_layout=True)
    plt.scatter(total_y, total_x, c=total_distance4plot, cmap='viridis', s=10)
    #* colorbar
    cbar = plt.colorbar(plt.scatter(total_y, total_x, marker='.', c=total_distance4plot, cmap='viridis', s=5),
                        location='bottom', orientation='horizontal', pad=0.1)
    cbar.set_label('Distance (m)', fontsize=fontsize_medium)
    #* plot start point
    plt.plot(total_y[0], total_x[0], marker='*', markersize=12, color='red')

    plt.grid()
    plt.xlabel('East-West', fontsize=fontsize_medium)
    plt.ylabel('North-South', fontsize=fontsize_medium)

    plt.savefig(os.path.join(position_folder_path, 'plot_track.png'))
    plt.show()









    """
    fig = plt.figure(figsize=(25, 20), tight_layout=True)
    #* 左列は4つのパネル，右列は2つのパネル，右列のパネルは縦並びで左列のパネルの高さの2倍
    gs = GridSpec(4, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1, 1])



    split_points = [0, int(len(sequence_id)/4), int(len(sequence_id)/2), int(3*len(sequence_id)/4), len(sequence_id)]

    #* Left side panels
    ax_left_0 = fig.add_subplot(gs[0, 0])
    ax_left_1 = fig.add_subplot(gs[1, 0])
    ax_left_2 = fig.add_subplot(gs[2, 0])
    ax_left_3 = fig.add_subplot(gs[3, 0])
    axes = [ax_left_0, ax_left_1, ax_left_2, ax_left_3]

    for i in range(len(axes)):
            #* plot X, Y, Z, distance
            axes[i].plot(positions4plot[:, 2], linestyle='-')
            axes[i].plot(positions4plot[:, 3], linestyle='--')
            axes[i].plot(positions4plot[:, 5], linestyle='-.')
            #* plot velocity
            axes2 = axes[i].twinx()
            axes2.plot(positions4plot[:, 1], color='black', alpha=0.2)
            axes2.set_ylim(0, 0.1)

            axes[i].grid()
            axes[i].set_xticks(total_record_num[:len(sequence_id)], sequence_id, rotation=90)
            axes[i].set_xlim(total_record_num[split_points[i]], total_record_num[split_points[i+1] - 1])
            axes[i].set_ylabel('[m]')
            axes2.set_ylabel('[cm/s]')
            if i == 0:
                axes[i].legend(['X (North-South) [m]', 'Y (East-West) [m]', 'Distance [m]'], loc = 'upper left')
                axes2.legend(['Velocity [cm/s]'], loc = 'upper right')


    #* Right side panels
    ax_right_0 = fig.add_subplot(gs[0:2, 1])
    ax_right_1 = fig.add_subplot(gs[2:4, 1])

    #* plot total distance
    ax_right_0.plot(total_distance4plot, '-')
    ax_right_0.set_xlabel('Sequence id', fontsize=fontsize_medium)
    ax_right_0.set_ylabel('Distance (m)', fontsize=fontsize_medium)
    ax_right_0.grid()
    #* sequence_idを2つ飛ばしでメモリにする
    ax_right_0.set_xticks(total_record_num[:len(sequence_id):2], sequence_id[::2], rotation=90)
    ax_right_0.set_xlim(0, positions4plot.shape[0])

    #* plot track of CE-4
    ax_right_1.scatter(total_y, total_x, c=total_distance4plot, cmap='viridis', s=10)
    #* colorbar
    cbar = plt.colorbar(ax_right_1.scatter(total_y, total_x, marker='.', c=total_distance4plot, cmap='viridis', s=5),
                        ax=ax_right_1, location='bottom', orientation='horizontal', pad=0.1)
    cbar.set_label('Distance (m)', fontsize=fontsize_medium)

    #* plot start point
    ax_right_1.plot(total_y[0], total_x[0], marker='*', markersize=12, color='red')
    ax_right_1.grid()
    ax_right_1.set_xlabel('East-West', fontsize=fontsize_medium)
    ax_right_1.set_ylabel('North-South', fontsize=fontsize_medium)
    """

    #* Convert X and Y ticks into Moon's coordinate as (0, 0) is (177.599100, -45.444600)
    """
    Moon_radius = 1737.4 * 10**3 # m
    deg_per_km = 360 / (2 * np.pi * Moon_radius)
    xticks = ax_right_1.get_xticks()
    yticks = ax_right_1.get_yticks()
    xticks_moon = xticks / 1000 * deg_per_km  + 177.599100
    yticks_moon = yticks / 1000 * deg_per_km - 45.444600
    ax_right_1.set_xticklabels(xticks_moon)
    ax_right_1.set_yticklabels(yticks_moon)


    plt.savefig(os.path.join(position_folder_path, 'plot_position.png'))
    plt.savefig(os.path.join(position_folder_path, 'plot_position.pdf'))
    plt.show()
    """
    return plt



if args.function_type == 'load':
    load_positions()
elif args.function_type == 'plot':
    plot()
elif args.function_type == 'both':
    load_positions()
    plot()
else:
    print('Invalid function type')
    exit()