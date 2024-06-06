import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from natsort import natsorted


#data_folder_path = 'LPR_2B/original_data'
data_folder_path = '/Volumes/SSD_kanda/LPR/LPR_2B/ECHO'
position_folder_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Position'
output_dir = os.path.join(os.path.dirname(data_folder_path), 'Position')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



def load_positions():
    for ECHO_data in tqdm(natsorted(os.listdir(data_folder_path))):
        record_count = []
        XPOSITION = []
        YPOSITION = []
        ZPOSITION = []
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
            XPOSITION.append(data[2, i]) # data[1, i] is VELOCITY
            YPOSITION.append(data[3, i])
            ZPOSITION.append(data[4, i])

            distance.append(distance[-1] + np.sqrt((XPOSITION[-1] - XPOSITION[-2])**2 + (YPOSITION[-1] - YPOSITION[-2])**2) if distance else 0)

        #* Save record_count, XPOSITION, YPOSITION, ZPOSITION as 4xN array
        positions = np.array([record_count, XPOSITION, YPOSITION, ZPOSITION, distance])

        #* sort by record_count
        positions = positions[:, np.argsort(positions[0])]

        #* Save positions with header
        header = 'record_number X Y Z distance'
        sequence_id = ECHO_data.split('_')[-1].split('.')[0]
        np.savetxt(os.path.join(output_dir, 'position_' + str(sequence_id) + '.txt'), positions.T, delimiter=' ', header=header, comments='')


#load_positions()


def read_and_plot():
    positions4plot = np.array([])
    total_distance4plot = np.array([])  # Initialize as numpy array
    total_x = np.array([])  # Initialize as numpy array
    total_y = np.array([])  # Initialize as numpy array
    record_num = []
    total_record_num = [0]
    sequence_id = []


    #* Load positions
    for positions_file in natsorted(os.listdir(position_folder_path)):
        if not positions_file.endswith('.txt'):
            continue
        full_path = os.path.join(position_folder_path, positions_file)
        positions = np.loadtxt(full_path, delimiter=' ', skiprows=1) # postions file contains header

        sequence_id.append(positions_file.split('_')[-1].split('.')[0])
        record_num.append(positions.shape[0])
        total_record_num.append(total_record_num[-1] + positions.shape[0] if total_record_num else 0)


        #* Load X, Y, Z, distance
        positions4plot = np.vstack((positions4plot, positions)) if positions4plot.size else positions

        #* calculate total distance
        if total_distance4plot.size > 0:
            last_total_distance = total_distance4plot[-1]
            total_distance4plot = np.concatenate((total_distance4plot, last_total_distance + positions[:, 4]))
        else:
            total_distance4plot = positions[:, 4]

        #* calculate total position
        if total_x.size > 0:
            last_total_x = total_x[-1]
            last_total_y = total_y[-1]
            total_x = np.concatenate((total_x, last_total_x + positions[:, 1]))
            total_y = np.concatenate((total_y, last_total_y + positions[:, 2]))
        else:
            total_x = positions[:, 1]
            total_y = positions[:, 2]
    #print(record_num)
    #print(total_record_num)

    #* plot
    fontsize_large = 20
    fontsize_medium = 18

    fig = plt.figure(figsize=(20, 20), tight_layout=True)
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
            axes[i].plot(positions4plot[:, 1], label='X (North-South)')
            axes[i].plot(positions4plot[:, 2], label='Y(East-West)', linestyle='--')
            axes[i].plot(positions4plot[:, 4], label='Distance', linestyle='-.')
            axes[i].grid()
            axes[i].set_xticks(total_record_num[:len(sequence_id)], sequence_id, rotation=90)
            axes[i].set_xlim(total_record_num[split_points[i]], total_record_num[split_points[i+1] - 1])
            if i == 0:
                axes[i].legend(['X (North-South)', 'Y (East-West)', 'Distance'], loc = 'upper right')


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
    #* colorbar, メモリはsequence_id
    #cbar = plt.colorbar(ax_right_1.scatter(total_y, total_x, marker='.', c=total_distance4plot, cmap='viridis', s=10),
    #                    ax=ax_right_1, ticks=sequence_id, location='bottom', orientation='horizontal', pad=0.1)
    #cbar.set_ticks(total_distance4plot)
    #   .set_label('Sequence ID', fontsize=fontsize_medium)
    #* plot sequence_id on the track
    #for i, txt in enumerate(sequence_id):
    #    if i % 2 == 0:
    #        ax_right_1.annotate(txt, (total_y[total_record_num[i]], total_x[total_record_num[i]]))

    #* plot start point
    ax_right_1.plot(total_y[0], total_x[0], marker='*', markersize=12, color='red')
    ax_right_1.grid()
    ax_right_1.set_xlabel('Y [m] (East-West)', fontsize=fontsize_medium)
    ax_right_1.set_ylabel('X [m] (North-South)', fontsize=fontsize_medium)


    plt.savefig(os.path.join(position_folder_path, 'plot_position.png'))
    plt.show()
    return plt

read_and_plot()