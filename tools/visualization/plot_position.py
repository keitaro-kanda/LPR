import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from natsort import natsorted
import argparse


#* input data folder path
position_folder_path = input('Input data folder path (Resampled_Data>position directory): ').strip()
if not os.path.exists(position_folder_path):
    raise ValueError('Data folder path does not exist. Please check the path and try again.')
channel_name = input('Input channel name (1, 2A, 2B): ').strip()
if channel_name not in ['1', '2A', '2B']:
    raise ValueError('Invalid channel name. Please enter 1, 2A, or 2B.')

#* Data folder path
# if channel_name == '1':
#     position_folder_path = '/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_1/Resampled_Data/position'
# elif channel_name == '2A':
#     position_folder_path = '/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2A/Resampled_Data/position'
# elif channel_name == '2B':
#     position_folder_path = '/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Resampled_Data/position'
# else:
#     raise ValueError("Invalid channel name. Please enter 1, 2A, or 2B.")

#* Output folder path
output_dir = os.path.join(os.path.dirname(position_folder_path), 'position_plot')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



def load_positions():
    alternative_seq_id = []
    VELOCITY = []
    XPOSITION = [] # Reference point cordinate
    YPOSITION = [] # Reference point cordinate
    ZPOSITION = [] # Reference point cordinate
    distance = []
    X_ref = [] # Landing-site cordinate
    Y_ref = [] # Landing-site cordinate
    Z_ref = [] # Landing-site cordinate

    id = 0 # alternative to sequence_id
    for data_file in tqdm(natsorted(os.listdir(position_folder_path))):
        id += 1
        #* Load only .txt files
        if not data_file.endswith('.txt'):
            continue
        if data_file.startswith('._'):
            continue
        data_path = os.path.join(position_folder_path, data_file)
        data = np.loadtxt(data_path, delimiter=' ')


        #* Load record_count, XPOSITION, YPOSITION, ZPOSITION
        for i in range(data.shape[1]):
            #* Extract sequence_id and convert to float
            alternative_seq_id.append(id)

            #* Extract data
            VELOCITY.append(data[0, i])
            XPOSITION.append(data[1, i])
            YPOSITION.append(data[2, i])
            ZPOSITION.append(data[3, i])
            X_ref.append(data[4, i])
            Y_ref.append(data[5, i])
            Z_ref.append(data[6, i])

            distance.append(distance[-1] + np.sqrt((XPOSITION[i] - XPOSITION[i-1])**2 + (YPOSITION[i] - YPOSITION[i-1])**2) if distance else 0)

    #* 保存用に配列を作成
    positions_data = np.column_stack([
        alternative_seq_id, VELOCITY, XPOSITION, YPOSITION, ZPOSITION, X_ref, Y_ref, Z_ref
    ])
    header = 'alternative_sequence_id Velocity XPOSITION YPOSITION ZPOSITION x_ref y_ref z_ref'
    
    #* 全て浮動小数点数データとして保存
    #header = 'sequence_id Velocity XPOSITION YPOSITION ZPOSITION x_ref y_ref z_ref'
    np.savetxt(os.path.join(output_dir, 'position.txt'), positions_data,
                #fmt='%.0e %.18e %.18e %.18e %.18e %.18e %.18e %.18e',  # sequence_idは小数点なしの浮動小数点数として保存
                delimiter=' ', header=header, comments='')
    
    print('Save positions as position.txt')


    #
    total_x = np.array([])
    total_y = np.array([])
    total_z = np.array([])
    for i in range(len(XPOSITION)):
        if i == 0:
            # total_x = np.array([XPOSITION[i]])
            # total_y = np.array([YPOSITION[i]])
            # total_z = np.array([ZPOSITION[i]])
            total_x = np.array([X_ref[0]])
            total_y = np.array([Y_ref[0]])
            total_z = np.array([Z_ref[0]])
        else:
            # total_x = np.append(total_x, total_x[-1] + XPOSITION[i])
            # total_y = np.append(total_y, total_y[-1] + YPOSITION[i])
            # total_z = np.append(total_z, total_z[-1] + ZPOSITION[i])
            #total_z = np.append(total_z, ZPOSITION[i])
            total_x = np.append(total_x, X_ref[i] + XPOSITION[i])
            total_y = np.append(total_y, Y_ref[i] + YPOSITION[i])
            total_z = np.append(total_z, Z_ref[i] + ZPOSITION[i])
    
    # Runnning average
    # window_length = 200
    # window = np.ones(window_length) / window_length
    # total_x = np.convolve(total_x, window, mode='valid')
    # total_y = np.convolve(total_y, window, mode='valid')
    # total_z = np.convolve(total_z, window, mode='valid')

    #* Save total position
    total_positions_data = np.column_stack([total_x, total_y, total_z])
    np.savetxt(os.path.join(output_dir, 'total_position.txt'), total_positions_data,
                fmt='%.18e %.18e %.18e', delimiter=' ', header='total_x total_y total_z', comments='')
    #* Plot Velocity
    fig = plt.figure(figsize=(20, 10), tight_layout=True)
    plt.plot(VELOCITY)
    plt.grid()
    plt.xlabel('Record number', fontsize=20)
    plt.ylabel('Velocity [m/s]', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.savefig(os.path.join(output_dir, 'plot_velocity.png'))
    plt.show()


    #* Plot XPOSITION, YPOSITION, ZPOSITION
    plot_list = [XPOSITION, YPOSITION, ZPOSITION]
    y_label = ['x [m]', 'y [m]', 'z [m]']
    fig, ax = plt.subplots(3, 1, figsize=(20, 15), tight_layout=True, sharex=True)
    for i in range(len(plot_list)):
        ax[i].plot(plot_list[i])
        ax[i].grid()
        ax[i].set_xlabel('Record number' , fontsize=20)
        ax[i].set_ylabel(y_label[i], fontsize=20)
        ax[i].tick_params(labelsize=18)
    plt.suptitle('XPOSITION, YPOSITION, ZPOSITION', fontsize=24)

    plt.savefig(os.path.join(output_dir, 'plot_position.png'))
    plt.show()


    #* Plot x_ref, y_ref, z_ref
    plot_list = [X_ref, Y_ref, Z_ref]
    y_label = ['x [m]', 'y [m]', 'z [m]']
    fig, ax = plt.subplots(3, 1, figsize=(20, 15), tight_layout=True, sharex=True)
    for i in range(len(plot_list)):
        ax[i].plot(plot_list[i])
        ax[i].grid()
        ax[i].set_xlabel('Record number' , fontsize=20)
        ax[i].set_ylabel(y_label[i], fontsize=20)
        ax[i].tick_params(labelsize=18)
    plt.suptitle('x_ref, y_ref, z_ref', fontsize=24)

    plt.savefig(os.path.join(output_dir, 'plot_reference.png'))
    plt.show()


    #* Plot x_ref, y_ref, z_ref
    plot_list = [total_x, total_y, total_z]
    y_label = ['x [m]', 'y [m]', 'z [m]']
    fig, ax = plt.subplots(3, 1, figsize=(20, 15), tight_layout=True, sharex=True)
    for i in range(len(plot_list)):
        ax[i].plot(plot_list[i])
        ax[i].grid()
        ax[i].set_xlabel('Record number' , fontsize=20)
        ax[i].set_ylabel(y_label[i], fontsize=20)
        ax[i].tick_params(labelsize=18)
    plt.suptitle('total_x, total_y, total_zf', fontsize=24)

    plt.savefig(os.path.join(output_dir, 'plot_total.png'))
    plt.show()


    #* Plot track of CE-4
    fig = plt.figure(figsize=(20, 20), tight_layout=True)
    plt.plot(total_x, total_y)
    plt.grid()
    plt.xlabel('East-West', fontsize=20)
    plt.ylabel('North-South', fontsize=20)
    plt.tick_params(labelsize=18)

    plt.savefig(os.path.join(output_dir, 'plot_track.png'))
    plt.show()


def plot():
    positions4plot = np.array([])  # Initialize as numpy array
    total_distance4plot = np.array([])  # Initialize as numpy array
    total_x = np.array([])  # Initialize as numpy array
    total_y = np.array([])  # Initialize as numpy array
    record_num = [0]
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
            positions4plot = np.hstack((positions4plot, positions))
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

    total_record_num.insert(0, 0)
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


    fig, axes = plt.subplots(4, 1, figsize=(30, 15), tight_layout=True, sharex=True)

    #* plot velocity
    axes[0].scatter(np.arange(len(Velocity)), Velocity *100, s=5)
    axes[0].set_ylabel('Velocity [cm/s]', fontsize=fontsize_large)
    axes[0].axhline(y=5.5, color='red', linestyle='--')
    axes[0].set_ylim(0, 7)

    #* plot X, Y, Z
    axes[1].plot(X, linestyle='-', label='X (North-South)')
    axes[1].plot(Y, linestyle='--', label='Y (East-West)')
    axes[1].plot(Z, linestyle='-.', label='Z [m]')
    axes[1].set_ylabel('Rover posi [m]', fontsize=fontsize_large)
    axes[1].legend(loc='lower right', fontsize=fontsize_medium)

    #* plot reference X, Y, Z
    axes[2].plot(Reference_X, linestyle='-', label='Reference_X')
    axes[2].plot(Reference_Y, linestyle='--', label='Reference_Y')
    axes[2].plot(Reference_Z, linestyle='-.', label='Reference_Z')
    axes[2].set_ylabel('Reference point posi [m]', fontsize=fontsize_large)
    axes[2].set_yscale('log')
    axes[2].legend(loc='upper right', fontsize=fontsize_medium)

    #* plot distance
    axes[3].plot(total_distance4plot)
    axes[3].set_ylabel('Total distance [m]', fontsize=fontsize_large)

    axes[3].set_xticks(total_record_num[:len(sequence_id):2], sequence_id[::2], rotation=90)

    #* Common settings
    for i in range(4):
        axes[i].grid()
        axes[i].tick_params(labelsize=fontsize_medium)

    #* set xticks
    fig.supxlabel('Sequence ID', fontsize=fontsize_large)

    plt.savefig(os.path.join(output_dir, 'plot_position.png'))
    plt.show()


    #* Plot track of CE-4
    fig = plt.figure(figsize=(20, 20), tight_layout=True)
    #plt.scatter(total_y, total_x, c=total_distance4plot, cmap='viridis', s=10)
    #* colorbar
    cbar = plt.colorbar(plt.scatter(total_y, total_x, marker='.', c=total_distance4plot, cmap='viridis', s=5),
                        location='bottom', orientation='horizontal', pad=0.1, aspect=50)
    cbar.set_label('Distance (m)', fontsize=fontsize_medium)
    cbar.ax.tick_params(labelsize=fontsize_medium)
    #* plot start point
    plt.plot(total_y[0], total_x[0], marker='*', markersize=12, color='red')

    #* plot sequence id
    for i in range(len(sequence_id)):
        if i % 10 == 0:
            plt.text(total_y[total_record_num[i]], total_x[total_record_num[i]], sequence_id[i], fontsize=fontsize_medium)
        else:
            continue

    plt.grid()
    plt.xlabel('East-West', fontsize=fontsize_large)
    plt.ylabel('North-South', fontsize=fontsize_large)
    plt.tick_params(labelsize=fontsize_medium)


    plt.savefig(os.path.join(output_dir, 'plot_track.png'))
    plt.show()

load_positions()
#plot()



"""
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
"""