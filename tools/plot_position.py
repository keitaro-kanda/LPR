import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec


#data_folder_path = 'LPR_2B/original_data'
data_folder_path = '/Volumes/SSD_kanda/LPR/LPR_2B/ECHO'



def load_positions():
    XPOSITION = []
    YPOSITION = []
    ZPOSITION = []
    record_number = []
    sequence_id = []
    for ECHO_data in tqdm(os.listdir(data_folder_path)):
        #* Load only .txt files
        if not ECHO_data.endswith('.txt'):
            continue
        if ECHO_data.startswith('._'):
            continue

        ECHO_data_path = os.path.join(data_folder_path, ECHO_data)
        data = np.loadtxt(ECHO_data_path, skiprows=1, delimiter=' ')
        for i in range(data.shape[1]):
            XPOSITION.append(data[1, i])
            YPOSITION.append(data[2, i])
            ZPOSITION.append(data[3, i])

        record_number.append(len(XPOSITION)) # その時点でのrecord_numberの総和を記録
        sequence_id.append(ECHO_data.split('_')[-1].split('.')[0])


    XPOSITION = np.array(XPOSITION)
    YPOSITION = np.array(YPOSITION)
    ZPOSITION = np.array(ZPOSITION)

    output_dir = os.path.join(os.path.dirname(data_folder_path), 'Position')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(output_dir + '/position.txt', np.array([XPOSITION, YPOSITION, ZPOSITION]))
    np.savetxt(output_dir + '/record_number.txt', record_number)
    np.savetxt(output_dir + '/sequence_id.txt', sequence_id, fmt='%s')

    return XPOSITION, YPOSITION, ZPOSITION, record_number, sequence_id

#XPOSITION, YPOSITION, ZPOSITION, record_number, sequence_id = load_positions()


def read_and_plot():
    #* Load positions
    positions = np.loadtxt('/Volumes/SSD_kanda/LPR/LPR_2B/Position/position.txt', delimiter=' ')
    XPOSITION = positions[0]
    YPOSITION = positions[1]
    ZPOSITION = positions[2]
    data_point_number = np.arange(0, len(XPOSITION))

    #* Load record_number and sequence_id
    record_number_total = np.loadtxt('/Volumes/SSD_kanda/LPR/LPR_2B/Position/record_number.txt', dtype=int)
    record_number = []

    #* record_number_listからその時点でのrecord_numberの総和を計算してrecord_numberに追加
    for i in range(0, len(record_number_total)):
        record_number.append(record_number_total[i] - record_number_total[i-1] if i else record_number_total[i])
    sequence_id = np.loadtxt('/Volumes/SSD_kanda/LPR/LPR_2B/Position/sequence_id.txt', dtype=str)
    print('Length of record_number:', len(record_number))
    print('Length of sequence_id:', len(sequence_id))

    distance = []
    #* 2点間の距離を計算してdistanceに追加，ただしそれぞれのrecord_numberで区切る
    for i in tqdm(range(len(record_number)), desc='Calculating distance'):
        X = XPOSITION[record_number_total[i] - record_number[i] : record_number_total[i]]
        Y = YPOSITION[record_number_total[i] - record_number[i] : record_number_total[i]]
        for j in range(len(X) - 1):
            distance.append(distance[-1] + np.sqrt((X[j+1] - X[j])**2 + (Y[j+1] - Y[j])**2) if distance else 0)


    #* Total positonの計算
    total_x = []
    total_y = []
    for i in tqdm(range(len(record_number)), desc='Calculating total distance'):
        if i == 0:
            total_x.append(XPOSITION[0 : record_number_total[i]])
            total_y.append(YPOSITION[0 : record_number_total[i]])
        else:
            last_x = total_x[-1]
            last_y = total_y[-1]
            x_list = XPOSITION[record_number_total[i] - record_number[i] : record_number_total[i]] + last_x[-1]
            y_list = YPOSITION[record_number_total[i] - record_number[i] : record_number_total[i]] + last_y[-1]
            total_x.append(x_list)
            total_y.append(y_list)
    print('Length of total_x:', len(total_x))

    #* plot
    fontsize_large = 20
    fontsize_medium = 18
    split_points = [0, len(record_number) // 4, len(record_number) // 2, 3 * len(record_number) // 4, len(record_number)]

    fig = plt.figure(figsize=(20, 20), tight_layout=True)
    gs = GridSpec(5, 2, width_ratios = [1, 1], figure=fig)

    #* Left side panels
    ax_left_0 = fig.add_subplot(gs[0, 0])
    ax_left_1 = fig.add_subplot(gs[1, 0])
    ax_left_2 = fig.add_subplot(gs[2, 0])
    ax_left_3 = fig.add_subplot(gs[3, 0])
    ax_left_4 = fig.add_subplot(gs[4, 0])
    axes = [ax_left_0, ax_left_1, ax_left_2, ax_left_3]

    for i in range(len(axes)):
            axes[i].scatter(data_point_number, XPOSITION, label='X (North-South)')
            axes[i].scatter(data_point_number, YPOSITION, label='Y(East-West)')
            axes[i].grid()
            axes[i].set_xticks(record_number_total, sequence_id, rotation=90)
            axes[i].set_xlim(record_number_total[split_points[i]], record_number_total[split_points[i+1] - 1])
    ax_left_4.plot(distance, '-', label='Distance')
    ax_left_4.set_xlabel('Trace number', fontsize=fontsize_medium)
    ax_left_4.set_ylabel('Distance (m)', fontsize=fontsize_medium)
    ax_left_4.grid()
    ax_left_4.set_xticks(record_number_total, sequence_id, rotation=90)
    ax_left_4.set_xlim(0, len(XPOSITION))

    fig.legend(['X (North-South)', 'Y (East-West)'], fontsize=fontsize_medium)

    #* Right side panels
    ax_right = fig.add_subplot(gs[:, 1])
    ax_right.plot(YPOSITION, XPOSITION, marker='.')
    ax_right.grid()
    ax_right.set_xlabel('Y (East-West)', fontsize=fontsize_medium)
    ax_right.set_ylabel('X (North-South)', fontsize=fontsize_medium)

    plt.show()
    return plt

read_and_plot()