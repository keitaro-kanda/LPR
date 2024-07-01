import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from natsort import natsorted
import argparse


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='resampling.py',
    description='Resampling ECHO data',
    epilog='End of help message',
    usage='python tools/resampling.py [path_type]',
)
parser.add_argument('path_type', choices = ['local', 'SSD'], help='Choose the path type')
args = parser.parse_args()


#* Define data folder path
if args.path_type == 'local' or args.path_type == 'test':
    data_folder_path = 'LPR_2B/ECHO'
elif args.path_type == 'SSD':
    data_folder_path = '/Volumes/SSD_kanda/LPR/LPR_2B/ECHO'


#* Define output folder path
output_dir = os.path.join(os.path.dirname(data_folder_path), 'Resampled_ECHO')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
txt_output_dir = os.path.join(output_dir, 'txt')
if not os.path.exists(txt_output_dir):
    os.makedirs(txt_output_dir)
plot_output_dir = os.path.join(output_dir, 'plot')
if not os.path.exists(plot_output_dir):
    os.makedirs(plot_output_dir)


def calc_difference(data, record_num):
    """
    #* Calculate the average of previous five records
    if record_num == 0:
        #Resampled_ECHO.append(np.insert(data[:, record_num], 0, record_num))
        difference_list.append(0)
    else:
        difference = np.abs(data[:, record_num] - data[:, record_num - 1])
        #if np.sum(difference) > difference_criteria:
        #Resampled_ECHO.append(np.insert(data[:, record_num], 0, record_num))
        difference_list.append(np.sum(difference))
    """
    if record_num ==0:
        resampled_ECHO.append(np.insert(data[:, record_num], 0, record_num))
    else:
        if record_num >= 5:
            average = np.mean(data[:, record_num - 5:record_num], axis=1)
            difference = np.abs(data[:, record_num] - average)
            difference_list.append(np.sum(difference))
        else:
            average = np.mean(data[:, :record_num], axis=1)
            difference = np.abs(data[:, record_num] - average)
            difference_list.append(np.sum(difference))
        if np.sum(difference) > criteria:
            resampled_ECHO.append(np.insert(data[:, record_num], 0, record_num))


def calc_difference_2(data):
    for i in range(data.shape[1]):
        if i > 0:
            difference = np.abs(data[1:, i] - data[1:, i-1]) # 0行目はrecord numberなので除外
            if np.sum(difference) < 4000:
                np.delete(data, i, axis=1)

#* Calculate running average of difference_list
def calc_running_average(list, record_num):
    if record_num == 0:
        running_average_list.append(0)
        resampled_ECHO.append(np.insert(data[:, record_num], 0, record_num))
        resampled_record.append(record_num)
    if i >= 10:
        running_average = np.mean(list[i-10:i])
        running_average_list.append(running_average)
        if running_average > criteria:
            resampled_ECHO.append(np.insert(data[:, record_num], 0, record_num))
            resampled_record.append(record_num)
    else:
        running_average = np.mean(list[:i])
        running_average_list.append(running_average)
        if running_average > criteria:
            resampled_ECHO.append(np.insert(data[:, record_num], 0, record_num))
            resampled_record.append(record_num)


criteria = 6500
#* Resampling
for ECHO_data in natsorted(os.listdir(data_folder_path)):
    #* Load ECHO data
    if not ECHO_data.endswith('.txt'):
        continue
    if ECHO_data.startswith('._'):
        continue

    ECHO_data_path = os.path.join(data_folder_path, ECHO_data)
    data = np.loadtxt(ECHO_data_path, delimiter=' ', skiprows=11)
    sequence_id = ECHO_data.split('_')[-1].split('.')[0]

    print(' ')
    print('ECHO data:', ECHO_data)
    print('record count number:', data.shape[1])


    #* Resampling
    resampled_ECHO = []
    difference_list = []
    running_average_list = []
    resampled_record = []

    #* Calculate difference and running average of difference
    for i in tqdm(range(data.shape[1]), desc=ECHO_data + ' Calculate difference'):
        calc_difference(data, i)
    resampled_ECHO = np.array(resampled_ECHO)
    #for i in tqdm(range(len(difference_list)), desc=ECHO_data + ' Calculate running average'):
    #    calc_running_average(difference_list, i)
    #for i in tqdm(range(len(resampled_ECHO)), desc=ECHO_data + ' Calculate difference 2'):
    #    calc_difference_2(resampled_ECHO)
    print('Resampled_ECHO:', len(resampled_ECHO))
    resampled_ECHO = np.array(resampled_ECHO).T

    #print(Resampled_ECHO.shape)

    #* Save resampled ECHO data
    #np.savetxt(txt_output_dir + '/' + sequence_id + '_resampled.txt', Resampled_ECHO, delimiter=' ')


    #* Plot which record is resampled on the data plot
    resampled_record = resampled_ECHO[0, :]
    font_large = 20
    font_medium = 18
    print('Number of resampled record:', len(resampled_record))


    #* Plot original B-scan
    plt.figure(figsize=(15, 10), tight_layout=True)
    plt.imshow(data, aspect='auto', cmap='seismic',
                extent=[0, data.shape[1], data.shape[0]*0.3125, 0],
                vmin=-50, vmax=50)
    plt.xlabel('Record count', fontsize=font_large)
    plt.ylabel('Time [ns]', fontsize=font_large)
    plt.title('Sequence ID: ' + sequence_id, fontsize=font_large)
    plt.tick_params(labelsize=font_medium)

    cbar = plt.colorbar()
    cbar.set_label('Amplitude', fontsize=font_large)
    cbar.ax.tick_params(labelsize=font_medium)

    plt.savefig(plot_output_dir + '/' + sequence_id + '_original.png')
    plt.show()
    plt.close()


    #* Plot original B-scan and shade resampled record
    fig, ax = plt.subplots(2,1, figsize=(15, 10), tight_layout=True, sharex=True)
    ax[0].imshow(data, aspect='auto', cmap='seismic',
                extent=[0, data.shape[1], data.shape[0]*0.3125, 0],
                vmin=-50, vmax=50)
    ax[0].set_ylabel('Time [ns]', fontsize=font_large)
    ax[0].tick_params(labelsize=font_medium)

    #* Shade resampled record
    for i in range(len(resampled_record)):
        ax[0].axvspan(resampled_record[i], resampled_record[i]+1, color='gray', alpha=0.1)

    #cbar = plt.colorbar(ax[0].images[0], ax=ax[0], )
    #cbar.set_label('Amplitude', fontsize=font_large)
    #cbar.ax.tick_params(labelsize=font_medium)

    ax[1].plot(difference_list, label='Difference')
    ax[1].plot(running_average_list, label='Running average')
    ax[1].set_ylabel('Difference', fontsize=font_large)
    ax[1].tick_params(labelsize=font_medium)
    ax[1].hlines(criteria, 0, len(difference_list), 'k', linestyles='dashed', label='Difference criteria')
    ax[1].legend(fontsize=font_medium)

    fig.suptitle('Sequence ID: ' + sequence_id, fontsize=font_large)
    fig.supxlabel('Record count', fontsize=font_large)

    plt.savefig(plot_output_dir + '/' + sequence_id + '_shaded.png')
    plt.show()
    plt.close()

    """
    plt.figure(figsize=(15, 10), tight_layout=True)
    plt.imshow(data, aspect='auto', cmap='seismic',
                extent=[0, data.shape[1], data.shape[0]*0.3125, 0],
                vmin=-50, vmax=50)
    plt.xlabel('Record count', fontsize=font_large)
    plt.ylabel('Time [ns]', fontsize=font_large)
    plt.title('Sequence ID: ' + sequence_id, fontsize=font_large)
    plt.tick_params(labelsize=font_medium)

    cbar = plt.colorbar()
    cbar.set_label('Amplitude', fontsize=font_large)
    cbar.ax.tick_params(labelsize=font_medium)
    #* Shade resampled record
    for i in range(len(resampled_record)):
        plt.axvspan(resampled_record[i], resampled_record[i]+1, color='gray', alpha=0.1)

    #plt.savefig(plot_output_dir + '/' + sequence_id + '_shaded.png')
    plt.show()
    #plt.close()


    #* Plot difference between the original data and the resampled data
    plt.figure(tight_layout=True)
    plt.plot(difference_list)
    plt.hlines(6000, 0, len(difference_list), 'r', linestyles='dashed')
    plt.xlabel('Record count')
    plt.ylabel('Difference')
    plt.show()
    """



    #* Plot resampled ECHO data
    plt.figure(figsize=(15, 10), tight_layout=True)
    plt.imshow(resampled_ECHO, aspect='auto', cmap='seismic',
                extent=[0, resampled_ECHO.shape[1], resampled_ECHO.shape[0]*0.3125, 0],
                vmin=-50, vmax=50)
    plt.xlabel('Record count', fontsize=font_large)
    plt.ylabel('Time [ns]', fontsize=font_large)
    plt.title('Sequence ID: ' + sequence_id, fontsize=font_large)
    plt.tick_params(labelsize=font_medium)

    cbar = plt.colorbar()
    cbar.set_label('Amplitude', fontsize=font_large)
    cbar.ax.tick_params(labelsize=font_medium)

    plt.savefig(plot_output_dir + '/' + sequence_id + '_resampled.png')
    plt.show()
    plt.close()
