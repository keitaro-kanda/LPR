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
if args.path_type == 'local':
    data_folder_path = 'LPR_2B/ECHO'
elif args.path_type == 'SSD':
    data_folder_path = '/Volumes/SSD_kanda/LPR/LPR_2B/ECHO'
output_dir = os.path.join(os.path.dirname(data_folder_path), 'Resampled_ECHO')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



def remove_redundant_rows(data, record_num):
    #* Calculate the average of previous five records
    difference_criteria = 6000

    if record_num ==0:
        Resampled_ECHO.append(np.insert(data[:, record_num], 0, record_num))
    else:
        if record_num > 10:
            average = np.mean(data[:, record_num - 10:record_num], axis=1)
            difference = np.abs(data[:, record_num] - average)
        else:
            difference = np.abs(data[:, record_num] - data[:, record_num - 1])
        if np.sum(difference) > difference_criteria:
            Resampled_ECHO.append(np.insert(data[:, record_num], 0, record_num))


#* Resampling
for ECHO_data in natsorted(os.listdir(data_folder_path)):
    #* Load ECHO data
    if not ECHO_data.endswith('.txt'):
        continue
    if ECHO_data.startswith('._'):
        continue

    ECHO_data_path = os.path.join(data_folder_path, ECHO_data)
    data = np.loadtxt(ECHO_data_path, delimiter=' ', skiprows=11)

    print(' ')
    print('ECHO data:', ECHO_data)
    print('record count number:', data.shape[1])
    #* Resampling
    Resampled_ECHO = []
    for i in tqdm(range(data.shape[1]), desc=ECHO_data):
        remove_redundant_rows(data, i)
    Resampled_ECHO = np.array(Resampled_ECHO).T
    print(Resampled_ECHO.shape)


    #* Plot which record is resampled on the data plot
    resampled_record = Resampled_ECHO[0, :]
    print('Number of resampled record:', len(resampled_record))
    plt.figure(tight_layout=True)
    plt.imshow(data, aspect='auto', cmap='seismic',
                extent=[0, data.shape[1], data.shape[0]*0.3125, 0],
                vmin=-50, vmax=50)
    plt.xlabel('Record count')
    plt.ylabel('Time [ns]')
    plt.colorbar()
    #* Shade resampled record
    for i in range(len(resampled_record)):
        plt.axvspan(resampled_record[i], resampled_record[i]+1, color='gray', alpha=0.1)
    plt.show()


    #* Plot resampled ECHO data
    plt.figure(tight_layout=True)
    plt.imshow(Resampled_ECHO, aspect='auto', cmap='seismic',
                extent=[0, Resampled_ECHO.shape[1], Resampled_ECHO.shape[0]*0.3125, 0],
                vmin=-50, vmax=50)
    plt.xlabel('Record count')
    plt.ylabel('Time [ns]')
    plt.colorbar()

    plt.show()