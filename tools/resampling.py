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
    average = np.mean(data[:, record_num - 5:record_num], axis=1)

    difference = np.abs(data[:, record_num] - average)

    #* If the difference is larger than criteria, remove the record
    difference_criteria = 0.1
    if np.sum(difference) > difference_criteria:
        np.insert(data[:, record_num], 0, record_num)
        Resampled_ECHO.append(data[:, record_num])


#* Resampling
for ECHO_data in natsorted(os.listdir(data_folder_path)):
    #* Load ECHO data
    if not ECHO_data.endswith('.txt'):
        continue
    if ECHO_data.startswith('._'):
        continue

    ECHO_data_path = os.path.join(data_folder_path, ECHO_data)
    data = np.loadtxt(ECHO_data_path, delimiter=' ', skiprows=11)


    #* Resampling
    for i in tqdm(range(data.shape[1]), desc=ECHO_data):
        Resampled_ECHO = []
        remove_redundant_rows(data, i)
    print(np.array(Resampled_ECHO).shape)

    plt.figure()
    plt.imshow(Resampled_ECHO, aspect='auto')
    plt.colorbar()

    plt.show()
