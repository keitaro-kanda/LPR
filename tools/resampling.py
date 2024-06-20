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


def remove_redundant_rows(data, threshold=0.9):
    # Calculate the mean of each row
    row_means = np.mean(data, axis=1)
    # Calculate the variance of each row
    row_variances = np.var(data, axis=1)
    # Normalize the variances
    normalized_variances = row_variances / np.max(row_variances)
    # Identify rows with normalized variance below the threshold
    redundant_rows = normalized_variances < threshold
    # Remove redundant rows
    cleaned_data = data[~redundant_rows, :]
    return cleaned_data


for ECHO_data in tqdm(natsorted(os.listdir(data_folder_path))):
    #* Load only .txt files
    if not ECHO_data.endswith('.txt'):
        continue
    if ECHO_data.startswith('._'):
        continue

    ECHO_data_path = os.path.join(data_folder_path, ECHO_data)
    data = np.loadtxt(ECHO_data_path, delimiter=' ')

    ECHO = data[5:, :]

    #* Remove redundant rows
    cleaned_ECHO = remove_redundant_rows(ECHO)

    #* Save cleaned data
    cleaned_data_path = os.path.join(output_dir, f'cleaned_{ECHO_data}')
    np.savetxt(cleaned_data_path, cleaned_ECHO, delimiter=' ')

    plt.figure(figsize=(10, 10))
    plt.imshow(cleaned_ECHO, aspect='auto', cmap='seismic')
    plt.colorbar()
    plt.title(f'Cleaned Data: {ECHO_data}')
    plt.savefig(os.path.join(output_dir, f'cleaned_{ECHO_data}.png'))
    plt.close()
print('Data resampling completed')