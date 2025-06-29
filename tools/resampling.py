import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from natsort import natsorted
import argparse
import cv2
from scipy import ndimage


#* input data folder path
channel_name = input('Input channel name (1, 2A, 2B): ').strip()

if channel_name == '1':
    data_folder_path = "/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_1/loaded_data_echo_position"
elif channel_name == '2A':
    data_folder_path = "/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2A/loaded_data_echo_position"
elif channel_name == '2B':
    data_folder_path = "/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/loaded_data_echo_position"
else:
    raise ValueError('Invalid channel name. Please enter 1, 2A, or 2B.')


#* Define output folder path
output_dir = os.path.join(os.path.dirname(data_folder_path), 'Resampled_Data')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
txt_output_dir = os.path.join(output_dir, 'txt')
if not os.path.exists(txt_output_dir):
    os.makedirs(txt_output_dir)
plot_output_dir = os.path.join(output_dir, 'plot')
if not os.path.exists(plot_output_dir):
    os.makedirs(plot_output_dir)
position_output_dir = os.path.join(output_dir, 'position')
if not os.path.exists(position_output_dir):
    os.makedirs(position_output_dir)


#* init data_2B for data and medf for interest extrapolation
#* 2048 is read by the header .2BL and it is fixed for all LPR_2B files [time dims]
#data_2B = np.zeros((0, 2048))
#medf = np.zeros(0)

def resampling(signal_data, position_data, sequence_id): # input is 2D array including position data
    #* Do not consider first 300 datapoints
    img = signal_data[300:, :]

    #* Compute the derivative of the data in x direction with Sobel with a kernel size of 5
	#* This should eliminate the horizonatal lines
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

    #* Denoise the sobelx with a median filter of size 5
	#* We need to reduce random noise for the next step
    med_denoised = ndimage.median_filter(sobelx, 5)

    #* Sum the absolute values of the denoised traces on time dims
	#* We should have cancelled signal_data on death traces --> highest value on the real traces
    med = np.sum(np.abs(med_denoised), axis = 0)

	#* Define a moving window to avoid taking into account small portions with high values.
	#* We want to take the signal_data only when it has a value highest than thres for window consecutive traces
    window = 16
    thres = 25000


	#* Init output vector as 0s
    medf = np.zeros(med.shape)
	#* for each trace
    for i in range(med.shape[0] - window):
		#* if all values are > thres
        if np.all(med[i : i + window] > thres):
            #* replace 0 with 1 in the output vector
            medf[i : i + window] = 1

    idx = np.where(medf == 1)[0]

    #* Save idx when the channel is 2B
    if channel_name == '2B':
        idx_dir = os.path.join(output_dir, 'idx')
        if not os.path.exists(idx_dir):
            os.makedirs(idx_dir)
        np.savetxt(idx_dir + '/' + str(sequence_id) + '_idx.txt', idx, delimiter=' ')

    #* Filter the data with the idx
    if np.sum(idx) == 0:
        data_filtered = signal_data[:, idx]
        position_data_filtered = position_data[:, idx]
        print('No interesting data found')
    else:
        data_filtered = signal_data[:, idx]
        position_data_filtered = position_data[:, idx]
        print('Raw data shape was: ', signal_data.shape)
        print('Filtered data shape is: ', data_filtered.shape)


        #* Save filtered data as .txt file
        np.savetxt(txt_output_dir + '/' + str(sequence_id) + '_resampled.txt', data_filtered, delimiter=' ')
        np.savetxt(position_output_dir + '/' + str(sequence_id) + '_resampled_position.txt', position_data_filtered, delimiter=' ')
        header_position = 'velocity, position_x, position_y, position_z, reference_point_x, reference_point_y, reference_point_z'
        np.savetxt(position_output_dir + '/' + str(sequence_id) + '_resampled_position.txt',
                    position_data_filtered, delimiter=' ', header=header_position)

    return sobelx, med_denoised, med, medf, data_filtered, position_data_filtered


#* Define the plot function for CH-2B
def plot_2B(raw_data, sobelx, med_denoised, med, medf, data_filtered, sequence_id):
    fig, ax = plt.subplots(6, 1, figsize=(12, 20), tight_layout=True, sharex=True)
    fontsize_large = 20
    fontsize_medium = 18
    fontsize_small = 16

    #* Plot raw data
    ax[0].imshow(raw_data, aspect='auto', cmap='seismic',
                extent=[0, raw_data.shape[1], raw_data.shape[0]*0.3125, 0],
                vmin=-15, vmax=15)
    ax[0].set_title('Raw data: ' + str(sequence_id), fontsize=fontsize_large)
    ax[0].set_ylabel('Time [ns]', fontsize=fontsize_medium)
    ax[0].tick_params(axis='both', which='major', labelsize=fontsize_small)

    #* Plot Sobel filtered data
    ax[1].imshow(sobelx, aspect='auto', cmap='seismic',
                extent=[0, sobelx.shape[1], sobelx.shape[0]*0.3125, 0],
                vmin=-15, vmax=15)
    ax[1].set_title('Sobel filtered', fontsize=fontsize_large)
    ax[1].set_ylabel('Time [ns]', fontsize=fontsize_medium)
    ax[1].tick_params(axis='both', which='major', labelsize=fontsize_small)

    #* Plot median denoised data
    ax[2].imshow(med_denoised, aspect='auto', cmap='seismic',
                extent=[0, med_denoised.shape[1], med_denoised.shape[0]*0.3125, 0],
                vmin=-15, vmax=15)
    ax[2].set_title('Random noise reduction', fontsize=fontsize_large)
    ax[2].set_ylabel('Time [ns]', fontsize=fontsize_medium)
    ax[2].tick_params(axis='both', which='major', labelsize=fontsize_small)

    #* Plot median data
    ax[3].plot(med, label='med')
    ax[3].hlines(20000, 0, med.shape[0], 'r', label='thres')
    ax[3].set_title('Filtered signals amplitude', fontsize=fontsize_large)
    ax[3].legend()
    ax[3].set_ylabel('Amplitude', fontsize=fontsize_medium)
    ax[3].tick_params(axis='both', which='major', labelsize=fontsize_small)

    #* Plot median filtered data
    ax[4].plot(medf)
    ax[4].set_title('Gate with interesting data', fontsize=fontsize_large)
    ax[4].tick_params(axis='both', which='major', labelsize=fontsize_small)

    #* Plot filtered data
    ax[5].imshow(data_filtered, aspect='auto', cmap='seismic',
                extent=[0, data_filtered.shape[1], data_filtered.shape[0]*0.3125, 0],
                vmin=-15, vmax=15)
    ax[5].set_title('Filtered ECHO data', fontsize=fontsize_large)
    ax[5].set_ylabel('Time [ns]', fontsize=fontsize_medium)
    ax[5].tick_params(axis='both', which='major', labelsize=fontsize_small)

    plt.xlim(0, raw_data.shape[1])
    fig.supxlabel('Trace number', fontsize=fontsize_medium)

    plt.savefig(plot_output_dir + '/' + str(sequence_id) + '_resampling_flow.png')
    plt.close()


#* Define the plot function for CH-2A
def plot_2A(raw_data, medf, data_filtered, sequence_id):
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), tight_layout=True, sharex=True)
    fontsize_large = 20
    fontsize_medium = 18
    fontsize_small = 16

    #* Plot raw data
    ax[0].imshow(raw_data, aspect='auto', cmap='seismic',
                extent=[0, raw_data.shape[1], raw_data.shape[0]*0.3125, 0],
                vmin=-15, vmax=15)
    ax[0].set_title('Raw data: ' + str(sequence_id), fontsize=fontsize_large)
    ax[0].set_ylabel('Time [ns]', fontsize=fontsize_medium)
    ax[0].tick_params(axis='both', which='major', labelsize=fontsize_small)

    #* Plot median filtered data
    ax[1].plot(medf)
    ax[1].set_title('Gate with interesting data', fontsize=fontsize_large)
    ax[1].tick_params(axis='both', which='major', labelsize=fontsize_small)

    #* Plot filtered data
    # data_filtered4plot = np.zeros((2048, data_filtered.shape[1]))
    # data_filtered4plot[:, :data_filtered.shape[1]] = data_filtered
    ax[2].imshow(data_filtered, aspect='auto', cmap='seismic',
                extent=[0, data_filtered.shape[1], data_filtered.shape[0]*0.3125, 0],
                vmin=-15, vmax=15)
    ax[2].set_title('Filtered ECHO data', fontsize=fontsize_large)
    ax[2].set_ylabel('Time [ns]', fontsize=fontsize_medium)
    ax[2].tick_params(axis='both', which='major', labelsize=fontsize_small)

    plt.xlim(0, raw_data.shape[1])
    fig.supxlabel('Trace number', fontsize=fontsize_medium)
    plt.savefig(plot_output_dir + '/' + str(sequence_id) + '_resampling_flow.png')
    plt.close()



#* Conduct resampling
total_trace_num = 0
resampled_trace_num = 0
for ECHO_data in tqdm(natsorted(os.listdir(data_folder_path))):
    #* Load ECHO data
    if not ECHO_data.endswith('.txt'):
        continue
    if ECHO_data.startswith('._'):
        continue

    #* Load the data
    ECHO_data_path = os.path.join(data_folder_path, ECHO_data)
    raw_data = np.loadtxt(ECHO_data_path, delimiter=' ', skiprows=1) # skip the header
    positions = raw_data[:7, :] # 7: B-scan data
    signals = raw_data[7:, :] # 0-6: velocity and position data
    #* Check the sequence ID
    sequence_id = ECHO_data.split('_')[-1].split('.')[0]

    print(' ')
    print('---------------------------------')
    print('ECHO data:', ECHO_data)
    print('record count number:', raw_data.shape[1])
    #* Count the number of traces before resampling
    total_trace_num += raw_data.shape[1]
    print('Now processing...')
    print('   ')

    #* Resampling
    """
    1: Not implemented yet
    2A: Using idx list from 2B and resample according to the idx list
    2B: Apply resampling function
    """
    if channel_name == '2B':
        sobelx, med_denoised, med, medf, data_filtered, positions_filtered = resampling(signals, positions, sequence_id)
        # data_filtered4plot = np.zeros((2048, data_filtered.shape[1]))
        # data_filtered4plot[:, :data_filtered.shape[1]] = data_filtered

        plot_2B(raw_data, sobelx, med_denoised, med, medf, data_filtered, sequence_id)
        print(' ')

    elif channel_name == '2A':
        #* Load idx from 2B
        idx_dir_path = '/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Resampled_Data/idx'
        idx_file_path = os.path.join(idx_dir_path, sequence_id + '_idx.txt')
        if not os.path.exists(idx_file_path):
            print(f'No idx file found for sequence {sequence_id}. Skipping...')
            continue
        idx = np.loadtxt(idx_file_path, delimiter=' ')
        
        #* Make window data for plotting
        if idx.ndim == 0:
            continue  # Skip if idx is empty
        else:
            medf = np.zeros(raw_data.shape[1])
            medf[idx.astype(int)] = 1

        #* Filter the data with the idx
        data_filtered = signals[:, idx.astype(int)]
        positions_filtered = positions[:, idx.astype(int)]
        print('Raw data shape was: ', raw_data.shape)
        print('Filtered data shape is: ', data_filtered.shape)
        #* Save filtered data as .txt file
        np.savetxt(txt_output_dir + '/' + str(sequence_id) + '_resampled.txt', data_filtered, delimiter=' ')
        np.savetxt(position_output_dir + '/' + str(sequence_id) + '_resampled_position.txt', positions_filtered, delimiter=' ')
        header_position = 'velocity, position_x, position_y, position_z, reference_point_x, reference_point_y, reference_point_z'
        np.savetxt(position_output_dir + '/' + str(sequence_id) + '_resampled_position.txt',
                    positions_filtered, delimiter=' ', header=header_position)

        plot_2A(raw_data, medf, data_filtered, sequence_id)
        print(' ')

    elif channel_name == '1':
        print('Channel 1 does not support resampling. Skipping...')
        continue
    else:
        raise ValueError('Invalid channel name. Please enter 1, 2A, or 2B.')


    #* Count the number of traces after resampling
    resampled_trace_num += data_filtered.shape[1]
    if ECHO_data == natsorted(os.listdir(data_folder_path))[-1]:
        with open(output_dir + '/total_trace_num.txt', 'w') as f:
            f.write('Number of total traces before resampling: ' + str(total_trace_num) + '\n')
            f.write('Number of total traces after resampling: '+ str(resampled_trace_num))



