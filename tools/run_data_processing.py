import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
import argparse
from tqdm import tqdm
from scipy import signal

#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='run_data_processing.py',
    description='Run bandpass filtering, time-zero correction, and background removal on resampled Bscan data',
    epilog='End of help message',
    usage='python tools/runs_data_processing.py [path_type] [function_type]',
)
parser.add_argument('path_type', choices = ['local', 'SSD'], help='Choose the path type')
parser.add_argument('function_type', choices=['calc', 'plot'], help='Choose the function type')
args = parser.parse_args()


#* Define data folder path
if args.path_type == 'local' or args.path_type == 'test':
    data_path = 'LPR_2B/Resampled_ECHO/Bscan.txt'
elif args.path_type == 'SSD':
    data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Resampled_data/Bscan.txt'
    print('Data path:', data_path)



#* Define output folder path
output_dir = os.path.join(os.path.dirname(os.path.dirname(data_path)), 'Processed_Data')
os.makedirs(output_dir, exist_ok=True)

dir_0 = os.path.join(output_dir, '0_Raw_data')
os.makedirs(dir_0, exist_ok=True)
dir_1 = os.path.join(output_dir, '1_Bandpass_filter')
os.makedirs(dir_1, exist_ok=True)
dir_2 = os.path.join(output_dir, '2_Time_zero_correction')
os.makedirs(dir_2, exist_ok=True)
dir_3 = os.path.join(output_dir, '3_Background_removal')
os.makedirs(dir_3, exist_ok=True)
dir_4 = os.path.join(output_dir, '4_Gain_function')
os.makedirs(dir_4, exist_ok=True)
dir_5 = os.path.join(output_dir, 'Merged_plot')
os.makedirs(dir_5, exist_ok=True)
print('Output dir:', output_dir)
print('   ')



#* Define parameters
sample_interval = 0.312500e-9# [s]
fs = 1/sample_interval  # Sampling frequency
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]
epsilon_0 = 8.854187817e-12  # [F/m]
c = 299792458  # [m/s]


#* Define functions
#* Bandpass filter
def bandpass_filter(input_data, lowcut, highcut, order): # Ascandata is 1D array
    #* Filter specifications
    #* Design the Butterworth band-pass filter
    #order = 4
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = signal.butter(order, [low, high], btype='band')

    #* Apply bandpass filter
    filtered_data = signal.filtfilt(b, a, input_data)

    return filtered_data



#* Time-zero correction
def time_correction(data):
    time_corrected_data = np.zeros(data.shape)

    #for i in tqdm(range(data.shape[1]), desc='Applying time-zero correction'):
    #    idx = np.where(np.abs(data[:, i]>0.5))[0][0]
    #    print('Index:', idx)
    #    time_corrected_data[:-idx, i] = data[idx:, i]

    #* Find peak time
    peak = [] # List of data point of peak
    #for i in tqdm(range(data.shape[1]), desc='Finding peak time'):
    #        peak_index = np.argmax(data[:, i])  # index number, not time
    #        peak.append(peak_index)
    for i in tqdm(range(data.shape[1]), desc='Finding peak time'):
            peak_index = np.where(data[:, i]>0)[0][0] # index number, not time
            peak.append(peak_index)
    time_zero_point = np.max(peak)
    #time_zero_record = peak.index(time_zero_point)
    print('Time zero point: ', time_zero_point)
    print('New time zero [s]', time_zero_point * sample_interval * 1e9, ' ns')
    for i in tqdm(range(data.shape[1]), desc='Aligning peak time'):
        shift = int((time_zero_point - peak[i])) # shift index number
        #* シフトする分はゼロで埋める
        if shift > 0:
            time_corrected_data[:, i] = np.concatenate([np.zeros(shift), data[:-shift, i]])
        elif shift < 0:
            time_corrected_data[:, i] = np.concatenate([data[np.abs(shift):, i], np.zeros(np.abs(shift))])
        else:
            time_corrected_data[:, i] = data[:, i]
    return time_corrected_data



#* Holizontal high pass filter
def background_removal(data):
    background_data = np.mean(data, axis=1)
    background_removed_data = np.zeros_like(data)
    for i in tqdm(range(data.shape[1]), desc='Subtracting background'):
        background_removed_data[:, i] =  data[:, i] - background_data
    return background_data,  background_removed_data



def gain(data, er, tan_delta, freq):
    t_2D = np.expand_dims(np.arange(0, data.shape[0]*sample_interval, sample_interval), axis=1)
    gain_func = t_2D**2 * c**2 / (4 * er) * np.exp(np.pi * t_2D * freq * np.sqrt(er * epsilon_0)* tan_delta)

    #* Plot gain function
    fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
    ax.plot(gain_func, t_2D/1e-9)

    ax.set_xlabel('Gain function', fontsize=20)
    ax.set_xscale('log')
    ax.set_xlim(1e-2, 1e5)
    ax.set_ylabel('2-way travel time [ns]', fontsize=20)
    ax.invert_yaxis()
    ax.tick_params(labelsize=18)
    ax.grid(which='major', axis='both', linestyle='-.')
    ax.text(0.1, 0.1, r'$\varepsilon_r = $' + str(er) + ', tan$\delta = $' + str(round(tan_delta, 3)),
            fontsize=18, transform=ax.transAxes)

    plt.savefig(dir_4 + '/Gain_function.png', format='png', dpi=120)
    plt.savefig(dir_4 + '/Gain_function.pdf', format='pdf', dpi=300)
    plt.close()
    print('Plot of gain function is successfully saved.')
    print(' ')

    output = data * gain_func
    return output


#* Processing pipeline
if args.function_type == 'calc':
    #* 0. Prepare raw Bscan data
    Raw_Bscan = np.loadtxt(data_path, delimiter=' ')
    print("Bscan shape:", Raw_Bscan.shape)
    print('')

    #* test
    #time_corrected_Bscan = time_correction(Raw_Bscan)


    #* 1. Bandpass filtering
    print('Applying bandpass filter')
    filtered_Bscan = np.zeros(Raw_Bscan.shape)
    for i in tqdm(range(Raw_Bscan.shape[1]), desc='Applying bandpass filter'):
                filtered_Bscan[:, i] = bandpass_filter(Raw_Bscan[:, i], 250e6, 750e6, 5)
    np.savetxt(dir_1 + '/1_Bscan_filter.txt', filtered_Bscan, delimiter=' ')
    print('Finished bandpass filtering')
    print(' ')


    #* 2. Time-zero correction
    print('Applying time-zero correction')
    time_corrected_Bscan = time_correction(filtered_Bscan)
    np.savetxt(dir_2 + '/2_Bscan_time_correction.txt', time_corrected_Bscan, delimiter=' ')
    print('Finished time-zero correction')
    print(' ')


    #* 3. Background removal
    print('Applying background removal')
    average, background_removed_Bscan = background_removal(time_corrected_Bscan)
    np.savetxt(dir_3 + '/3_Bscan_background_removal.txt', background_removed_Bscan, delimiter=' ')
    print('Finished background removal')
    print(' ')



    #* 4. Gain function
    print('Applying gain function')
    #t_2D = np.expand_dims(np.linspace(0, background_removed_Bscan.shape[0] *sample_interval, background_removed_Bscan.shape[0]), axis=1)
    #gained_Bscan = background_removed_Bscan * t_2D ** 1.7
    gained_Bscan = gain(background_removed_Bscan, 3.4, 0.006, 500e6)
    #gained_Bscan = gained_Bscan / np.amax(gained_Bscan)
    np.savetxt(dir_4 + '/4_Bscan_gain.txt', gained_Bscan, delimiter=' ')
    print('Finished gain function')
    print(' ')

elif args.function_type == 'plot':
    #* 0. Prepare raw Bscan data
    Raw_Bscan = np.loadtxt(data_path, delimiter=' ')
    print("Bscan shape:", Raw_Bscan.shape)


    #* 1. Bandpass filtering
    filtered_Bscan = np.loadtxt(dir_1 + '/1_Bscan_filter.txt', delimiter=' ')

    #* 2. Time-zero correction
    time_corrected_Bscan = np.loadtxt(dir_2 + '/2_Bscan_time_correction.txt', delimiter=' ')

    #* 3. Background removal
    background_removed_Bscan = np.loadtxt(dir_3 + '/3_Bscan_background_removal.txt', delimiter=' ')

    #* 4. Gain function
    gained_Bscan = np.loadtxt(dir_4 + '/4_Bscan_gain.txt', delimiter=' ')

    print('Finished data loading')



plot_data = [Raw_Bscan, filtered_Bscan, time_corrected_Bscan, background_removed_Bscan, gained_Bscan]
title = ['Raw B-scan', 'Bandpass filter', 'Time-zero correction', 'Background removal', 'Gain function']
dir_list = [dir_0, dir_1, dir_2, dir_3, dir_4]


#* Plot
font_large = 20
font_medium = 18
font_small = 16

#* Plot single panel figure x
print('   ')
print('Plotting single panel figure x 5')
for i in range(len(plot_data)):
    fig = plt.figure(figsize=(18, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    #fig, ax = plt.subplots(1, 1, figsize=(18, 6), tight_layout=True)
    if i == 4:
        im = ax.imshow(plot_data[i], aspect='auto', cmap='seismic',
                extent=[0, plot_data[i].shape[1]*trace_interval, plot_data[i].shape[0]*sample_interval*1e9, 0],
                vmin=-np.amax(np.abs(plot_data[i]))/5, vmax=np.amax(np.abs(plot_data[i]))/5
                )
    else:
        im = ax.imshow(plot_data[i], aspect='auto', cmap='seismic',
                    extent=[0, plot_data[i].shape[1]*trace_interval, plot_data[i].shape[0]*sample_interval*1e9, 0],
                    vmin=-30, vmax=30
                    )
    ax.tick_params(axis='both', which='major', labelsize=font_small)
    ax.set_title(title[i], fontsize=font_large)

    ax.set_xlabel('Trace number', fontsize=font_medium)
    ax.set_ylabel('Time (ns)', fontsize=font_medium)

    delvider = axgrid1.make_axes_locatable(ax)
    cax = delvider.append_axes('right', size='3%', pad=0.1)
    plt.colorbar(im, cax=cax, orientation = 'vertical').set_label('Amplitude', fontsize=font_large)
    cax.tick_params(labelsize=font_small)

    plt.savefig(dir_list[i] + '/' + str(i) + '_' + title[i] + '.png', format='png', dpi=120)
    plt.savefig(dir_list[i] + '/' + str(i) + '_' + title[i] + '.pdf', format='pdf', dpi=600)
    print('Finished plotting', title[i])
    plt.close()


#* plot 5 panel figure
print('   ')
print('Plotting 5 panel figure')
fig, ax = plt.subplots(len(plot_data), 1, figsize=(18, 20), tight_layout=True, sharex=True)


for i in range(len(plot_data)):
    if i == 4:
        im = ax[i].imshow(plot_data[i], aspect='auto', cmap='seismic',
                extent=[0, plot_data[i].shape[1]*trace_interval, plot_data[i].shape[0]*sample_interval*1e9, 0],
                vmin=-np.amax(np.abs(plot_data[i]))/5, vmax=np.amax(np.abs(plot_data[i]))/5
                )
    else:
        im = ax[i].imshow(plot_data[i], aspect='auto', cmap='seismic',
                    extent=[0, plot_data[i].shape[1]*trace_interval, plot_data[i].shape[0]*sample_interval*1e9, 0],
                    vmin=-30, vmax=30
                    )
    ax[i].tick_params(axis='both', which='major', labelsize=font_small)
    ax[i].set_title(title[i], fontsize=font_large)

    delvider = axgrid1.make_axes_locatable(ax[i])
    cax = delvider.append_axes('right', size='3%', pad=0.1)
    plt.colorbar(im, cax=cax, orientation = 'vertical').set_label('Amplitude', fontsize=font_large)
    cax.tick_params(labelsize=font_small)

fig.supxlabel('Distance [m]', fontsize=font_medium)
fig.supylabel('Time (ns)', fontsize=font_medium)



#* save plot
plt.savefig(dir_5 + '/Merged.png', format='png', dpi=120)
plt.savefig(dir_5 + '/Merged.pdf', format='pdf', dpi=600)
print('Finished plotting 5 panel figure')

plt.show()