import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from bandpass import processing_filtering
from remove_background import processing_background_removal
from time_zero_correction import proccessing_time_zero_correction
import argparse
from tqdm import tqdm
from scipy import signal

#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='run_data_processing.py',
    description='Run bandpass filtering, time-zero correction, and background removal on resampled Bscan data',
    epilog='End of help message',
    usage='python tools/resampling.py [path_type] [function_type]',
)
parser.add_argument('path_type', choices = ['local', 'SSD'], help='Choose the path type')
parser.add_argument('function_type', choices=['calc', 'plot'], help='Choose the function type')
args = parser.parse_args()


#* Define data folder path
if args.path_type == 'local' or args.path_type == 'test':
    data_path = 'LPR_2B/Resampled_ECHO/Bscan.txt'
elif args.path_type == 'SSD':
    data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Resampled_ECHO/Bscan.txt'
    print('Data path:', data_path)



#* Define output folder path
output_dir = os.path.join(os.path.dirname(os.path.dirname(data_path)), 'Processed_Bscan')
os.makedirs(output_dir, exist_ok=True)
png_dir = os.path.join(output_dir, 'png')
os.makedirs(png_dir, exist_ok=True)
pdf_dir = os.path.join(output_dir, 'pdf')
os.makedirs(pdf_dir, exist_ok=True)
txt_dir = os.path.join(output_dir, 'txt')
os.makedirs(txt_dir, exist_ok=True)
print('Output dir:', output_dir)
print('   ')



#* Define parameters
sample_interval = 0.312500e-9# [s]
fs = 1/sample_interval  # Sampling frequency
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]


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
            peak_index = np.where(data[:, i]>0.5)[0][0] # index number, not time
            peak.append(peak_index)
    time_zero_point = np.median(peak)
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
def horizontal_high_pass_filter(data, time_delay, cuttoff, order):
    """
    normal_cutoff = cuttoff / (0.5 * fs)

    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = np.zeros(data.shape)
    for i in tqdm(range(time_delay, data.shape[0]), desc='Applying horizontal high pass filter'):
        y[i, :] = signal.filtfilt(b, a, data[i, :])
    return y
    """
    background_data = np.mean(data, axis=1)
    background_removed_data = np.zeros_like(data)
    for i in tqdm(range(data.shape[1]), desc='Subtracting background'):
        background_removed_data[:, i] =  data[:, i] - background_data
    return background_data,  background_removed_data



#* Processing pipeline
if args.function_type == 'calc':
    #* 0. Prepare raw Bscan data
    Raw_Bscan = np.loadtxt(data_path, delimiter=' ')
    print("Bscan shape:", Raw_Bscan.shape)

    #* test
    #time_corrected_Bscan = time_correction(Raw_Bscan)


    #* 1. Bandpass filtering
    filtered_Bscan = np.zeros(Raw_Bscan.shape)
    for i in tqdm(range(Raw_Bscan.shape[1]), desc='Applying bandpass filter'):
                filtered_Bscan[:, i] = bandpass_filter(Raw_Bscan[:, i], 250e6, 750e6, 5)
    np.savetxt(txt_dir + '/1_filtered_Bscan.txt', filtered_Bscan, delimiter=' ')
    print('Finished bandpass filtering')


    #* 2. Time-zero correction
    time_corrected_Bscan = time_correction(filtered_Bscan)
    np.savetxt(txt_dir + '/2_aligned_Bscan.txt', time_corrected_Bscan, delimiter=' ')
    print('Finished time-zero correction')


    #* 3. Background removal
    average, background_removed_Bscan = horizontal_high_pass_filter(time_corrected_Bscan, 0, 9.6e6, 5)
    np.savetxt(txt_dir + '/3_background_removed_Bscan.txt', background_removed_Bscan, delimiter=' ')
    print('Finished background removal')


    #* 4. Gain function
    t_2D = np.expand_dims(np.linspace(0, background_removed_Bscan.shape[0] *sample_interval, background_removed_Bscan.shape[0]), axis=1)
    gained_Bscan = background_removed_Bscan * t_2D ** 1.7
    gained_Bscan = gained_Bscan / np.amax(gained_Bscan)
    np.savetxt(txt_dir + '/gained_Bscan.txt', gained_Bscan, delimiter=' ')
    print('Finished gain correction')

elif args.function_type == 'plot':
    #* 0. Prepare raw Bscan data
    Raw_Bscan = np.loadtxt(data_path, delimiter=' ')
    print("Bscan shape:", Raw_Bscan.shape)


    #* 1. Bandpass filtering
    filtered_Bscan = np.loadtxt(txt_dir + '/1_filtered_Bscan.txt', delimiter=' ')

    #* 2. Time-zero correction
    time_corrected_Bscan = np.loadtxt(txt_dir + '/2_aligned_Bscan.txt', delimiter=' ')

    #* 3. Background removal
    background_removed_Bscan = np.loadtxt(txt_dir + '/3_background_removed_Bscan.txt', delimiter=' ')

    #* 4. Gain function
    gained_Bscan = np.loadtxt(txt_dir + '/gained_Bscan.txt', delimiter=' ')

    print('Finished data loading')

"""
if args.function_type == 'calc':
    filtered_Bscan = np.zeros(Raw_Bscan.shape)
    for i in tqdm(range(Raw_Bscan.shape[1]), desc='Applying bandpass filter'):
                filtered_Bscan[:, i] = bandpass_filter(Raw_Bscan[:, i], 250e6, 750e6, 5)
    np.savetxt(txt_dir + '/1_filtered_Bscan.txt', filtered_Bscan, delimiter=' ')
elif args.function_type == 'plot':
    filtered_Bscan = np.loadtxt(txt_dir + '/1_filtered_Bscan.txt', delimiter=' ')
else:
    print('Invalid function type')
    exit()
print('Finished bandpass filtering')
print(filtered_Bscan.shape)



#* process time zero correction
if args.function_type == 'calc':
    time_zero_correction = proccessing_time_zero_correction(filtered_Bscan)
    peak_time = time_zero_correction.find_peak_time()
    aligned_Bscan = time_zero_correction.align_peak_time()
    #aligned_Bscan = time_zero_correction.zero_corrections()
    np.savetxt(txt_dir + '/2_aligned_Bscan.txt', aligned_Bscan, delimiter=' ')
elif args.function_type == 'plot':
    aligned_Bscan = np.loadtxt(txt_dir + '/2_aligned_Bscan.txt', delimiter=' ')
print('Finished time-zero correction')
print(aligned_Bscan.shape)



#* process background removal
if args.function_type == 'calc':
    #background_removal = processing_background_removal(aligned_Bscan)
    #background_data, background_removed_Bscan = background_removal.subtract_background()

    #* Holizontal high pass filter
    def horizontal_high_pass_filter(data, time_delay, cuttoff, fs, order):
        normal_cutoff = cuttoff / (0.5 * fs)

        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        y = data
        for i in tqdm(range(time_delay, data.shape[0]), desc='Applying horizontal high pass filter'):
            y[i, :] = signal.filtfilt(b, a, data[i, :])
        return y

    background_removed_Bscan = horizontal_high_pass_filter(aligned_Bscan, 30, 9.6e6, 1/sample_interval, 5)
    np.savetxt(txt_dir + '/3_background_removed_Bscan.txt', background_removed_Bscan, delimiter=' ')
elif args.function_type == 'plot':
    background_removed_Bscan = np.loadtxt(txt_dir + '/3_background_removed_Bscan.txt', delimiter=' ')
print('Finished background removal')
print(background_removed_Bscan.shape)


#* Process gain function
if args.function_type == 'calc':
    t_2D = np.expand_dims(np.linspace(0, background_removed_Bscan.shape[0] *sample_interval, background_removed_Bscan.shape[0]), axis=1)
    gained_Bscan = background_removed_Bscan * t_2D ** 1.7
    #* Gain function from [Feng et al. (2023)]
    eps_r = 3.4
    loss_tangent = 0.006
    c = 3e8 # [m/s]
    v = c / np.sqrt(eps_r)
    wavelength = v / 500e6 # [m]
    r = t_2D * v / 2
    alpha = np.pi / wavelength * np.sqrt(eps_r) * loss_tangent
    gained_Bscan = background_removed_Bscan * r**2 * np.exp(2 * alpha * r)
    gained_Bscan = gained_Bscan / np.max(gained_Bscan)
    np.savetxt(txt_dir + '/gained_Bscan.txt', gained_Bscan, delimiter=' ')
elif args.function_type == 'plot':
    gained_Bscan = np.loadtxt(txt_dir + '/gained_Bscan.txt', delimiter=' ')
print('Finished gain function')


print('Finished all processing')
if args.function_type == 'calc':
    np.savetxt(txt_dir + '/4_processed_Bscan.txt', gained_Bscan, delimiter=' ')
elif args.function_type == 'plot':
    gained_Bscan = np.loadtxt(txt_dir + '/4_processed_Bscan.txt', delimiter=' ')
"""


plot_data = [Raw_Bscan, filtered_Bscan, time_corrected_Bscan, background_removed_Bscan, gained_Bscan]
title = ['Raw B-scan', 'Bandpass filtered B-scan', 'Time-zero corrected B-scan', 'Background removed B-scan', 'Gained B-scan']


#sample_interval_ns = sample_interval * 1e9
#trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]


#* Plot
font_large = 20
font_medium = 18
font_small = 16


#* Plot averaged background
"""
t = np.linspace(0, background_removed_Bscan.shape[0] *sample_interval, background_removed_Bscan.shape[0])
plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(t, background_data)

plt.title('Background noise', fontsize=font_large)
plt.xlabel('Time [ns]', fontsize=font_medium)
plt.ylabel('Amplitude', fontsize=font_medium)
plt.tick_params(axis='both', which='major', labelsize=font_small)

plt.savefig(output_dir + '/Background_noise.png')
plt.close()
"""


#* Plot gain function
"""
plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(t, t**1.7)

plt.title('Gain function', fontsize=font_large)
plt.xlabel('2-way travel time [ns]', fontsize=font_medium)
plt.ylabel('Gain', fontsize=font_medium)
plt.tick_params(axis='both', which='major', labelsize=font_small)
plt.yscale('log')

plt.savefig(output_dir + '/Gain_function.png')
plt.close()
"""


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
                vmin=-0.05, vmax=0.05
                )
    else:
        im = ax.imshow(plot_data[i], aspect='auto', cmap='seismic',
                    extent=[0, plot_data[i].shape[1]*trace_interval, plot_data[i].shape[0]*sample_interval*1e9, 0],
                    vmin=-15, vmax=15
                    )
    ax.tick_params(axis='both', which='major', labelsize=font_small)
    ax.set_title(title[i], fontsize=font_large)

    ax.set_xlabel('Trace number', fontsize=font_medium)
    ax.set_ylabel('Time (ns)', fontsize=font_medium)

    delvider = axgrid1.make_axes_locatable(ax)
    cax = delvider.append_axes('right', size='3%', pad=0.1)
    plt.colorbar(im, cax=cax, orientation = 'vertical').set_label('Amplitude', fontsize=font_large)
    cax.tick_params(labelsize=font_small)

    plt.savefig(png_dir + '/' + str(i+1) + '_' + title[i] + '.png', format='png', dpi=300)
    plt.savefig(pdf_dir + '/' + str(i+1) + '_' + title[i] + '.pdf', format='pdf', dpi=300)
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
                vmin=-0.05, vmax=0.05
                )
    else:
        im = ax[i].imshow(plot_data[i], aspect='auto', cmap='seismic',
                    extent=[0, plot_data[i].shape[1]*trace_interval, plot_data[i].shape[0]*sample_interval*1e9, 0],
                    vmin=-15, vmax=15
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
plt.savefig(png_dir + '/0_Processed_Bscan.png', format='png', dpi=300)
plt.savefig(pdf_dir + '/0_Processed_Bscan.pdf', format='pdf', dpi=300)
print('Finished plotting 5 panel figure')

plt.show()