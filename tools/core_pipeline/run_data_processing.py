import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm
from scipy import signal
from natsort import natsorted


#* Get input parameters
resampled_data_dir = input('Input data folder path (Resampled_Data>txt directory): ').strip()
if not os.path.exists(resampled_data_dir):
    raise ValueError('Data folder path does not exist. Please check the path and try again.')

# path_type = input('Input channel name (2A, 2B): ').strip()
# #* Define data folder path
# if path_type == '2A':
#     resampled_data_dir = '/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2A/Resampled_Data/txt'
# elif path_type == '2B':
#     resampled_data_dir = '/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Resampled_Data/txt'
# if not os.path.exists(resampled_data_dir):
#         print('Error: Data path does not exist:', resampled_data_dir)
#         exit(1)

print('Select function type (calc or plot):')
function_type = input().strip().lower()
if function_type not in ['calc', 'plot']:
    print('Error: Function type must be calc or plot')
    exit(1)
else:
    print('\nAvailable processing steps:')
    print('0: Raw data integration')
    print('1: Bandpass filter')
    print('2: Time-zero correction')
    print('3: Background removal')
    print('4: Gain function')
    print('5: Terrain correction')
    print('\nEnter processing order (comma-separated, e.g., 0,1,2,3,4,5):')
    processing_order_input = input().strip()
    try:
        processing_order = [int(x.strip()) for x in processing_order_input.split(',')]
        if any(step < 0 or step > 5 for step in processing_order):
            print('Error: Processing steps must be between 0 and 5')
            exit(1)
        if 0 not in processing_order:
            print('Error: Step 0 (Raw data integration) is required')
            exit(1)
    except ValueError:
        print('Error: Invalid processing order format')
        exit(1)

#* Get processing order
# if function_type == 'calc':
#     print('\nAvailable processing steps:')
#     print('0: Raw data integration')
#     print('1: Bandpass filter')
#     print('2: Time-zero correction')
#     print('3: Background removal')
#     print('4: Gain function')
#     print('5: Terrain correction')
#     print('\nEnter processing order (comma-separated, e.g., 0,1,2,3,4,5):')
#     processing_order_input = input().strip()
#     try:
#         processing_order = [int(x.strip()) for x in processing_order_input.split(',')]
#         if any(step < 0 or step > 5 for step in processing_order):
#             print('Error: Processing steps must be between 0 and 5')
#             exit(1)
#         if 0 not in processing_order:
#             print('Error: Step 0 (Raw data integration) is required')
#             exit(1)
#     except ValueError:
#         print('Error: Invalid processing order format')
#         exit(1)
# else:
#     processing_order = [0, 1, 2, 3, 4, 5]  # Default order for plot mode





#* Define output folder path with processing order suffix
processing_order_str = '_'.join(map(str, processing_order))
output_base_dir = os.path.dirname(os.path.dirname(resampled_data_dir)) + '/Processed_Data'
output_dir = os.path.join(output_base_dir, f'order_{processing_order_str}')
os.makedirs(output_dir, exist_ok=True)

#* Create directories based on processing order
dir_dict = {}
step_names = {
    0: 'Raw_data',
    1: 'Bandpass_filter', 
    2: 'Time_zero_correction',
    3: 'Background_removal',
    4: 'Gain_function',
    5: 'Terrain_correction'
}

for i, step in enumerate(processing_order):
    dir_name = f'{step}_{step_names[step]}'
    dir_path = os.path.join(output_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    dir_dict[step] = dir_path

print('Output dir:', output_dir)
print('Processing order:', processing_order)
print('   ')



#* Define parameters
sample_interval = 0.312500e-9# [s]
fs = 1/sample_interval  # Sampling frequency
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]
epsilon_0 = 8.854187817e-12  # [F/m]
c = 299792458  # [m/s]
reciever_time_delay = 28.203e-9  # [s], [Su et al., 2014]
epsilon_r = 4.5 # Relative permittivity of the medium [Zhang et al., 2025, ApJL]



#* Define functions

#* Integrate resampled data
def integrate_resampled_data():
    Raw_Bscan = np.zeros((0, 0))  # Placeholder for raw B-scan data
    for file in tqdm(natsorted(os.listdir(resampled_data_dir)), desc='Integrating resampled data'):
        if file.startswith('.'):
            continue

        # Path to the resampled data file
        file_path = os.path.join(resampled_data_dir, file)
        # Load resampled data
        try:
            # まずはShift-JISで読み込みを試行
            resampled_data = np.loadtxt(file_path, delimiter=' ', encoding='shift-jis')
        except UnicodeDecodeError:
            # Shift-JISで失敗した場合、UTF-16で再試行
            try:
                resampled_data = np.loadtxt(file_path, delimiter=' ', encoding='utf-16')
            except Exception as e:
                # どちらのエンコーディングでも読み込めなかった場合
                print(f"\nエラー: ファイル '{file}' の読み込みに失敗しました。")
                print(f"詳細: {e}")
                # エラーが発生した場合はプログラムを停止
                raise
        except Exception as e:
            # その他の予期せぬエラー
            print(f"\n予期せぬエラーがファイル '{file}' で発生しました。")
            print(f"詳細: {e}")
            raise
        # Integrate data
        if Raw_Bscan.size == 0:
            Raw_Bscan = resampled_data
        else:
            Raw_Bscan = np.hstack((Raw_Bscan, resampled_data))
    return Raw_Bscan



#* Bandpass filter
def bandpass_filter(input_data, lowcut, highcut, order): # Ascandata is 1D array
    #* Filter specifications
    #* Design the Butterworth band-pass filter using SOS format
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    sos = signal.butter(order, [low, high], btype='band', output='sos')

    #* Apply bandpass filter using sosfiltfilt for better numerical stability
    filtered_data = signal.sosfiltfilt(sos, input_data)

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
def background_removal(data, output_dir=None):
    background_data = np.mean(data, axis=1)
    # save background data
    np.savetxt(output_dir + '/background_data.txt', background_data)

    # Background removal
    background_removed_data = np.zeros_like(data)
    for i in tqdm(range(data.shape[1]), desc='Subtracting background'):
        background_removed_data[:, i] =  data[:, i] - background_data

    # Plot background data
    if output_dir is not None:
        fig, ax = plt.subplots(figsize=(6, 10), tight_layout=True)
        time = np.arange(0, data.shape[0]*sample_interval/1e-9, sample_interval/1e-9)
        # ax[0].plot(background_data, time)

        # ax[0].set_xlabel('Amplitude', fontsize=20)
        # ax[0].set_ylabel('Time [ns]', fontsize=20)
        # ax[0].tick_params(labelsize=18)
        # ax[0].grid(which='major', axis='both', linestyle='-.')
        # ax[0].invert_yaxis()
        # ax[0].set_ylim(np.max(time), 0)

        ax.plot(np.log(np.abs(background_data)), time)

        ax.set_xlabel('Log amplitude', fontsize=20)
        ax.set_ylabel('Time [ns]', fontsize=20)
        ax.tick_params(labelsize=18)
        ax.grid(which='major', axis='both', linestyle='-.')
        ax.invert_yaxis()
        ax.set_ylim(np.max(time), 0)

        plt.savefig(output_dir + '/Background_data.png', format='png', dpi=120)
        plt.savefig(output_dir + '/Background_data.pdf', format='pdf', dpi=300)
        plt.close()
        print('Plot of background data is successfully saved.')
        print(' ')

    #* Plot background removed average data
    if output_dir is not None:
        fig, ax = plt.subplots(figsize=(6, 10), tight_layout=True)
        time = np.arange(0, data.shape[0]*sample_interval/1e-9, sample_interval/1e-9)
        background_removed_data_log = np.log(np.abs(background_removed_data))
        background_removed_log_ave = np.mean(background_removed_data_log, axis=1)
        print('Background removed average:', background_removed_log_ave)
        print('Background removed average shape: ', background_removed_log_ave.shape)
        background_removed_log_std = np.std(background_removed_data_log, axis=1)
        print('Background removed average standard deviation:', background_removed_log_std)
        print('Background removed average standard deviation shape: ', background_removed_log_std.shape)
        # ax[0].plot(background_removed_ave, time)
        # ax[0].fill_betweenx(time, background_removed_ave - background_removed_std,
        #                     background_removed_ave + background_removed_std, alpha=0.6)

        # ax[0].set_xlabel('Amplitude', fontsize=20)
        # ax[0].set_ylabel('Time [ns]', fontsize=20)
        # ax[0].tick_params(labelsize=18)
        # ax[0].grid(which='major', axis='both', linestyle='-.')
        # ax[0].invert_yaxis()
        # ax[0].set_ylim(np.max(time), 0)

        ax.plot(background_removed_log_ave, time)
        ax.fill_betweenx(time, background_removed_log_ave - background_removed_log_std,
                            background_removed_log_ave + background_removed_log_std, alpha=0.6)

        ax.set_xlabel('Log amplitude', fontsize=20)
        ax.set_ylabel('Time [ns]', fontsize=20)
        ax.tick_params(labelsize=18)
        ax.grid(which='major', axis='both', linestyle='-.')
        ax.invert_yaxis()
        ax.set_ylim(np.max(time), 0)

        plt.savefig(output_dir + '/Background_removed_data.png', format='png', dpi=120)
        plt.savefig(output_dir + '/Background_removed_data.pdf', format='pdf', dpi=300)
        plt.close()
        print('Plot of background removed data is successfully saved.')
        print(' ')

    return background_data,  background_removed_data



def gain(data, er, tan_delta, freq, output_dir=None):
    t_2D = np.expand_dims(np.arange(0, data.shape[0]*sample_interval, sample_interval), axis=1)
    gain_func = t_2D**2 * c**2 / (4 * er) * np.exp(np.pi * t_2D * freq * np.sqrt(er * epsilon_0)* tan_delta)

    #* Plot gain function
    if output_dir is not None:
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

        plt.savefig(output_dir + '/Gain_function.png', format='png', dpi=120)
        plt.savefig(output_dir + '/Gain_function.pdf', format='pdf', dpi=300)
        plt.close()
        print('Plot of gain function is successfully saved.')
        print(' ')

    output = data * gain_func

    if output_dir is not None:
        background_gained = np.mean(output, axis=1)
        fig, ax = plt.subplots(1, 2, figsize=(12, 8), tight_layout=True)
        time = np.arange(0, data.shape[0]*sample_interval/1e-9, sample_interval/1e-9)
        ax[0].plot(background_gained, time)

        ax[0].set_xlabel('Amplitude', fontsize=20)
        ax[0].set_ylabel('Time [ns]', fontsize=20)
        ax[0].tick_params(labelsize=18)
        ax[0].grid(which='major', axis='both', linestyle='-.')
        ax[0].invert_yaxis()
        ax[0].set_ylim(np.max(time), 0)

        ax[1].plot(background_gained, time)

        ax[1].set_xlabel('Amplitude', fontsize=20)
        ax[1].set_ylabel('Time [ns]', fontsize=20)
        ax[1].tick_params(labelsize=18)
        ax[1].grid(which='major', axis='both', linestyle='-.')
        ax[1].invert_yaxis()
        ax[1].set_ylim(100, 0)

        plt.savefig(output_dir + '/Background_gained.png', format='png', dpi=120)
        plt.savefig(output_dir + '/Background_gained.pdf', format='pdf', dpi=300)
        plt.close()
        print('Plot of background data is successfully saved.')
        print(' ')

    return output



def terrain_correction(data, z_profile):
    """
    Apply terrain correction to the B-scan data based on the given z-profile.
    The z-profile is a 1D array representing the terrain elevation at each trace.
    z=0 is set to the ground level of the starting point
    """
    #* Prepare empty array for corrected data
    z_max = np.max(z_profile) # Maximum elevation in the z-profile
    z_min = np.min(z_profile) # Minimum elevation in the z-profile
    print('z_max: ', z_max, 'z_min: ', z_min)
    t_expand_min = np.abs(int(z_max / c / sample_interval)) # index number, don't use subsurface velocity
    t_expand_max = np.abs(int(z_min / c / sample_interval)) # index number
    corrected_data = np.zeros((data.shape[0] + t_expand_min + t_expand_max, data.shape[1]))
    corrected_data[:, :] = np.nan  # Initialize with NaN to avoid artifacts
    print('Expanded time range: ', -t_expand_min*sample_interval/1e-9, 'to', (data.shape[0]+t_expand_max)*sample_interval/1e-9, ' ns')
    print('Data shape after terrain correction: ', corrected_data.shape)

    #* Apply terrain correction
    for i in tqdm(range(data.shape[1]), desc='Applying terrain correction'):
        # Calculate the depth based on the z-profile
        equivalent_time = z_profile[i] / c  # Convert depth to time [s]
        # Apply the correction to each trace
        start_index = int(equivalent_time / sample_interval)
        start_row = np.abs(t_expand_min) - start_index
        end_row = start_row + data.shape[0]
        corrected_data[start_row:end_row, i] = data[:, i]
    return corrected_data, -t_expand_min*sample_interval, (data.shape[0] + t_expand_max)*sample_interval #, data, -OO [s], (data.shape[0] + OO) [s]


#* Processing pipeline
if function_type == 'calc':
    #* Initialize data storage
    processed_data = {}
    
    #* Execute processing steps in the specified order
    for i, step in enumerate(processing_order):
        if step == 0:  # Raw data integration
            print('0. Integrating raw data')
            processed_data[0] = integrate_resampled_data()
            print('Integrated resampled data shape:', processed_data[0].shape)
            #* time delayを考慮して、データをスライス
            processed_data[0] = processed_data[0][int(reciever_time_delay/sample_interval):, :]
            print("Bscan shape (after data slice):", processed_data[0].shape)
            print('')
            np.savetxt(dir_dict[0] + '/0_Raw_Bscan.txt', processed_data[0], delimiter=' ')
            #* Save B-scan shape as a text file
            with open(dir_dict[0] + '/Bscan_shape.txt', 'w') as f:
                f.write(f'Bscan shape: {processed_data[0].shape[0]} x {processed_data[0].shape[1]}\n')
            print('Finished raw data integration')
            print(' ')
        
        elif step == 1:  # Bandpass filtering
            print('1. Applying bandpass filter')
            prev_step = processing_order[i-1] if i > 0 else 0
            input_data = processed_data[prev_step]
            processed_data[1] = np.zeros(input_data.shape)
            for j in tqdm(range(input_data.shape[1]), desc='Applying bandpass filter'):
                processed_data[1][:, j] = bandpass_filter(input_data[:, j], 250e6, 750e6, 5)
            np.savetxt(dir_dict[1] + '/1_Bscan_filter.txt', processed_data[1], delimiter=' ')
            print('Finished bandpass filtering')
            print(' ')
        
        elif step == 2:  # Time-zero correction
            print('2. Applying time-zero correction')
            prev_step = processing_order[i-1] if i > 0 else 0
            input_data = processed_data[prev_step]
            processed_data[2] = time_correction(input_data)
            np.savetxt(dir_dict[2] + '/2_Bscan_time_correction.txt', processed_data[2], delimiter=' ')
            print('Finished time-zero correction')
            print(' ')
        
        elif step == 3:  # Background removal
            print('3. Applying background removal')
            prev_step = processing_order[i-1] if i > 0 else 0
            input_data = processed_data[prev_step]
            average, processed_data[3] = background_removal(input_data, dir_dict[3])
            np.savetxt(dir_dict[3] + '/3_Bscan_background_removal.txt', processed_data[3], delimiter=' ')
            print('Finished background removal')
            print(' ')
        
        elif step == 4:  # Gain function
            print('4. Applying gain function')
            prev_step = processing_order[i-1] if i > 0 else 0
            input_data = processed_data[prev_step]
            processed_data[4] = gain(input_data, 3.4, 0.006, 500e6, dir_dict[4])
            np.savetxt(dir_dict[4] + '/4_Bscan_gain.txt', processed_data[4], delimiter=' ')
            print('Finished gain function')
            print(' ')
        
        elif step == 5:  # Terrain correction
            print('5. Applying terrain correction')
            prev_step = processing_order[i-1] if i > 0 else 0
            input_data = processed_data[prev_step]
            #* Load z-profile data
            position_profile_path = "/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Resampled_Data/position_plot/total_position.txt"
            z_profile = np.loadtxt(position_profile_path, delimiter=' ', skiprows=1)[:, 2]
            processed_data[5], time_min, time_max = terrain_correction(input_data, z_profile)
            np.savetxt(dir_dict[5] + '/5_Terrain_correction.txt', processed_data[5], delimiter=' ')
            print('Finished terrain correction')
            print(' ')

    print('----- Finished data processing -----')
    print(' ')



elif function_type == 'plot':
    print('----- Start loading data -----')
    
    #* Load data based on processing order
    processed_data = {}
    for step in processing_order:
        if step == 0:
            processed_data[0] = np.loadtxt(dir_dict[0] + '/0_Raw_Bscan.txt', delimiter=' ')
            if not os.path.exists(dir_dict[0] + '/0_Raw_Bscan.txt'):
                raise FileNotFoundError(f'Raw B-scan data not found in {dir_dict[0]}/0_Raw_Bscan.txt')
                exit(1)
        elif step == 1:
            processed_data[1] = np.loadtxt(dir_dict[1] + '/1_Bscan_filter.txt', delimiter=' ')
        elif step == 2:
            processed_data[2] = np.loadtxt(dir_dict[2] + '/2_Bscan_time_correction.txt', delimiter=' ')
        elif step == 3:
            processed_data[3] = np.loadtxt(dir_dict[3] + '/3_Bscan_background_removal.txt', delimiter=' ')
        elif step == 4:
            processed_data[4] = np.loadtxt(dir_dict[4] + '/4_Bscan_gain.txt', delimiter=' ')
        elif step == 5:
            processed_data[5] = np.loadtxt(dir_dict[5] + '/5_Terrain_correction.txt', delimiter=' ')
    
    #* Special handling for terrain correction time range
    if 5 in processed_data:
        position_profile_path = "/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Resampled_Data/position_plot/total_position.txt"
        z_profile = np.loadtxt(position_profile_path, delimiter=' ', skiprows=1)[:, 2]
        time_min = - np.amax(z_profile) / c  # Convert depth to time [s]
        time_max = processed_data[0].shape[0] * sample_interval - np.amin(z_profile) / c  # Convert depth to time [s]
    
    print("Finished data loading")
    print(' ')

#* Prepare plot data in processing order
plot_data = []
title = []
dir_list = []
step_names = {
    0: 'Raw B-scan',
    1: 'Bandpass filter', 
    2: 'Time-zero correction',
    3: 'Background removal',
    4: 'Gain function',
    5: 'Terrain correction'
}

for step in processing_order:
    if step in processed_data:
        plot_data.append(processed_data[step])
        title.append(step_names[step])
        dir_list.append(dir_dict[step])


#* Plot
font_large = 20
font_medium = 18
font_small = 16

#* Plot single panel figure x
print('----- Start plotting -----')
print(f'Plotting single panel figure x {len(plot_data)}')
for i in range(len(plot_data)):
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(111)
    
    # Get the processing step for this plot
    step = processing_order[i]
    
    if step == 4:  # Gain function
        im = ax.imshow(plot_data[i], aspect='auto', cmap='seismic',
                extent=[0, plot_data[i].shape[1]*trace_interval, plot_data[i].shape[0]*sample_interval*1e9, 0],
                vmin=-np.amax(np.abs(plot_data[i]))/10, vmax=np.amax(np.abs(plot_data[i]))/10
                )
    elif step == 5:  # Terrain correction
        im = ax.imshow(plot_data[i], aspect='auto', cmap='seismic',
                extent=[0, plot_data[i].shape[1]*trace_interval, time_max*1e9, time_min*1e9],
                vmin=-np.nanmax(np.abs(plot_data[i]))/10, vmax=np.nanmax(np.abs(plot_data[i]))/10
                )
        ax.set_yticks(np.arange(0, plot_data[i].shape[0] * sample_interval / 1e-9, 100))
    else:
        im = ax.imshow(plot_data[i], aspect='auto', cmap='seismic',
                    extent=[0, plot_data[i].shape[1]*trace_interval, plot_data[i].shape[0]*sample_interval*1e9, 0],
                    vmin=-10, vmax=10
                    )
    ax.tick_params(axis='both', which='major', labelsize=font_small)
    #ax.set_title(title[i], fontsize=font_large)

    ax.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax.set_ylabel('Time [ns]', fontsize=font_medium)

    #* Add the second Y-axis
    ax2 = ax.twinx()
    # Get the range of the original Y-axis (time)
    t_min, t_max = ax.get_ylim()
    # Calculate the corresponding depth range
    # depth [m] = (time [ns] * 1e-9) * v_gpr [m/s] / 2
    depth_min = (t_min * 1e-9) * c / np.sqrt(epsilon_r) / 2
    depth_max = (t_max * 1e-9) * c / np.sqrt(epsilon_r) / 2
    # 新しいY軸に深さの範囲とラベルを設定
    ax2.set_ylim(depth_min, depth_max)
    ax2.set_ylabel(r'Depth [m] ($\varepsilon_r = 4.5$)', fontsize=font_medium)
    ax2.tick_params(axis='y', which='major', labelsize=font_small)

    # #* Add colorbar
    # delvider = axgrid1.make_axes_locatable(ax)
    # cax = delvider.append_axes('right', size='3%', pad=0.1)
    # cax.set_position(cax.get_position().translated(0.08, 0)) # 右に少しずらす
    # plt.colorbar(im, cax=cax, orientation = 'vertical').set_label('Amplitude', fontsize=font_large)
    # cax.tick_params(labelsize=font_small)

    # * Adjust layout for colorbar layout
    fig.subplots_adjust(bottom=0.18, right=0.9)
    ### カラーバーを右下に横向きで追加 ###
    # Figureのサイズを基準とした位置とサイズ [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.65, 0.05, 0.2, 0.05]) # [x, y, 幅, 高さ]
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=font_small)
    ### ここまで ###

    #* Saving
    plt.savefig(dir_list[i] + '/' + str(step) + '_' + title[i] + '.png', format='png', dpi=120)
    plt.savefig(dir_list[i] + '/' + str(step) + '_' + title[i] + '.pdf', format='pdf', dpi=600)
    print('Finished plotting', title[i])
    plt.close()


# #* plot 5 panel figure
# print('   ')
# print('Plotting 5 panel figure')
# fig, ax = plt.subplots(len(plot_data), 1, figsize=(18, 20), tight_layout=False, sharex=True)
# ax2_list = []

# for i in range(len(plot_data)):
#     if i == 4:
#         im = ax[i].imshow(plot_data[i], aspect='auto', cmap='viridis',
#                 extent=[0, plot_data[i].shape[1]*trace_interval, plot_data[i].shape[0]*sample_interval*1e9, 0],
#                 vmin=-np.amax(np.abs(plot_data[i]))/15, vmax=np.amax(np.abs(plot_data[i]))/15
#                 )
#     else:
#         im = ax[i].imshow(plot_data[i], aspect='auto', cmap='viridis',
#                     extent=[0, plot_data[i].shape[1]*trace_interval, plot_data[i].shape[0]*sample_interval*1e9, 0],
#                     vmin=-10, vmax=10
#                     )
#     ax[i].tick_params(axis='both', which='major', labelsize=font_small)
#     ax[i].set_title(title[i], fontsize=font_large)

#     current_ax2 = ax[i].twinx()
#     ax2_list.append(current_ax2)

#     t_min, t_max = ax[i].get_ylim()
#     depth_min = (t_min * 1e-9) * c / np.sqrt(epsilon_r) / 2
#     depth_max = (t_max * 1e-9) * c / np.sqrt(epsilon_r) / 2
#     # 新しいY軸に深さの範囲とラベルを設定
#     current_ax2.set_ylim(depth_min, depth_max)
#     current_ax2.tick_params(axis='y', which='major', labelsize=font_small)


#     ### ★★★ 修正点: 各サブプロットの下にカラーバーを配置 ★★★ ###
#     # fig.add_axes() の代わりに make_axes_locatable を使用します
#     divider = axgrid1.make_axes_locatable(ax[i])
#     # ax[i] の下 ("bottom") に、高さが5%の新しい軸 cax を作成します
#     # pad はプロットとカラーバーの間の余白です
#     cax = divider.append_axes("bottom", size="3%", pad=0.5)
#     cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
#     cbar.ax.tick_params(labelsize=font_small)
#     ### ここまで ###

# fig.supxlabel('Moving distance [m]', fontsize=font_medium)
# fig.supylabel('Time [ns]', fontsize=font_medium)



# #* save plot
# plt.savefig(dir_5 + '/Merged.png', format='png', dpi=120)
# plt.savefig(dir_5 + '/Merged.pdf', format='pdf', dpi=600)
# print('Finished plotting 5 panel figure')

# plt.show()