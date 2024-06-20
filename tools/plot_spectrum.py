import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
import argparse
from matplotlib.colors import LogNorm
from scipy.fft import fft
from scipy.signal import windows


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='plot_spectrum.py',
    description='Calculate and plot spectrum from ECHO data',
    epilog='End of help message',
    usage='python tools/plot_spectrum.py [path_type]',
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


#* FFT parameters
N = 2048
sample_interval = 0.312500 * 1e-9  # [s]


# Bandpass filter parameters
low_cutoff = 250e6  # [Hz]
high_cutoff = 750e6  # [Hz]


#* Loop through each ECHO data file
for ECHO_data in natsorted(os.listdir(data_folder_path)):
    if not ECHO_data.endswith('.txt'):
        continue
    if ECHO_data.startswith('._'):
        continue

    ECHO_data_path = os.path.join(data_folder_path, ECHO_data)
    sequence_id = ECHO_data.split('_')[-1].split('.')[0]
    data = np.loadtxt(ECHO_data_path, delimiter=' ')
    #print(data.shape)

    ECHO = data[8:, :]  # Skip header lines

    data_point_num, record_num = ECHO.shape
    #print(data_point_num, record_num)
    frequencies = np.fft.fftfreq(N, d=sample_interval)[:N//2] / 1e6  # [MHz]


    # Create a bandpass window function
    window = np.zeros(N//2)
    band = (frequencies >= low_cutoff) & (frequencies <= high_cutoff)
    window[band] = windows.hamming(np.sum(band))

    #* Initialize a matrix to store the spectrogram
    spectrogram = np.zeros((N//2, record_num))

    #* Calculate the FFT for each point in the scan
    for i in range(record_num):
        signal = ECHO[:, i]
        spectrum = fft(signal, n=N)[:N//2]
        spectrogram[:, i] = np.abs(spectrum)
        
        # Apply the window function (bandpass filter)
        #filtered_spectrum = spectrum * window

        # Take only the positive frequencies for plotting
        #spectrogram[:, i] = np.abs(filtered_spectrum)

    # Avoid log of zero by adding a small value
    #spectrogram[spectrogram == 0] = 1e-10

    #* Plot spectrogram
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(np.arange(record_num), frequencies, 10 * np.log10(spectrogram), shading='gouraud', norm=LogNorm())


    plt.ylabel('Frequency [MHz]')
    plt.xlabel('Record Count')
    plt.title(sequence_id)
    plt.colorbar(label='Intensity [dB]')

    #* Save plot
    output_path = os.path.join(output_dir, ECHO_data.replace('.txt', '_spectrogram.png'))
    #plt.savefig(output_path)
    plt.show()
    #plt.close()


"""
for ECHO_data in natsorted(os.listdir(data_folder_path)):
    #* Load only .txt files
    if not ECHO_data.endswith('.txt'):
        continue
    if ECHO_data.startswith('._'):
        continue

    ECHO_data_path = os.path.join(data_folder_path, ECHO_data)
    data = np.loadtxt(ECHO_data_path, delimiter=' ')

    ECHO = data[5:, :]


    #* Calculate spectrogram
    f, t, Sxx = signal.spectrogram(ECHO.flatten(), fs=1/sample_interval, nperseg=1024, noverlap=512, nfft=1024)
"""


"""
    #* FFT in each row and make 2D spectrum
    for i in tqdm(range(ECHO.shape[1]), desc=f'Processing {ECHO_data}...'):
        spectrum = np.fft.fft(ECHO[:, i])
        spectrum = np.abs(spectrum)

        if i == 0:
            spectrum_2D = spectrum[:, np.newaxis]
            plt.plot(spectrum)
            plt.show()
        else:
            spectrum_2D = np.hstack((spectrum_2D, spectrum[:, np.newaxis]))


    #* Plot spectrum
    plt.figure(figsize=(10, 12))
    plt.imshow(spectrum_2D, aspect='auto', cmap='rainbow',
                norm=LogNorm(vmin=1, vmax=1000))
    plt.colorbar()
    plt.xlabel('Record count')
    plt.ylabel('Frequency [Hz]')

    plt.show()
"""
