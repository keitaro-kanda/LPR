import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm
from scipy import signal



class processing_filtering:
    def __init__(self, Bscan_data, sample_interval=0.312500e-9):
        self.Bscan_data = Bscan_data # input is Bscan data array (2D), not file path
        self.filtered_Bscan = np.zeros(self.Bscan_data.shape)
        self.sample_interval = sample_interval


    def bandpass_filter(self, Ascandata): # Ascandata is 1D array
        #* 0. Prepare window function to bandpass filter of 250-750 MHz
        window = signal.windows.hamming(len(Ascandata))
        #window = signal.windows.hamming(len(Ascandata))

        #* 1. FFT
        fft_data = np.fft.fft(Ascandata * window)
        freq = np.linspace(0, 1/self.sample_interval, len(fft_data))
        #fft_data = np.fft.fft(Ascandata)
        #freq = np.linspace(0, 1/self.sample_interval, len(fft_data))


        #* Filter specifications
        lowcut = 250e6 # [Hz]
        highcut = 750e6 # [Hz]
        fs = 1/self.sample_interval  # Sampling frequency

        #* Design the Butterworth band-pass filter
        order = 4
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = signal.butter(order, [low, high], btype='band')

        #* Apply bandpass filter
        filtered_data = signal.filtfilt(b, a, Ascandata)
        return filtered_data
        """
        #* 2. Bandpass filter
        low_stop = 150e6 # [Hz]
        low_bandpass = 250.0e6 # [Hz]
        high_bandpass = 750.0e6 # [Hz]
        high_stop = 850e6 # [Hz]

        #* Design FIR (finite impulse response) filter
        N = 2048
        FIR = signal.firwin(N, [low_stop, low_bandpass, high_bandpass, high_stop], pass_zero=False, fs=1/self.sample_interval)

        #* Apply FIR filter
        fft_data = np.fft.fft(Ascandata)
        fft_data = np.fft.fftshift(fft_data)
        filered_data = np.convolve(fft_data, FIR, mode='same')
        filterd_data = np.fft.ifftshift(filered_data)

        return filterd_data

        #* 3. Inverse FFT
        #filtered_data = np.fft.ifft(fft_data).real
        #return filtered_data
        """


    def apply_bandpass_filter(self):
        for i in tqdm(range(self.Bscan_data.shape[1]), desc='Applying bandpass filter'):
            self.filtered_Bscan[:, i] = self.bandpass_filter(self.Bscan_data[:, i])
        return self.filtered_Bscan


    def plot(self):
        fig = plt.figure(figsize=(12, 7), tight_layout=True)
        ax = fig.add_subplot(111)
        plt.imshow(self.filtered_Bscan, aspect='auto', cmap='seismic',
                    extent=[0, self.filtered_Bscan.shape[1], self.filtered_Bscan.shape[0]*self.sample_interval*1e9, 0],
                    vmin=-100, vmax=100
                    )
        ax.set_xlabel('Trace number', fontsize=18)
        ax.set_ylabel('Time (ns)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)

        #title = os.path.splitext(os.path.basename(Ascans_file_path))[0]
        #plt.title(title, fontsize=20)


        #* plot colorbar
        delvider = axgrid1.make_axes_locatable(ax)
        cax = delvider.append_axes('right', size='5%', pad=0.1)
        plt.colorbar(cax=cax).set_label('Amplitude', fontsize=18)
        cax.tick_params(labelsize=16)

        plt.show()

#* example useage
#Bscan_data = np.loadtxt('/Volumes/SSD_kanda/LPR/LPR_2B/Raw_Bscan/Bscan.txt', delimiter=' ')
#processer = processing_filtering(Bscan_data)
#processer.apply_bandpass_filter()
#processer.plot()


"""
#* Bandpass filter
def bandpass(Ascandata):
    #* 1. FFT
    fft_data = np.fft.fft(Ascandata)
    freq = np.linspace(0, 1/sample_interval, len(fft_data))

    #* 2. Bandpass filter
    low_freq = 250.0e6 # [Hz]
    high_freq = 750.0e6 # [Hz]
    fft_data[(freq < low_freq)] = 0
    fft_data[(freq > high_freq)] = 0

    #* 3. Inverse FFT
    filtered_data = np.fft.ifft(fft_data).real

#* Apply bandpass filter
filterd_Bscan = np.zeros(Bscan_data.shape)
for i in tqdm(range(Bscan_data.shape[1]), desc='Applying bandpass filter'):
    filterd_Bscan[:, i] = bandpass(Bscan_data[:, i])


#* plot
fig = plt.figure(figsize=(12, 7), tight_layout=True)
ax = fig.add_subplot(111)
plt.imshow(filterd_Bscan, aspect='auto', cmap='seismic',
            extent=[0, filterd_Bscan.shape[1], filterd_Bscan.shape[0]*sample_interval, 0],
            vmin=-100, vmax=100
            )
ax.set_xlabel('Trace number', fontsize=18)
ax.set_ylabel('Time (ns)', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)

#title = os.path.splitext(os.path.basename(Ascans_file_path))[0]
#plt.title(title, fontsize=20)


#* plot colorbar
delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax).set_label('Amplitude', fontsize=18)
cax.tick_params(labelsize=16)

plt.show()
"""