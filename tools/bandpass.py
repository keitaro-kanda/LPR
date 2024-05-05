import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm


Bscan_data = np.loadtxt('Ascans/LPR_2B_echo_data.txt', skiprows=1, delimiter=' ')

#sample_interval = 0.312500e-9  # [s]


class processing_filtering:
    def __init__(self, Bscan_data, sample_interval=0.312500e-9):
        self.Bscan_data = Bscan_data # input is Bscan data array, not file path
        self.filtered_Bscan = np.zeros(self.Bscan_data.shape)
        self.sample_interval = sample_interval


    def bandpass_filter(self, Ascandata):

        #* 1. FFT
        fft_data = np.fft.fft(Ascandata)
        freq = np.linspace(0, 1/self.sample_interval, len(fft_data))

        #* 2. Bandpass filter
        low_freq = 250.0e6 # [Hz]
        high_freq = 750.0e6 # [Hz]
        fft_data[(freq < low_freq)] = 0
        fft_data[(freq > high_freq)] = 0

        #* 3. Inverse FFT
        filtered_data = np.fft.ifft(fft_data).real
        return filtered_data


    def apply_bandpass_filter(self):
        for i in tqdm(range(self.Bscan_data.shape[1]), desc='Applying bandpass filter'):
            self.filtered_Bscan[:, i] = self.bandpass_filter(self.Bscan_data[:, i])
        return self.filtered_Bscan


    def plot(self):
        fig = plt.figure(figsize=(12, 7), tight_layout=True)
        ax = fig.add_subplot(111)
        plt.imshow(self.filtered_Bscan, aspect='auto', cmap='seismic',
                    extent=[0, self.filtered_Bscan.shape[1], self.filtered_Bscan.shape[0]*self.sample_interval, 0],
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