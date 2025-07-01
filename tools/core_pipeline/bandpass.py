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


    def bandpass_filter(self, Ascandata, lowcut, highcut, order): # Ascandata is 1D array
        #* Filter specifications
        #lowcut = 250e6 # [Hz]
        #highcut = 750e6 # [Hz]
        fs = 1/self.sample_interval  # Sampling frequency

        #* Design the Butterworth band-pass filter
        #order = 4
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = signal.butter(order, [low, high], btype='band')

        #* Apply bandpass filter
        filtered_data = signal.filtfilt(b, a, Ascandata)

        return filtered_data


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