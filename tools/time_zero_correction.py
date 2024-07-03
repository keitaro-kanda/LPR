import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm




#* B-scanに含まれるそれぞれの波形のピーク時刻を揃える
#* 1. それぞれの波形のピーク時刻を求める
#* 2. それぞれのピーク時刻を揃える
#* 3. 揃えたピーク時刻に基づいて波形を切り出す

class proccessing_time_zero_correction:
    def __init__(self, Bscan_data, sample_interval=0.312500):
        self.Bscan_data = Bscan_data
        self.sample_interval = sample_interval
        self.peak = []
        self.aligned_Bscan = np.zeros(self.Bscan_data.shape)

    def zero_corrections(self):
        #data_2 = np.zeros((self.Bscan_data.shape))
        for i in tqdm(range(self.Bscan_data.shape[1])):
            idx = np.where((np.abs(self.Bscan_data[:, i])>0.5))[0][0]
            self.aligned_Bscan[:idx, i] = self.Bscan_data[idx:, i]
        return self.aligned_Bscan

    def find_peak_time(self):
        for i in tqdm(range(self.Bscan_data.shape[1]), desc='Finding peak time'):
            peak_index = np.argmax(self.Bscan_data[:, i]) # index number, not time
            self.peak.append(peak_index)
        return self.peak

    def align_peak_time(self):
        #* Align peak time at median peak time
        time_zero_point = np.median(self.peak)
        time_zero_record = self.peak.index(time_zero_point)
        print('Time zero point: ', time_zero_point, 'record number: ', time_zero_record)
        print('New time zero [s]', time_zero_point * self.sample_interval)
        for i in tqdm(range(self.Bscan_data.shape[1]), desc='Aligning peak time'):
            shift = int((time_zero_point - self.peak[i])) # shift index number
            #* シフトする分はゼロで埋める
            if shift > 0:
                self.aligned_Bscan[:, i] = np.concatenate([np.zeros(shift), self.Bscan_data[:len(self.Bscan_data)-shift, i]])
            elif shift < 0:
                self.aligned_Bscan[:, i] = np.concatenate([self.Bscan_data[abs(shift):, i], np.zeros(abs(shift))])
            else:
                self.aligned_Bscan[:, i] = self.Bscan_data[:, i]
        return np.array(self.aligned_Bscan)

    def plot(self):
        fig = plt.figure(figsize=(20, 7), tight_layout=True)
        ax = fig.add_subplot(111)
        plt.imshow(self.aligned_Bscan, aspect='auto', cmap='seismic',
                    extent=[0, self.aligned_Bscan.shape[1], self.aligned_Bscan.shape[0]*self.sample_interval, 0],
                    vmin=-50, vmax=50
                    )
        ax.set_xlabel('Trace number', fontsize=18)
        ax.set_ylabel('Time (ns)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)

        plt.title('Aligned B-scan', fontsize=20)

        plt.show()


#* example useage
#Bscan_data = np.loadtxt('/Volumes/SSD_kanda/LPR/LPR_2B/Raw_Bscan/Bscan.txt', delimiter=' ')
#correction = proccessing_time_zero_correction(Bscan_data)
#correction.find_peak_time()
#correction.align_peak_time()
#correction.plot()