import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mpl_toolkits.axes_grid1 as axgrid1


Bscan_data = np.loadtxt('Ascans/LPR_2B_echo_data.txt', skiprows=1, delimiter=' ')

#sample_interval = 0.312500e-9  # [s]


class processing_background_removal():
    def __init__(self, Bscan_data, sample_interval=0.312500e-9):
        self.Bscan_data = Bscan_data # input is Bscan data array, not file path
        self.sample_interval = sample_interval


    def subtract_background(self):
        self.background_data = np.mean(self.Bscan_data, axis=1)
        self.background_removed_Bscan = np.zeros_like(self.Bscan_data)
        for i in tqdm(range(self.Bscan_data.shape[1]), desc='Subtracting background'):
            self.background_removed_Bscan[:, i] =  self.Bscan_data[:, i] - self.background_data
        return self.background_removed_Bscan


    def plot(self):
        fig = plt.figure(figsize=(12, 7), tight_layout=True)
        ax = fig.add_subplot(111)
        plt.imshow(self.background_removed_Bscan, aspect='auto', cmap='seismic',
                    extent=[0, self.background_removed_Bscan.shape[1], self.background_removed_Bscan.shape[0]*self.sample_interval, 0],
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


#processer = processing_background_removal(Bscan_data)
#background_removed_Bscan = processer.subtract_background()
#processer.plot()

"""
#* make background data by averaging all Ascans
background = np.mean(Bscan_data, axis=1)
plt.plot(background)
plt.show()


#* subtract the background from all Ascans
background_removed_Bscan = np.zeros(Bscan_data.shape)
for i in tqdm(range(Bscan_data.shape[1]), desc='Subtracting background'):
    background_removed_Bscan[:, i] =  Bscan_data[:, i] - background


#* plot
fig = plt.figure(figsize=(12, 7), tight_layout=True)
ax = fig.add_subplot(111)
plt.imshow(background_removed_Bscan, aspect='auto', cmap='seismic',
            extent=[0, background_removed_Bscan.shape[1], background_removed_Bscan.shape[0]*sample_interval, 0],
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