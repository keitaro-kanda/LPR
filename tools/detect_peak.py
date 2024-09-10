import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
import argparse
from scipy import signal
from tqdm import tqdm


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='detect_peak.py',
    description='Detect echo peak from B-scan data and plot',
    epilog='End of help message',
    usage='python tools/detect_peak.py [data_path]',
)
parser.add_argument('data_path', help='Path to the txt file of data.')
args = parser.parse_args()



#* Define the data path
data_path = args.data_path
output_dir = os.path.join(os.path.dirname(data_path), 'detect_peak')
os.makedirs(output_dir, exist_ok=True)



#* Parameters
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]



#* Load data
print('Loading data...')
data = np.loadtxt(data_path, delimiter=' ')
print('Data shape:', data.shape)



#* Calculate the envelope
envelope = np.abs(signal.hilbert(data))



#* Detect peak from the B-scan data
scatter_x_idx = []
scatter_time_idx = []
scatter_value = []
for i in tqdm(range(data.shape[1])):
    Ascan = envelope[int(50e-9/sample_interval):, i]

    #* Calculate the background
    background = np.mean(np.abs(Ascan))

    #* Detect the peak in the envelope
    threshold = background * 3

    j = 0
    while j < len(Ascan):
        if envelope[i] > threshold:
            start = j
            while j < len(envelope) and envelope[j] > threshold:
                j += 1
            end = j
            scatter_time_idx.append(np.argmax(np.abs(Ascan[start:end])) + start)
            scatter_value.append(Ascan[scatter_time_idx[-1]])
            scatter_x_idx.append(i)
        i += 1



#* Plot the detected peak
fig, ax = plt.subplots(1, 1, figsize=(18, 10))

#* Second plot, scatter plot with imshow of envelope
im = ax.imshow(envelope, aspect='auto', cmap='jet',
                extent=[0, envelope.shape[1]*trace_interval, envelope.shape[0]*sample_interval*1e9, 0],
                vmin=-np.amax(np.abs(envelope))/5, vmax=np.amax(np.abs(envelope))/5
                )
scatter = ax.scatter(scatter_x_idx, scatter_time_idx, c=scatter_value, cmap=cm, s=1)


#* Set labels
ax.set_xlabel('Distance [m]', fontsize=20)
ax.set_ylabel('Time [ns]', fontsize=20)
ax.tick_params(labelsize=16)


#* Colorbar for the scatter plot
cbar1 = fig.colorbar(scatter, ax=ax, orientation='horizontal')
cbar1.set_label('Peak amplitude', fontsize=20)
cbar1.ax.tick_params(labelsize=16)

#* Colorbar for the envelope
cbar2 = plt.colorbar(im, cax=ax, orientation='vertical')
cbar2.set_label('Amplitude', fontsize=20)
cbar2.ax.tick_params(labelsize=16)


#* Save the plot
plt.savefig(output_dir + '/peak_detection.png', format='png', dpi=120)
plt.savefig(output_dir + '/peak_detection.pdf', format='pdf', dpi=600)
plt.show()