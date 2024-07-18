import json
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
from tqdm import tqdm
import argparse



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='resampling.py',
    description='Resampling ECHO data',
    epilog='End of help message',
    usage='python tools/resampling.py [type]',
)
parser.add_argument('type', choices = ['calc', 'plot'], help='Choose the function type')
args = parser.parse_args()


#* Define the data path
data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Processed_Bscan/txt/gained_Bscan.txt'
output_dir = '/Volumes/SSD_kanda/LPR/LPR_2B/test/autocorrelation'
os.makedirs(output_dir, exist_ok=True)



#* Load data
print('Loading data...')
print('   ')
Bscan_data = np.loadtxt(data_path, delimiter=' ')
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]



#* Define the function to calculate the autocorrelation
N = Bscan_data.shape[0]
def calc_autocorrelation(Ascan): # data: 1D array
    #* Calculate the autocorrelation
    auto_corr = np.zeros(N)
    data_ave = np.mean(Ascan)
    sigma = 1 / N * np.sum((Ascan - data_ave)**2)
    for i in range(1, N+1):
        auto_corr[i-1] = 1 / N *np.sum((Ascan[i:] - data_ave) * (Ascan[:-i] - data_ave)) / sigma

    return auto_corr



#* Calculate the autocorrelation of the data
if args.type=='calc':
    auto_corr = np.zeros(Bscan_data.shape)
    for i in tqdm(range(Bscan_data.shape[1]), desc='Calculating autocorrelation'):
        auto_corr[:, i] = calc_autocorrelation(Bscan_data[:, i])
    np.savetxt(output_dir + '/autocorrelation.txt', auto_corr, delimiter=' ')
elif args.type=='plot':
    auto_corr = np.loadtxt(output_dir + '/autocorrelation.txt', delimiter=' ')
else:
    raise ValueError('Invalid function type')



#* Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
norm = plt.Normalize(vmin=-1, vmax=1)
im = ax.imshow(auto_corr, cmap='jet', aspect='auto',
            extent=[0, auto_corr.shape[1]*trace_interval, auto_corr.shape[0]*sample_interval*1e9, 0],
            norm=norm)
delvider= axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', '5%', pad='3%')
cbar = plt.colorbar(im, cax=cax)

plt.show()
