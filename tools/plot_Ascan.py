import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
import argparse
import json


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='plot_Ascan.py',
    description='Plot A-scan from B-scan data',
    epilog='End of help message',
    usage='python tools/fk_migration.py [data_path] [x] [t_first] [t_last]',
)
parser.add_argument('data_path', help='Path to the txt file of data.')
parser.add_argument('x', type=float, help='x position [m] of the A-scan')
parser.add_argument('t_first', type=int, help='Start time of the A-scan [ns]')
parser.add_argument('t_last', type=int, help='Last time of the A-scan [ns]')
args = parser.parse_args()



#* Define the data path
data_path = args.data_path
output_dir = os.path.join(os.path.dirname(data_path), 'A_scan')
os.makedirs(output_dir, exist_ok=True)



#* Parameters
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]



#* Load data
print('Loading data...')
Bscan = np.loadtxt(data_path, delimiter=' ')
print('Data shape:', Bscan.shape)



#* Extract A-scan data from B-scan data
x_idx = int(args.x / trace_interval)
t_first_idx = int(args.t_first * 1e-9 / sample_interval)
t_last_idx = int(args.t_last * 1e-9 / sample_interval)
Ascan = Bscan[t_first_idx:t_last_idx, x_idx]

t_array = np.arange(t_first_idx, t_last_idx) * sample_interval / 1e-9



#* Closeup B-scan around the A-scan
x_first_idx = x_idx - int(15/trace_interval) # index
x_last_idx = x_idx + int(15/trace_interval) # index
Bscan_trim = Bscan[t_first_idx:t_last_idx, x_first_idx:x_last_idx]



#* Plot A-scan and closeup B-scan
fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharey=True, tight_layout=True)
ax[0].plot(Ascan, t_array, color='black')
ax[0].set_xlabel('Amplitude', fontsize=20)
ax[0].grid(True)
ax[0].tick_params(labelsize=18)
ax[0].set_xlim(-np.amax(np.abs(Ascan))*1.2, np.amax(np.abs(Ascan))*1.2)

im = ax[1].imshow(Bscan_trim,
                aspect='auto', cmap='seismic',
                vmin=-np.amax(np.abs(Bscan))/5, vmax=np.amax(np.abs(Bscan))/5,
                extent=[x_first_idx*trace_interval, x_last_idx*trace_interval, args.t_last, args.t_first])
ax[1].set_xlabel('Distance [m]', fontsize=20)
ax[1].tick_params(labelsize=18)

ax[1].vlines(args.x, args.t_first, args.t_last, color='k', linestyle='-.')

delvider = axgrid1.make_axes_locatable(ax[1])
cax = delvider.append_axes('right', size='3%', pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Amplitude', fontsize=20)
cbar.ax.tick_params(labelsize=18)


fig.supxlabel('Time [ns]', fontsize=20)

plt.savefig(os.path.join(output_dir, f'{args.x}_t{args.t_first}_{args.t_last}.png'), dpi=120)
plt.savefig(os.path.join(output_dir, f'{args.x}_t{args.t_first}_{args.t_last}.pdf'), dpi=300)
plt.show()