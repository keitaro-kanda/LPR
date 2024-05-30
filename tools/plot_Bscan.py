import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1


Ascans_file_path = '/Volumes/SSD_kanda/LPR/LPR_2B/ECHO/ECHO_0001.txt'
print(os.path.split(Ascans_file_path)[0])

#* load data
Ascans = np.loadtxt(Ascans_file_path, skiprows=1, delimiter=' ')
print("Ascans shape:", Ascans.shape)
sample_interval = 0.312500  # [ns]


#* plot
fig = plt.figure(figsize=(20, 7), tight_layout=True)
ax = fig.add_subplot(111)
plt.imshow(Ascans, aspect='auto', cmap='seismic',
            extent=[0, Ascans.shape[1], Ascans.shape[0]*sample_interval, 0],
            vmin=-100, vmax=100
            )
ax.set_xlabel('Trace number', fontsize=18)
ax.set_ylabel('Time (ns)', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)

title = os.path.splitext(os.path.basename(Ascans_file_path))[0]
plt.title(title, fontsize=20)


#* plot colorbar
delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax).set_label('Amplitude', fontsize=18)
cax.tick_params(labelsize=16)

# save plot
#plot_name = os.path.splitext(os.path.basename(Ascans_file_path))[0] + '.png'
#output_dir = os.path.dirname(Ascans_file_path)
#plt.savefig(output_dir + '/' + plot_name)
plt.show()