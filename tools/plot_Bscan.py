import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1


Ascans_file_path = 'Ascans/LPR_2B_echo_data.txt'

#* load data
Ascans = np.loadtxt(Ascans_file_path, skiprows=0, delimiter=' ')

sample_interval = 0.312500  # [ns]


#* plot
fig = plt.figure(figsize=(12, 7), tight_layout=True)
ax = fig.add_subplot(111)
plt.imshow(Ascans, aspect='auto', cmap='seismic',
            extent=[0, Ascans.shape[1], Ascans.shape[0]*sample_interval, 0],
            vmin=-30, vmax=30
            )
ax.set_xlabel('Trace number', fontsize=18)
ax.set_ylabel('Time (ns)', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)

title = os.path.splitext(os.path.basename(Ascans_file_path))[0]
plt.title(title, fontsize=20)


# plot with colorbar
delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax).set_label('Amplitude', fontsize=18)
cax.tick_params(labelsize=16)

# save plot
plot_name = os.path.splitext(os.path.basename(Ascans_file_path))[0] + '.png'
output_dir = os.path.dirname(Ascans_file_path)
plt.savefig(output_dir + '/' + plot_name)
plt.show()