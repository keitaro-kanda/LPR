import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm



data_path = '/Volumes/SSD_Kanda_BUFFALO/LPR/LPR_2B/Processed_Data/3_Background_removal/3_Bscan_background_removal.txt'
#* Define output folder path
output_dir = os.path.join(os.path.dirname(data_path), 'Hilbert')
os.makedirs(output_dir, exist_ok=True)

print('Loading data...')
print('   ')
data = np.loadtxt(data_path, delimiter=' ')


#* Parameters
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]
Gsys = 133.3 # [dB]
Gsys_lin = 10**(Gsys/10)
c = 299792458 # [m/s]
f0 = 500e6 # [Hz]


#* Assumed parameters
epsilon_rock = 9.0
epsilon_regolith = 3.0
tan_delta = 0.07 # [Feng et al. (2022)]


#* Calculate transmission coefficient of surface
T = (4 * np.sqrt(epsilon_regolith) * 1) / np.abs(1 + np.sqrt(epsilon_regolith))**2

#* Calculate reflection coefficient of rock
R = ((np.sqrt(epsilon_rock) - np.sqrt(epsilon_regolith)) / (np.sqrt(epsilon_rock) + np.sqrt(epsilon_regolith)))**2


Pmin = np.abs(data).min()
print('Pmin:', Pmin)


#* Calculate RCS
RCS = np.zeros_like(data)
A = (4 * np.pi)**3 * f0**2 * c / (8 * T**2 * np.sqrt(epsilon_regolith))
t_array = np.arange(data.shape[0]) * sample_interval

for i in tqdm(range(data.shape[1])):
    for j in range(data.shape[0]):
        t = t_array[j]
        RCS[j, i] = A * t**3 * (1/np.exp(-2 * f0 * tan_delta * t)) * data[j, i] / (Pmin * Gsys_lin) # GsysとRは入ってない！！！



#* Plot
print('Plotting...')
plt.figure(figsize=(18, 6), facecolor='w', edgecolor='w', tight_layout=True)
im = plt.imshow(RCS, cmap='viridis', aspect='auto',
                extent=[0, RCS.shape[1] * trace_interval,
                RCS.shape[0] * sample_interval / 1e-9, 0],
                #vmin=0, vmax=np.amax(RCS)/5
                )

plt.xlabel('x [m]', fontsize=20)
plt.ylabel('t [ns]', fontsize=20)
plt.tick_params(labelsize=18)


delvider = axgrid1.make_axes_locatable(plt.gca())
cax = delvider.append_axes('right', size='5%', pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Amplitude', fontsize=20)
cbar.ax.tick_params(labelsize=18)

plt.show()