import numpy as np
import matplotlib.pyplot as plt
import os


Ascans_file_path = 'Ascans/LPR_2B_echo_data.txt'

#* load data
Ascans = np.loadtxt(Ascans_file_path)
print(Ascans.shape)
print(Ascans.shape[0])
print(Ascans.shape[1])

sample_interval = 0.312500  # [ns]


#* plot
plt.figure(figsize=(10, 10))
plt.imshow(Ascans, aspect='auto', cmap='rainbow',
            extent=[0, Ascans.shape[1], Ascans.shape[0]*sample_interval, 0],
            vmin=-20, vmax=20
            )
plt.xlabel('Trace number')
plt.ylabel('Time (ns)')
plt.colorbar(label='Amplitude')



title = os.path.splitext(os.path.basename(Ascans_file_path))[0]
plt.title(title)

plt.show()