import os
import numpy as np
import matplotlib.pyplot as plt


data_dile_path = 'LPR_2B/CE4_GRAS_LPR-2B_SCI_N_20231216075001_20231217065500_0316_A.2B_echo_data.txt'
sample_intarval = 0.312500 # [ns]

# load data
data = np.loadtxt(data_dile_path)
TWT = np.arange(len(data)) * sample_intarval

# plot
plt.figure(figsize=(10, 5))
plt.plot(TWT, data)
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude')

plt.show()


