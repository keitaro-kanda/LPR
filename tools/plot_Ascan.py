import os
import numpy as np
import matplotlib.pyplot as plt


Ascans_file_path = 'Ascans/LPR_2B_echo_data.txt'
sample_intarval = 0.312500 # [ns]

# load data
data = np.loadtxt(Ascans_file_path)
TWT = np.arange(len(data)) * sample_intarval

print(data.shape[0])
print(data.shape[1])
for i in range(len(data[0])):
    plt.figure(figsize=(10, 5))
    plt.plot(TWT, data[i, :])

    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude')
    plt.title(f'Ascan {i}')

    plt.show()


