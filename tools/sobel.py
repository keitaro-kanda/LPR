import numpy as np
import matplotlib.pyplot as plt
import cv2
import mpl_toolkits.axes_grid1 as axgrid1
import os


data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Processed_Bscan/txt/gained_Bscan.txt'
output_dir = '/Volumes/SSD_kanda/LPR/LPR_2B/test/sobel'
os.makedirs(output_dir, exist_ok=True)

Bscan_data = np.loadtxt(data_path, delimiter=' ')
sample_interval = 0.312500e-9  # [s]

#* Compute the derivative of the data in x direction with Sobel with a kernel size of 5
#* This should eliminate the horizonatal lines
sobelx = cv2.Sobel(Bscan_data[int(60e-9/sample_interval):, :], cv2.CV_64F, 1, 0, ksize=3)
sobelx = np.abs(sobelx)
sobely = cv2.Sobel(Bscan_data[int(60e-9/sample_interval):, :], cv2.CV_64F, 0, 1, ksize=3)
sobely = np.abs(sobely)

sobel_combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

data_list = [sobelx, sobely, sobel_combined]
title = ['Sobel X', 'Sobel Y', 'Sobel Combined']

plt.figure(figsize=(18, 18))
for i in range(len(data_list)):
    plt.subplot(3, 1, i+1)
    plt.imshow(data_list[i], aspect='auto', cmap='jet',
                extent=[0, data_list[i].shape[1], data_list[i].shape[0]*sample_interval*1e9, 0],
                vmin=0, vmax=np.amax(data_list[i])/2
                )

    delvider = axgrid1.make_axes_locatable(plt.gca())
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(cax=cax).set_label('Amplitude', fontsize=18)
    plt.title(title[i], fontsize=20)

plt.savefig(output_dir + '/sobel.png')
plt.show()