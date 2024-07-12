import numpy as np
import matplotlib.pyplot as plt
import cv2
import mpl_toolkits.axes_grid1 as axgrid1


data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Processed_Bscan/txt/gained_Bscan.txt'

Bscan_data = np.loadtxt(data_path, delimiter=' ')
sample_interval = 0.312500e-9  # [s]

#* Compute the derivative of the data in x direction with Sobel with a kernel size of 5
#* This should eliminate the horizonatal lines
sobelx = cv2.Sobel(Bscan_data[int(60e-9/sample_interval):, :], cv2.CV_64F, 1, 0, ksize=3)
sobelx = np.abs(sobelx)
sobely = cv2.Sobel(Bscan_data[int(60e-9/sample_interval):, :], cv2.CV_64F, 0, 1, ksize=3)
sobely = np.abs(sobely)

sobel_combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)


plt.figure(figsize=(18, 18))
plt.subplot(311)
plt.imshow(sobelx, aspect='auto', cmap='jet',
            extent=[60, sobelx.shape[1], sobelx.shape[0]*sample_interval*1e9, 0],
            vmin=0, vmax=np.amax(sobelx)/2
            )
plt.colorbar()
plt.title('Sobel X', fontsize=20)

plt.subplot(312)
plt.imshow(sobely, aspect='auto', cmap='jet',
            extent=[60, sobely.shape[1], sobely.shape[0]*sample_interval*1e9, 0],
            vmin=0, vmax=np.amax(sobely)/2
            )
plt.colorbar()
plt.title('Sobel Y', fontsize=20)

plt.subplot(313)
plt.imshow(sobel_combined, aspect='auto', cmap='jet',
            extent=[0, sobel_combined.shape[1], sobel_combined.shape[0]*sample_interval*1e9, 0],
            vmin=0, vmax=np.amax(sobel_combined)/2
            )
plt.colorbar()

plt.show()