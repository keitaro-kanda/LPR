import json
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
import argparse
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline



#* Data path
data_path = '/Volumes/SSD_kanda/LPR/LPR_2B/Processed_data/txt/4_gained_Bscan.txt'


#* Load data
print('Loading data...')
Bscan_data = np.loadtxt(data_path, delimiter=' ')
normalized_data = Bscan_data / np.amax(Bscan_data)  # Normalize the data
print('Data shape:', Bscan_data.shape)
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]



#* Define the function to calculate the f-k migration
def fk_migration(data, epsilon_r):

    #* Calculate the temporal angular frequency
    omega = 2 * np.pi * np.fft.fftfreq(data.shape[0], sample_interval)
    omega = omega[1: len(omega)//2]
    print('shape of omega: ', omega.shape)

    #* 2D Fourier transform, frequency-wavenaumber domain
    FK = np.fft.fft2(data)
    FK = FK[:len(omega), :]

    #* Calculate the wavenumber in x direction
    kx = 2 * np.pi * np.fft.fftfreq(data.shape[1], trace_interval)
    #kx = kx[1: len(kx)//2]
    #print(kx)
    print('shape of kx: ', kx.shape)

    #* Interpolate from frequency (ws) into wavenumber (kz)
    v = 299792458 / np.sqrt(epsilon_r)
    interp_real = RectBivariateSpline(np.fft.fftshift(kx), omega, np.fft.fftshift(FK.real, axes=1).T, kx=1, ky=1)
    interp_imag = RectBivariateSpline(np.fft.fftshift(kx), omega, np.fft.fftshift(FK.imag, axes=1).T, kx=1, ky=1)

    #* interpolation will move from frequency-wavenumber to wavenumber-wavenumber, KK = D(kx,kz,t=0)
    KK = np.zeros_like(FK)

    #* Calculate the wavenumber in z direction
    for zj in tqdm(range(len(omega)), desc='Calculating wavenumber in z direction'):
        kz_j = omega[zj] * 2 / v

        for xi in range(len(kx)):
            kx_i = kx[xi]
            omega_j = v / 2 * np.sqrt(kz_j**2 + kx_i**2)
            #kz_j = np.sqrt(omega[zj]**2 / v**2 - kx_i**2)
            #omega_j = np.sqrt(kx_i**2 + kz_j**2) * v

            #* Get the interpolated FFT values, real and imaginary, S(kx, kz, t=0)
            KK[zj, xi] = interp_real(kx_i, omega_j)[0, 0] + 1j * interp_imag(kx_i, omega_j)[0, 0]

    #* All vertical waevnumbers
    kz = omega * 2 / v

    #* Calculate the scaling factor
    """
    『地中レーダ』 p. 151
    omega^2 = (kx^2 + kz^2) * v^2
    dw = kz / sqrt(kx^2 + kz^2) * dky
    """
    kX, kZ = np.meshgrid(kx, kz)
    with np.errstate(divide='ignore', invalid='ignore'):
        scaling = kZ / np.sqrt(kX**2 + kZ**2)
    KK *= scaling
    #* The DC current should be zero
    KK[0, 0] = 0 + 0j



    #* Inverse 2D Fourier transform to get time domain data
    fk_data = np.fft.ifft2(KK)
    print('fk_data shape: ', fk_data.shape)

    return fk_data, v



#* Run the f-k migration function
er = 3.4 # Feng et al. (2024)
fk_data, v = fk_migration(Bscan_data, er)



#* Save the data
output_dir = '/Volumes/SSD_kanda/LPR/LPR_2B/fk_migration'
os.makedirs(output_dir, exist_ok=True)
np.savetxt(os.path.join(output_dir, 'fk_migration.txt'), np.abs(fk_data), delimiter=',')



#* Plot
plt.figure(figsize=(18, 6), facecolor='w', edgecolor='w')
im = plt.imshow(np.abs(fk_data), cmap='jet', aspect='auto',
                extent=[0, fk_data.shape[1] * trace_interval,
                fk_data.shape[0] * sample_interval * v / 2, 0],
                #vmin=0, vmax=30
                )

plt.xlabel('x [m]', fontsize=20)
plt.ylabel('z [m] (assume ' r'$\varepsilon_r = $'+ str(er) + ')', fontsize=20)
plt.tick_params(labelsize=18)


delvider = axgrid1.make_axes_locatable(plt.gca())
cax = delvider.append_axes('right', size='5%', pad=0.5)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=18)


plt.savefig(os.path.join(output_dir, 'fk_migration.png'))
plt.savefig(os.path.join(output_dir, 'fk_migration.pdf'), format='pdf', dpi=300)
plt.show()