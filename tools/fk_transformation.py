import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os



data_path = input("Enter the path to the data file: ").strip()
if not os.path.exists(data_path):
    raise FileNotFoundError(f"The file {data_path} does not exist.")


# Make output directory
output_dir = os.path.join(os.path.dirname(data_path), 'fk_transformation')
os.makedirs(output_dir, exist_ok=True)


# Load data
print('Loading data...')
Bscan_data = np.loadtxt(data_path, delimiter=' ')


# Set constants
dt = 0.312500e-9  # Sample interval in seconds
dx = 3.6e-2  # Trace interval in meters, from Li et al. (2020), Sci. Adv.
c = 299792458  # Speed of light in m/s

# Transform data
# Original data is in time-space domain, we need to transform it to frequency-wavenumber domain
print('Transforming data to frequency-wavenumber domain...')
Bscan_data = Bscan_data / np.amax(Bscan_data)  # Normalize the data
N = Bscan_data.shape[0]  # Number of traces
t = np.arange(N) * dt  # Time vector
f = np.fft.fftfreq(N, dt)  # Frequency vector
f = np.fft.fftshift(f)  # Shift zero frequency to center
f_MHz = f * 1e-6  # Convert frequency to MHz for plotting
K = np.fft.fftfreq(N, dx)  # Wavenumber vector
K = np.fft.fftshift(K)  # Shift zero wavenumber to center

# 2D Fourier transform to get frequency-wavenumber data
KK = np.fft.fft2(Bscan_data)
KK_shifted = np.fft.fftshift(KK)  # Shift zero frequency to center
KK_power_log = 20 * np.log(np.abs(KK_shifted))  # Logarithm of the absolute value for better visualization
# Plotting the frequency-wavenumber data
plt.figure(figsize=(18, 6))
plt.imshow(KK_power_log, aspect='auto',
            extent=(K.min(), K.max(), f_MHz.min(), f_MHz.max()),
            cmap='turbo', origin='lower', vmin=0, vmax=np.max(np.abs(KK_power_log))/2)
plt.colorbar(label='Amplitude')
plt.xlabel('Wavenumber (1/m)')
plt.ylabel('Frequency (Hz)')
plt.title('Frequency-Wavenumber Domain Representation')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'frequency_wavenumber_representation.png'))
plt.show()