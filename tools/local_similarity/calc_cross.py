import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import mpl_toolkits.axes_grid1 as axgrid1


# Set constants
dt = 0.312500e-9  # Sample interval in seconds
dx = 3.6e-2  # Trace interval in meters, from Li et al. (2020), Sci. Adv.
c = 299792458  # Speed of light in m/s
epsilon_r = 4.5  # Relative permittivity, from Feng et al. (2024)
dz = dt * c / np.sqrt(epsilon_r) / 2  # Depth interval in meters
sigma_smoothing = 0.5  # ガウス平滑化の標準偏差 (Local Similarityスペクトル全体に適用)
soft_threshold_value = 0.5 # 局所正規化相互相関は-1から1なので、0.5などの中央値を仮設定
reflection_mask = None  # 反射領域のマスク (必要に応t場合はリストで指定)
local_max_neighborhood = 5  # 局所最大値を検出するための近傍サイズ

# Debugging pixels (row, col)
debug_pixel_rows = np.array([100, 500, 1000, 1500], dtype=np.int64)
debug_pixel_cols = np.array([100, 5000, 10000, 20000], dtype=np.int64)

# Input paths (省略)
data_paths = [
    ["/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2A/Processed_Data/4_Gain_function/fk_migration/fk_migration.txt",
        "/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Processed_Data/4_Gain_function/fk_migration/fk_migration.txt"],
    ["/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2A/Processed_Data/5_Terrain_correction/5_Terrain_correction.txt",
        "/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Processed_Data/5_Terrain_correction/5_Terrain_correction.txt"]
]

# User input for data type selection (省略)
data_type = input("Select data type (1 for Gained data, 2 for Terrain corrected data): ").strip()
if data_type not in ['1', '2']:
    raise ValueError("Invalid data type selected. Please choose 1 or 2.")

# Determine the data paths based on user input (省略)
if data_type == '1':
    data1_path, data2_path = data_paths[0]
else:
    data1_path, data2_path = data_paths[1]

if not os.path.exists(data1_path):
    raise FileNotFoundError(f"The file {data1_path} does not exist.")
if not os.path.exists(data2_path):
    raise FileNotFoundError(f"The file {data2_path} does not exist.")

# Load data (省略)
print("Loading data...")
data1 = np.loadtxt(data1_path)
print(f"Data 1 shape: {data1.shape}")
data2 = np.loadtxt(data2_path)
print(f"Data 2 shape: {data2.shape}")
print(" ")

# Set axis (省略)
x_axis = np.arange(data1.shape[1]) * dx
z_axis = np.arange(data1.shape[0]) * dz

# Output directory (省略)
base_dir = "/Volumes/SSD_Kanda_SAMSUNG/LPR/Local_similarity/cross"
if data_type == '1':
    output_dir = os.path.join(base_dir, '4_Gain_function')
else:
    output_dir = os.path.join(base_dir, '5_Terrain_correction')
os.makedirs(output_dir, exist_ok=True)


# 2つのデータの積を計算
multiply = data1 * data2

# Make histglam
multiply_flat = np.log(np.abs(multiply)).flatten()

plt.figure(figsize=(10, 6))
plt.hist(multiply_flat, bins=10, range=[0, 20], edgecolor='black')
plt.xlabel('Log amplitude')
plt.ylabel('Number')

plt.grid(axis='y', alpha=0.75)

plt.savefig(os.path.join(output_dir, 'multiply_hist.png'))
plt.close()


# Soft threshold
SNR = np.amax(multiply) / np.amin(multiply)
print("SNR: ", SNR)

threshold = np.amin(multiply) * 2e5 # 0.1: Zhang et al. (2021), appendix C
mask = multiply >= threshold
multiply[multiply < threshold] = 0
multiply[mask] = multiply[mask] - threshold

multiply = np.log(np.abs(multiply + 1e-12)) # log(0)を避ける
np.savetxt(os.path.join(output_dir, "multiply.txt"), multiply, delimiter=' ')





# plot
print("Plotting the cross-correlation...")
fig, ax = plt.subplots(figsize=(18, 6))
im = ax.imshow(multiply.reshape(data1.shape), aspect='auto', cmap='viridis',
                    extent=[x_axis.min(), x_axis.max(), z_axis.max(), z_axis.min()],
                    vmin = 0, vmax = np.amax(multiply))
                    #origin='lower')
ax.set_xlabel('Distance (m)', fontsize=20)
ax.set_ylabel('Depth (m)', fontsize=20)
ax.tick_params(labelsize=18)

delvider = axgrid1.make_axes_locatable(plt.gca())
cax = delvider.append_axes('right', size='5%', pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Log amplitude', fontsize=20)
cbar.ax.tick_params(labelsize=18)

plt.savefig(os.path.join(output_dir, 'cross.png'))
plt.show()