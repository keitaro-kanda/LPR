import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
from tqdm import tqdm


def calc_cross_correlation(data1, data2):
    """
    Calculate the cross-correlation between two 2D
    """
    if data1.shape != data2.shape:
        raise ValueError("Data shapes do not match for cross-correlation.")
    
    # Normalize the data
    data1 = data1 / np.amax(np.abs(data1))
    data2 = data2 / np.amax(np.abs(data2))
    
    correlation = np.zeros_like(data1)  # Initialize correlation array with the same shape as data1
    # Calculate cross-correlation
    for i in tqdm(range(data1.shape[1])):
        correlation[:, i] = np.correlate(data1[:, i], data2[:, i])

    return correlation


def main():
    # Set constants
    sample_interval = 0.312500e-9  # Sample interval in seconds
    trace_interval = 3.6e-2  # Trace interval in meters, from Li et al. (2020), Sci. Adv.
    c = 299792458  # Speed of light in m/s
    epsilon_r = 4.5  # Relative permittivity, from Feng et al. (2024)

    # Input paths
    data_paths = [
        ["/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2A/Processed_Data/4_Gain_function/4_Bscan_gain.txt",
            "/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Processed_Data/4_Gain_function/4_Bscan_gain.txt"],
        ["/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2A/Processed_Data/5_Terrain_correction/5_Terrain_correction.txt",
            "/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Processed_Data/5_Terrain_correction/5_Terrain_correction.txt"]
    ]

    # User input for data type selection
    data_type = input("Select data type (1 for Gained data, 2 for Terrain corrected data): ").strip()
    if data_type not in ['1', '2']:
        raise ValueError("Invalid data type selected. Please choose 1 or 2.")
    
    # Determine the data paths based on user input
    if data_type == '1':
        data1_path, data2_path = data_paths[0]
    else:
        data1_path, data2_path = data_paths[1]

    if not os.path.exists(data1_path):
        raise FileNotFoundError(f"The file {data1_path} does not exist.")
    if not os.path.exists(data2_path):
        raise FileNotFoundError(f"The file {data2_path} does not exist.")

    # Load data
    print("Loading data...")
    data1 = np.loadtxt(data1_path)
    print(f"Data 1 shape: {data1.shape}")
    data2 = np.loadtxt(data2_path)
    print(f"Data 2 shape: {data2.shape}")
    print(" ")

    # Set axis
    x_axis = np.arange(data1.shape[1]) * trace_interval  # x-axis in meters
    z_axis = np.arange(data1.shape[0]) * sample_interval * c / np.sqrt(epsilon_r) / 2  # z-axis in meters

    # Output directory
    base_dir = "/Volumes/SSD_Kanda_SAMSUNG/LPR/Local_similarity"
    if data_type == '1':
        output_dir = os.path.join(base_dir, '4_Gain_function')
    else:
        output_dir = os.path.join(base_dir, '5_Terrain_correction')
    os.makedirs(output_dir, exist_ok=True)

    # Calculate cross-correlation
    print("Calculating cross-correlation...")
    correlation = calc_cross_correlation(data1, data2)
    print("Cross-correlation calculated.")
    print(f"Cross-correlation shape: {correlation.shape}")
    print(" ")

    # Plotting the cross-correlation
    print("Plotting the cross-correlation...")
    fig, ax = plt.subplots(figsize=(18, 6))
    im = ax.imshow(correlation.reshape(data1.shape), aspect='auto', cmap='turbo',
                        extent=[x_axis.min(), x_axis.max(), z_axis.max(), z_axis.min()],
                        origin='lower')
    ax.set_xlabel('Distance (m)', fontsize=20)
    ax.set_ylabel('Depth (m)', fontsize=20)
    ax.tick_params(labelsize=18)

    delvider = axgrid1.make_axes_locatable(plt.gca())
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Amplitude', fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    plt.savefig(os.path.join(output_dir, 'cross_correlation.png'))
    plt.show()

if __name__ == "__main__":
    main()