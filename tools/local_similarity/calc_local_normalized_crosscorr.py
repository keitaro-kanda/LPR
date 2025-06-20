import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
from scipy.ndimage import gaussian_filter # ガウス平滑化用
from numba import jit, prange # prange は parallel=True のループで必要


def apply_gaussian_smoothing(data: np.ndarray, sigma: float) -> np.ndarray:
    """
    データにガウス平滑化を適用する。
    Local SimilarityのS_m関数の実装として使用されることを想定。

    Args:
        data (np.ndarray): 平滑化を適用するデータ。
        sigma (float): ガウスカーネルの標準偏差。大きいほど強く平滑化される。

    Returns:
        np.ndarray: 平滑化適用後のデータ。
    """
    return gaussian_filter(data, sigma=sigma, mode='nearest')


@jit(nopython=True)
def _compute_local_normalized_cross_correlation_element(sub_A_flat, sub_B_flat, debug_mode=False, mean_center_data=True):
    """
    局所正規化相互相関の単一要素計算をNumbaで高速化するヘルパー関数。
    mean_center_data: Trueの場合、ピアソン相関 (平均値中心化)。Falseの場合、正規化済み相互相関 (非平均値中心化)。
    """
    n = len(sub_A_flat)
    if n == 0:
        return 0.0

    sum_A_sq = 0.0
    sum_B_sq = 0.0
    sum_AB = 0.0
    
    if mean_center_data:
        mean_A = np.mean(sub_A_flat)
        mean_B = np.mean(sub_B_flat)
    else:
        mean_A = 0.0 # 平均値中心化しない
        mean_B = 0.0 # 平均値中心化しない

    for k in range(n):
        val_A = sub_A_flat[k] - mean_A
        val_B = sub_B_flat[k] - mean_B

        sum_A_sq += val_A * val_A
        sum_B_sq += val_B * val_B
        sum_AB += val_A * val_B

    denominator = np.sqrt(sum_A_sq * sum_B_sq)
    
    # デバッグ情報出力
    if debug_mode:
        print("  sub_A_flat stats: Min=", np.min(sub_A_flat), ", Max=", np.max(sub_A_flat), ", Mean=", np.mean(sub_A_flat), ", Std=", np.std(sub_A_flat))
        print("  sub_B_flat stats: Min=", np.min(sub_B_flat), ", Max=", np.max(sub_B_flat), ", Mean=", np.mean(sub_B_flat), ", Std=", np.std(sub_B_flat))
        print("  sum_A_sq:", sum_A_sq, ", sum_B_sq:", sum_B_sq)
        print("  sum_AB:", sum_AB)
        print("  denominator:", denominator)

    if denominator == 0:
        correlation = 0.0
    else:
        correlation = sum_AB / denominator
    
    if debug_mode:
        print("  Local Normalized Cross-Correlation:", correlation)

    return correlation


@jit(nopython=True, parallel=True)
def calculate_local_similarity_spectrum_fast(data_A: np.ndarray, data_B: np.ndarray,
                                             dx: float, dz: float,
                                             window_size_m: tuple, # ウィンドウサイズをメートル単位で指定
                                             mean_center_correlation: bool, # 相関の平均値中心化の有無
                                             debug_pixel_rows=None, debug_pixel_cols=None) -> np.ndarray:
    """
    高速化されたLocal Similarityスペクトル計算関数 (局所正規化相互相関を使用)。
    Numbaと行列演算の最適化を適用。
    window_size_m: (time_win_m, space_win_m) - ウィンドウサイズをメートル単位で指定。
    mean_center_correlation: Trueの場合、ピアソン相関 (平均値中心化)。Falseの場合、正規化済み相互相関 (非平均値中心化)。
    debug_pixel_rows, debug_pixel_cols: デバッグ情報を表示する行と列のインデックス配列。
    """
    rows, cols = data_A.shape
    local_similarity_spectrum = np.zeros((rows, cols))

    # ウィンドウサイズをメートル単位からピクセル単位に変換
    time_win_pixels = max(1, int(window_size_m[0] / dz))
    space_win_pixels = max(1, int(window_size_m[1] / dx))
    
    # ウィンドウサイズが偶数の場合、奇数になるように調整（中央揃えのため）
    if time_win_pixels % 2 == 0: time_win_pixels += 1
    if space_win_pixels % 2 == 0: space_win_pixels += 1


    has_debug_pixels = debug_pixel_rows is not None and debug_pixel_cols is not None

    for i in prange(rows):
        for j in prange(cols):
            current_debug_mode = False
            if has_debug_pixels:
                for dbg_idx in range(len(debug_pixel_rows)):
                    if i == debug_pixel_rows[dbg_idx] and j == debug_pixel_cols[dbg_idx]:
                        current_debug_mode = True
                        break 

            r_start = max(0, i - time_win_pixels // 2)
            r_end = min(rows, i + time_win_pixels // 2 + 1) # +1はスライスのため
            c_start = max(0, j - space_win_pixels // 2)
            c_end = min(cols, j + space_win_pixels // 2 + 1) # +1はスライスのため

            sub_A_flat = data_A[r_start:r_end, c_start:c_end].flatten()
            sub_B_flat = data_B[r_start:r_end, c_start:c_end].flatten()

            if current_debug_mode:
                print(f"\n--- Debugging pixel ({i}, {j}) ---")
            
            local_similarity_spectrum[i, j] = _compute_local_normalized_cross_correlation_element(
                sub_A_flat, sub_B_flat, debug_mode=current_debug_mode, mean_center_data=mean_center_correlation)
            
            if current_debug_mode:
                print(f"--- End debugging pixel ({i}, {j}) ---\n")

    return local_similarity_spectrum


def apply_soft_threshold(similarity_spectrum: np.ndarray, threshold: float) -> np.ndarray:
    """
    Local Similarityスペクトルにソフト閾値関数を適用する 。
    論文の式(8) に基づく 。
    """
    thresholded_spectrum = np.where(similarity_spectrum > threshold, similarity_spectrum, 0)
    return thresholded_spectrum


def mute_reflection_area(spectrum: np.ndarray, reflection_area_mask) -> np.ndarray:
    """
    反射領域をミュートする。
    reflection_area_maskは、ミュートする領域のインデックス（例: [(r_start, r_e, c_start, c_e), ...])
    またはブールマスクとして提供されると想定。
    """
    muted_spectrum = spectrum.copy()
    if isinstance(reflection_area_mask, np.ndarray) and reflection_area_mask.dtype == bool:
        muted_spectrum[reflection_area_mask] = 0
    elif isinstance(reflection_area_mask, list):
        for area in reflection_area_mask:
            r_s, r_e, c_s, c_e = area
            muted_spectrum[r_s:r_e, c_s:c_e] = 0
    return muted_spectrum


from scipy.ndimage import maximum_filter

def extract_local_maximums(spectrum: np.ndarray, neighborhood_size: int = 3) -> list[tuple[int, int]]:
    """
    Local Similarityスペクトルから局所最大値を抽出し、岩石の位置を特定する 。
    """
    local_max = (spectrum == maximum_filter(spectrum, size=neighborhood_size))
    rock_locations = np.argwhere(local_max & (spectrum > 0)).tolist()
    return rock_locations


def plot_spectrum_with_rocks(data: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray,
                             output_dir: str, output_name: str, rock_locations: list = None,
                             downsample_factor: int = 1, x_range_plot: tuple = None):
    """
    スペクトルデータと岩石位置をプロットする関数。
    ダウンサンプリングとX軸範囲指定のオプションを追加。
    """
    # データのダウンサンプリング
    if downsample_factor > 1:
        data_display = data[::downsample_factor, ::downsample_factor]
        x_axis_display = x_axis[::downsample_factor]
        z_axis_display = z_axis[::downsample_factor]
    else:
        data_display = data
        x_axis_display = x_axis
        z_axis_display = z_axis

    fig, ax = plt.subplots(figsize=(18, 6))

    # X軸範囲の調整
    if x_range_plot is not None:
        x_min_plot, x_max_plot = x_range_plot
        col_idx_min = np.searchsorted(x_axis_display, x_min_plot, side='left')
        col_idx_max = np.searchsorted(x_axis_display, x_max_plot, side='right')
        
        data_display = data_display[:, col_idx_min:col_idx_max]
        x_axis_display = x_axis_display[col_idx_min:col_idx_max]
        
        extent = [x_axis_display.min(), x_axis_display.max(), z_axis_display.max(), z_axis_display.min()]
    else:
        extent = [x_axis_display.min(), x_axis_display.max(), z_axis_display.max(), z_axis_display.min()]

    im = ax.imshow(data_display, aspect='auto', cmap='viridis', # cmapをviridisに変更（相関値に適したカラーマップ）
                        extent=extent,
                        origin='lower')

    ax.set_xlabel('Distance (m)', fontsize=20)
    ax.set_ylabel('Depth (m)', fontsize=20)
    ax.tick_params(labelsize=18)

    # 岩石の位置をオーバーレイ
    if rock_locations:
        rock_x_coords = []
        rock_z_coords = []
        for loc in rock_locations:
            current_x = loc[1] * x_axis[1] 
            if x_range_plot is not None and (current_x < x_min_plot or current_x > x_max_plot):
                continue
            rock_x_coords.append(current_x)
            rock_z_coords.append(loc[0] * z_axis[1]) 
        
        if rock_x_coords:
            ax.plot(rock_x_coords, rock_z_coords, 'ro', markersize=5, alpha=0.7)

    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Correlation', fontsize=20) # ラベルをSimilarityからCorrelationに変更
    cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_name + '.pdf'), dpi=300)
    plt.close(fig)


def main():
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
        ["/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2A/Processed_Data/4_Gain_function/4_Bscan_gain.txt",
            "/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Processed_Data/4_Gain_function/4_Bscan_gain.txt"],
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
    base_dir = "/Volumes/SSD_Kanda_SAMSUNG/LPR/Local_similarity/local_normalized_crosscorr"
    if data_type == '1':
        output_dir = os.path.join(base_dir, '4_Gain_function')
    else:
        output_dir = os.path.join(base_dir, '5_Terrain_correction')
    os.makedirs(output_dir, exist_ok=True)

    # Local Similarityスペクトルの計算
    print("Calculating Local Similarity Spectrum (fast version using Local Normalized Cross-Correlation)...")
    # window_size_m をタプルで渡す (例: 0.2m x 0.2m のウィンドウ)
    # mean_center_correlation を True/False で指定
    similarity_spectrum = calculate_local_similarity_spectrum_fast(
        data1, data2, dx, dz, 
        window_size_m=(2.0, 2.0), # ここを調整: 例として0.2m x 0.2mに設定
        mean_center_correlation=False, # ピアソン相関を維持
        debug_pixel_rows=debug_pixel_rows, 
        debug_pixel_cols=debug_pixel_cols
    )
    np.savetxt(os.path.join(output_dir, 'local_correlation_spectrum_raw.txt'), similarity_spectrum, fmt='%.6f')
    print("Finished")
    print(" ")
    
    # Local Similarityスペクトルの統計情報を表示 (閾値調整の参考に)
    print("Correlation Spectrum Statistics:")
    print(f"  Min: {np.min(similarity_spectrum):.6f}")
    print(f"  Max: {np.max(similarity_spectrum):.6f}")
    print(f"  Mean: {np.mean(similarity_spectrum):.6f}")
    print(f"  Std Dev: {np.std(similarity_spectrum):.6f}")
    print("Please examine the 'local_correlation_raw.pdf' plot to determine an appropriate soft_threshold_value.")
    print(" ")

    # Local Similarityスペクトル全体に平滑化を適用
    if sigma_smoothing > 0:
        print("Applying Gaussian smoothing to correlation spectrum...")
        similarity_spectrum_smoothed = apply_gaussian_smoothing(similarity_spectrum, sigma=sigma_smoothing)
        np.savetxt(os.path.join(output_dir, 'local_correlation_spectrum_smoothed.txt'), similarity_spectrum_smoothed, fmt='%.6f')
        print("Finished")
        print(" ")
    else:
        similarity_spectrum_smoothed = similarity_spectrum # 平滑化しない場合はそのまま次のステップへ

    # ソフト閾値関数の適用
    print(f"Applying soft threshold with value: {soft_threshold_value}...")
    thresholded_spectrum = apply_soft_threshold(similarity_spectrum_smoothed, soft_threshold_value)
    np.savetxt(os.path.join(output_dir, 'local_correlation_spectrum_thresholded.txt'), thresholded_spectrum, fmt='%.6f')
    print("Finished")
    print(" ")

    # 反射領域のミュート (オプション)
    print("Muting reflection area...")
    if reflection_mask is not None:
        muted_spectrum = mute_reflection_area(thresholded_spectrum, reflection_mask)
    else:
        muted_spectrum = thresholded_spectrum
    np.savetxt(os.path.join(output_dir, 'local_correlation_spectrum_muted.txt'), muted_spectrum, fmt='%.6f')
    print("Finished")
    print(" ")

    # 局所最大値の抽出
    print("Extracting local maximums for rock locations...")
    rock_locations = extract_local_maximums(muted_spectrum, local_max_neighborhood)
    print(f"Detected {len(rock_locations)} rock locations.")
    rock_locations_file = os.path.join(output_dir, 'rock_locations.txt')
    with open(rock_locations_file, 'w') as f:
        for loc in rock_locations:
            f.write(f"{loc[0]} {loc[1]}\n")
    print("Finished")
    print(" ")

    # Plotting results
    print("Plotting results...")
    plot_downsample_factor = 1

    plot_spectrum_with_rocks(similarity_spectrum, x_axis, z_axis, output_dir, 'local_correlation_raw',
                             rock_locations=None, downsample_factor=plot_downsample_factor)
    
    plot_spectrum_with_rocks(similarity_spectrum, x_axis, z_axis, output_dir, 'local_correlation_raw_with_rocks',
                             rock_locations=rock_locations, downsample_factor=plot_downsample_factor)

    if sigma_smoothing > 0:
        plot_spectrum_with_rocks(similarity_spectrum_smoothed, x_axis, z_axis, output_dir, 'local_correlation_smoothed',
                                 rock_locations=None, downsample_factor=plot_downsample_factor)


    plot_spectrum_with_rocks(thresholded_spectrum, x_axis, z_axis, output_dir, 'local_correlation_thresholded',
                             rock_locations=rock_locations, downsample_factor=plot_downsample_factor)

    if reflection_mask is not None:
        plot_spectrum_with_rocks(muted_spectrum, x_axis, z_axis, output_dir, 'local_correlation_muted',
                                 rock_locations=rock_locations, downsample_factor=plot_downsample_factor)
    print("Plotting finished.")

if __name__ == "__main__":
    main()