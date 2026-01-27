import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm
from scipy import signal
from scipy import ndimage


# Interactive input section
print('データファイルのパスを入力してください:')
data_path = input().strip()

print('メディアンフィルターのカーネルサイズを入力してください (例: 5):')
median_kernel_size = int(input().strip())

print('処理順序を選択してください:')
print('1: エンベロープ計算 → メディアンフィルタ')
print('2: メディアンフィルタ → エンベロープ計算')
processing_order = input('選択 (1/2): ').strip()

# Parameter definitions
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2        # [m]
c = 299792458                  # [m/s]
epsilon_r = 4.5               # Relative permittivity

# Path validation and directory creation
if not os.path.exists(data_path):
    print('エラー: 指定されたファイルが存在しません')
    exit(1)

# Validate processing order input
if processing_order not in ['1', '2']:
    print('エラー: 無効な選択です。1または2を選択してください。')
    exit(1)

# Output directory setup with hierarchical structure
base_output_dir = os.path.join(os.path.dirname(data_path), 'envelope_median_filter')

# Create processing order subdirectory
if processing_order == '1':
    order_dir_name = 'envelope_first'
    order_description = 'envelope_median'
else:
    order_dir_name = 'median_first'
    order_description = 'median_envelope'

# Create kernel size subdirectory
kernel_dir_name = f'kernel_{median_kernel_size}'

# Complete directory path: envelope_median_filter/envelope_first/kernel_5/
output_dir = os.path.join(base_output_dir, order_dir_name, kernel_dir_name)
os.makedirs(output_dir, exist_ok=True)

print(f'出力ディレクトリ: {output_dir}')

# Load data
print('Loading data...')
Bscan_data = np.loadtxt(data_path, delimiter=' ')
print(f'データ形状: {Bscan_data.shape}')

# NaN value handling
nan_count = np.sum(np.isnan(Bscan_data))
total_count = Bscan_data.size
if nan_count > 0:
    print(f'NaN値検出: {nan_count} / {total_count} ({nan_count/total_count*100:.2f}%)')
    # Handle NaN values
    Bscan_data_clean = np.nan_to_num(Bscan_data, nan=0.0, posinf=0.0, neginf=0.0)
else:
    print('NaN値は検出されませんでした。')
    Bscan_data_clean = Bscan_data.copy()


def calculate_envelope(data):
    """
    ヒルベルト変換を使用してエンベロープを計算
    
    Parameters:
    -----------
    data : np.ndarray
        入力B-scanデータ (time x traces)
    
    Returns:
    --------
    envelope : np.ndarray
        エンベロープデータ
    """
    print('エンベロープを計算中...')
    envelope = np.zeros_like(data)
    
    for i in tqdm(range(data.shape[1]), desc='Calculating envelope'):
        # 各トレースに対してヒルベルト変換を適用
        analytic_signal = signal.hilbert(data[:, i])
        envelope[:, i] = np.abs(analytic_signal)
    
    return envelope


def apply_median_filter(data, kernel_size):
    """
    メディアンフィルターを適用
    
    Parameters:
    -----------
    data : np.ndarray
        入力データ
    kernel_size : int
        メディアンフィルターのカーネルサイズ
    
    Returns:
    --------
    filtered_data : np.ndarray
        フィルター処理後のデータ
    """
    print(f'メディアンフィルター (カーネルサイズ: {kernel_size}) を適用中...')
    filtered_data = ndimage.median_filter(data, size=kernel_size)
    return filtered_data


def plot_zoomed_region(original_data, intermediate_data, final_data, x_min, t_min, x_max, t_max,
                      vmax_original, vmax_intermediate, vmax_final,
                      output_path, processing_order, region_name):
    """
    指定された領域の拡大プロットを生成
    
    Parameters:
    -----------
    original_data : np.ndarray
        元のB-scanデータ
    intermediate_data : np.ndarray
        中間処理データ
    final_data : np.ndarray
        最終処理データ
    x_min, t_min, x_max, t_max : float
        拡大領域の境界 [m], [ns]
    output_path : str
        出力パス
    processing_order : str
        処理順序
    region_name : str
        領域名（ファイル名用）
    """
    # Font size standards
    font_large = 20      # Titles and labels
    font_medium = 18     # Axis labels  
    font_small = 16      # Tick labels
    
    # Convert physical units to array indices
    x_min_idx = int(x_min / trace_interval)
    x_max_idx = int(x_max / trace_interval)
    t_min_idx = int((t_min * 1e-9) / sample_interval)
    t_max_idx = int((t_max * 1e-9) / sample_interval)
    
    # Ensure indices are within data bounds
    x_min_idx = max(0, min(x_min_idx, original_data.shape[1] - 1))
    x_max_idx = max(0, min(x_max_idx, original_data.shape[1] - 1))
    t_min_idx = max(0, min(t_min_idx, original_data.shape[0] - 1))
    t_max_idx = max(0, min(t_max_idx, original_data.shape[0] - 1))
    
    # Extract zoomed regions
    zoom_original = original_data[t_min_idx:t_max_idx, x_min_idx:x_max_idx]
    zoom_intermediate = intermediate_data[t_min_idx:t_max_idx, x_min_idx:x_max_idx]
    zoom_final = final_data[t_min_idx:t_max_idx, x_min_idx:x_max_idx]
    
    # Handle NaN values for plotting
    plot_original = np.nan_to_num(zoom_original, nan=0.0)
    plot_intermediate = np.nan_to_num(zoom_intermediate, nan=0.0)
    plot_final = np.nan_to_num(zoom_final, nan=0.0)
    
    # Color scale settings
    # vmax_original = np.nanmax(np.abs(plot_original)) / 5
    # vmax_intermediate = np.nanmax(np.abs(plot_intermediate)) / 5
    # vmax_final = np.nanmax(np.abs(plot_final)) / 5
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), tight_layout=True, sharex=True)
    
    # Plot original data
    im1 = axes[0].imshow(plot_original, aspect='auto', cmap='seismic',
                        extent=[x_min, x_max, t_max, t_min],
                        vmin=-vmax_original, vmax=vmax_original)
    axes[0].set_title(f'Original B-scan Data (Region: {region_name})', fontsize=font_large)
    axes[0].set_ylabel('Time [ns]', fontsize=font_medium)
    axes[0].tick_params(labelsize=font_small)
    
    # Colorbar for original data
    divider1 = axgrid1.make_axes_locatable(axes[0])
    cax1 = divider1.append_axes('right', size='5%', pad=0.1)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label('Amplitude', fontsize=font_medium)
    cbar1.ax.tick_params(labelsize=font_small)
    
    # Set titles and colormaps based on processing order
    if processing_order == '1':
        # Envelope → Median Filter
        intermediate_title = f'Envelope Data (Region: {region_name})'
        final_title = f'Median Filtered Envelope (Region: {region_name})'
        intermediate_cmap = 'viridis'
        final_cmap = 'viridis'
        intermediate_vmin = 0
        final_vmin = 0
    else:
        # Median Filter → Envelope
        intermediate_title = f'Median Filtered Data (Region: {region_name})'
        final_title = f'Envelope of Median Filtered Data (Region: {region_name})'
        intermediate_cmap = 'viridis'
        final_cmap = 'viridis'
        intermediate_vmin = 0
        final_vmin = 0
    
    # Plot intermediate data
    im2 = axes[1].imshow(plot_intermediate, aspect='auto', cmap=intermediate_cmap,
                        extent=[x_min, x_max, t_max, t_min],
                        vmin=intermediate_vmin, vmax=vmax_intermediate)
    axes[1].set_title(intermediate_title, fontsize=font_large)
    axes[1].set_ylabel('Time [ns]', fontsize=font_medium)
    axes[1].tick_params(labelsize=font_small)
    
    # Colorbar for intermediate data
    divider2 = axgrid1.make_axes_locatable(axes[1])
    cax2 = divider2.append_axes('right', size='5%', pad=0.1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label('Amplitude', fontsize=font_medium)
    cbar2.ax.tick_params(labelsize=font_small)
    
    # Plot final data
    im3 = axes[2].imshow(plot_final, aspect='auto', cmap=final_cmap,
                        extent=[x_min, x_max, t_max, t_min],
                        vmin=final_vmin, vmax=vmax_final)
    axes[2].set_title(final_title, fontsize=font_large)
    axes[2].set_ylabel('Time [ns]', fontsize=font_medium)
    axes[2].set_xlabel('Distance [m]', fontsize=font_medium)
    axes[2].tick_params(labelsize=font_small)
    
    # Colorbar for final data
    divider3 = axgrid1.make_axes_locatable(axes[2])
    cax3 = divider3.append_axes('right', size='5%', pad=0.1)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar3.set_label('Amplitude', fontsize=font_medium)
    cbar3.ax.tick_params(labelsize=font_small)
    
    # Standard save pattern
    plt.savefig(f'{output_path}.png', dpi=120)   # Web quality
    plt.savefig(f'{output_path}.pdf', dpi=600)   # Publication quality
    plt.close()  # Close to save memory


def plot_comparison(original_data, intermediate_data, final_data, output_path, processing_order):
    """
    元データ、中間処理データ、最終処理データの比較プロット
    """
    # Font size standards
    font_large = 20      # Titles and labels
    font_medium = 18     # Axis labels  
    font_small = 16      # Tick labels
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 15), tight_layout=True, sharex=True)
    
    # Prepare data for plotting (handle NaN values)
    plot_original = np.nan_to_num(original_data, nan=0.0)
    plot_intermediate = np.nan_to_num(intermediate_data, nan=0.0)
    plot_final = np.nan_to_num(final_data, nan=0.0)
    
    # Color scale settings
    vmax_original = np.nanmax(np.abs(plot_original)) / 5
    vmax_intermediate = np.nanmax(np.abs(plot_intermediate)) / 5
    vmax_final = np.nanmax(np.abs(plot_final)) / 5
    
    # Plot original data
    im1 = axes[0].imshow(plot_original, aspect='auto', cmap='seismic',
                        extent=[0, original_data.shape[1] * trace_interval,
                               original_data.shape[0] * sample_interval / 1e-9, 0],
                        vmin=-vmax_original, vmax=vmax_original)
    axes[0].set_title('Original B-scan Data', fontsize=font_large)
    axes[0].set_ylabel('Time [ns]', fontsize=font_medium)
    axes[0].tick_params(labelsize=font_small)
    
    # Colorbar for original data
    divider1 = axgrid1.make_axes_locatable(axes[0])
    cax1 = divider1.append_axes('right', size='5%', pad=0.1)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label('Amplitude', fontsize=font_medium)
    cbar1.ax.tick_params(labelsize=font_small)
    
    # Set titles and colormaps based on processing order
    if processing_order == '1':
        # Envelope → Median Filter
        intermediate_title = 'Envelope Data'
        final_title = f'Median Filtered Envelope (kernel size: {median_kernel_size})'
        intermediate_cmap = 'viridis'
        final_cmap = 'viridis'
        intermediate_vmin = 0
        final_vmin = 0
    else:
        # Median Filter → Envelope
        intermediate_title = f'Median Filtered Data (kernel size: {median_kernel_size})'
        final_title = 'Envelope of Median Filtered Data'
        intermediate_vmin = 0
        final_vmin = 0
    
    # Plot intermediate data
    im2 = axes[1].imshow(plot_intermediate, aspect='auto', cmap='viridis',
                        extent=[0, intermediate_data.shape[1] * trace_interval,
                               intermediate_data.shape[0] * sample_interval / 1e-9, 0],
                        vmin=intermediate_vmin, vmax=vmax_intermediate)
    axes[1].set_title(intermediate_title, fontsize=font_large)
    axes[1].set_ylabel('Time [ns]', fontsize=font_medium)
    axes[1].tick_params(labelsize=font_small)
    
    # Colorbar for intermediate data
    divider2 = axgrid1.make_axes_locatable(axes[1])
    cax2 = divider2.append_axes('right', size='5%', pad=0.1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label('Amplitude', fontsize=font_medium)
    cbar2.ax.tick_params(labelsize=font_small)
    
    # Plot final data
    im3 = axes[2].imshow(plot_final, aspect='auto', cmap='viridis',
                        extent=[0, final_data.shape[1] * trace_interval,
                               final_data.shape[0] * sample_interval / 1e-9, 0],
                        vmin=final_vmin, vmax=vmax_final)
    axes[2].set_title(final_title, fontsize=font_large)
    axes[2].set_ylabel('Time [ns]', fontsize=font_medium)
    axes[2].set_xlabel('Distance [m]', fontsize=font_medium)
    axes[2].tick_params(labelsize=font_small)
    
    # Colorbar for final data
    divider3 = axgrid1.make_axes_locatable(axes[2])
    cax3 = divider3.append_axes('right', size='5%', pad=0.1)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar3.set_label('Amplitude', fontsize=font_medium)
    cbar3.ax.tick_params(labelsize=font_small)
    
    # Standard save pattern
    plt.savefig(f'{output_path}.png', dpi=120)   # Web quality
    plt.savefig(f'{output_path}.pdf', dpi=600)   # Publication quality
    plt.show()

    return vmax_original, vmax_intermediate, vmax_final


def main():
    """
    メイン処理関数
    """
    if processing_order == '1':
        # Processing order 1: Envelope → Median Filter
        print('処理順序: エンベロープ計算 → メディアンフィルタ')
        
        # Calculate envelope first
        intermediate_data = calculate_envelope(Bscan_data_clean)
        
        # Apply median filter to envelope
        final_data = apply_median_filter(intermediate_data, median_kernel_size)
        
        # File naming (simplified since directory structure contains the info)
        intermediate_output_path = os.path.join(output_dir, 'envelope_data.txt')
        final_output_path = os.path.join(output_dir, 'envelope_median_filtered.txt')
        plot_output_path = os.path.join(output_dir, 'comparison_plot')
        
        print(f'エンベロープデータを保存: {intermediate_output_path}')
        print(f'メディアンフィルター後データを保存: {final_output_path}')
        
    else:
        # Processing order 2: Median Filter → Envelope
        print('処理順序: メディアンフィルタ → エンベロープ計算')
        
        # Apply median filter first
        intermediate_data = apply_median_filter(Bscan_data_clean, median_kernel_size)
        
        # Calculate envelope of filtered data
        final_data = calculate_envelope(intermediate_data)
        
        # File naming (simplified since directory structure contains the info)
        intermediate_output_path = os.path.join(output_dir, 'median_filtered_data.txt')
        final_output_path = os.path.join(output_dir, 'median_envelope_data.txt')
        plot_output_path = os.path.join(output_dir, 'comparison_plot')
        
        print(f'メディアンフィルター後データを保存: {intermediate_output_path}')
        print(f'エンベロープデータを保存: {final_output_path}')
    
    # Save processed data
    np.savetxt(intermediate_output_path, intermediate_data, delimiter=' ')
    np.savetxt(final_output_path, final_data, delimiter=' ')
    
    # Create comparison plot
    vmax_original, vmax_intermediate, vmax_final = plot_comparison(Bscan_data_clean, intermediate_data, final_data, plot_output_path, processing_order)
    
    print(f'比較プロットを保存: {plot_output_path}.png/.pdf')
    
    # Generate zoomed region plots for predefined areas
    zoom_regions = [
        [300, 60, 450, 160],    # Region 1
        [550, 40, 650, 160],    # Region 2
        [1150, 50, 1250, 150],  # Region 3
        [350, 350, 500, 500],   # Region 4
        [1300, 400, 1450, 550]  # Region 5
    ]
    
    print('拡大プロットを生成中...')
    zoom_output_dir = os.path.join(output_dir, 'zoomed_regions')
    os.makedirs(zoom_output_dir, exist_ok=True)
    
    for i, (x_min, t_min, x_max, t_max) in enumerate(zoom_regions, 1):
        region_name = f'Region_{i}'
        zoom_output_path = os.path.join(zoom_output_dir, f'zoom_{region_name}_x{x_min}-{x_max}_t{t_min}-{t_max}')
        
        print(f'  {region_name}: x={x_min}-{x_max}m, t={t_min}-{t_max}ns')
        
        plot_zoomed_region(
            Bscan_data_clean, intermediate_data, final_data,
            x_min, t_min, x_max, t_max,
            vmax_original, vmax_intermediate, vmax_final,
            zoom_output_path, processing_order, region_name
        )
    
    print(f'拡大プロットを保存: {zoom_output_dir}/')
    print('処理完了!')


if __name__ == "__main__":
    main()