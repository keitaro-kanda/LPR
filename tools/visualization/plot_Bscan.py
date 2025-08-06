"""
This code makes B-scan plots from resampled ECHO data.
If you want to make B-scan plot of each sequence, you can use resampling.py.
"""
# 1. Imports (standardized order)
import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm
from scipy import signal


# 2. Interactive input section
data_path = input('データファイルのパスを入力してください: ').strip()
if not os.path.exists(data_path):
    print('エラー: 指定されたファイルが存在しません')
    exit(1)
if not data_path.lower().endswith('.txt'):
    print("エラー: B-scanデータはテキストファイル(.txt)を指定してください。")
    exit(1)

print('データの種類を選択してください:')
print('1: Raw data')
print('2: Bandpass filtered')
print('3: Time-zero corrected')
print('4: Background removed')
print('5: Gain corrected')
print('6: Terrain corrected')
data_type_choice = input('選択 (1-6): ').strip()

data_type_map = {
    '1': 'raw',
    '2': 'bandpass_filtered',
    '3': 'time_zero_corrected',
    '4': 'background_filtered',
    '5': 'gained',
    '6': 'terrain_corrected'
}

if data_type_choice not in data_type_map:
    print('エラー: 無効な選択です。1-6の数字を入力してください。')
    exit(1)

data_type = data_type_map[data_type_choice]
print(f'選択されたデータ種類: {data_type}')

print('エンベロープを計算しますか？（y/n、デフォルト：n）:')
envelope_option = input().strip().lower()
use_envelope = envelope_option.startswith('y') if envelope_option else False

print('プロット範囲を限定しますか？（y/n、デフォルト：n）:')
use_plot_range = input().strip().lower().startswith('y')
plot_range = None
if use_plot_range:
    print('プロット範囲を入力してください（x_start x_end y_start y_end）[m, m, ns, ns]:')
    try:
        x0, x1, y0, y1 = map(float, input().split())
        plot_range = [x0, x1, y0, y1]
    except:
        print('無効な形式のため範囲設定を無効化します。')
        use_plot_range = False
        plot_range = None



# 3. Parameter definitions
sample_interval = 0.312500e-9  # [s] - Time sampling interval
trace_interval = 3.6e-2       # [m] - Spatial trace interval
c = 299792458                 # [m/s] - Speed of light
epsilon_r = 4.5              # Relative permittivity of lunar regolith
reciever_time_delay = 28.203e-9  # [s] - Hardware delay


# 4. Path validation and directory creation
# Already done above

# 5. Output directory setup
output_dir = os.path.join(os.path.dirname(data_path), 'Trimmed_plot_selected_range')
if use_envelope:
    output_dir = os.path.join(output_dir, 'envelope')

os.makedirs(output_dir, exist_ok=True)


# 6. Main processing functions
def calculate_envelope(data):
    """Calculate envelope of the data using Hilbert transform"""
    envelope_data = np.abs(signal.hilbert(data, axis=0))
    return envelope_data


def find_time_zero(data):
    """Find time zero based on first non-NaN value at x=0"""
    first_trace = data[:, 0]
    first_non_nan_idx = np.where(~np.isnan(first_trace))[0]
    if len(first_non_nan_idx) > 0:
        return first_non_nan_idx[0]
    return 0


# Font size standards
font_large = 20      # Titles and labels
font_medium = 18     # Axis labels  
font_small = 16      # Tick labels

def single_plot(plot_data, time_zero_idx):
    """Create B-scan plot with dual y-axis and bottom-right colorbar"""
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(111)
    
    # Calculate time array with time-zero correction
    time_array = (np.arange(plot_data.shape[0]) - time_zero_idx) * sample_interval * 1e9  # [ns]
    
    # データタイプに応じた設定 (run_data_processing.pyと同じロジック)
    if data_type == 'gained':  # Step 4: Gain function
        if use_envelope:
            vmin = 0
            vmax = np.nanmax(np.abs(plot_data))/10
        else:
            vmin = -np.nanmax(np.abs(plot_data))/10
            vmax = np.nanmax(np.abs(plot_data))/10
    elif data_type == 'terrain_corrected':  # Step 5: Terrain correction  
        if use_envelope:
            vmin = 0
            vmax = np.nanmax(np.abs(plot_data))/10
        else:
            vmin = -np.nanmax(np.abs(plot_data))/10
            vmax = np.nanmax(np.abs(plot_data))/10
    else:  # Steps 0-3: Raw, Bandpass, Time-zero, Background
        if use_envelope:
            vmin = 0
            vmax = 10
        else:
            vmin = -10
            vmax = 10
    
    # Create the main plot
    im = ax.imshow(plot_data, aspect='auto', cmap='viridis',
                   extent=[0, plot_data.shape[1]*trace_interval, 
                          time_array[-1], time_array[0]],
                   vmin=vmin, vmax=vmax)
    
    # プロット範囲の設定
    if use_plot_range:
        ax.set_xlim(plot_range[0], plot_range[1])
        ax.set_ylim(plot_range[3], plot_range[2])  # y軸は逆転しているため、順序を入れ替え
    
    ax.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax.set_ylabel('Time [ns]', fontsize=font_medium)
    ax.tick_params(axis='both', which='major', labelsize=font_small)

    # Add the second Y-axis for depth
    ax2 = ax.twinx()
    # Get the range of the original Y-axis (time)
    t_min, t_max = ax.get_ylim()
    # Calculate the corresponding depth range
    # depth [m] = (time [ns] * 1e-9) * v_gpr [m/s] / 2
    depth_min = (t_min * 1e-9) * c / np.sqrt(epsilon_r) / 2
    depth_max = (t_max * 1e-9) * c / np.sqrt(epsilon_r) / 2
    # Set depth range and labels for new Y-axis
    ax2.set_ylim(depth_min, depth_max)
    ax2.set_ylabel(r'Depth [m] ($\varepsilon_r = 4.5$)', fontsize=font_medium)
    ax2.tick_params(axis='y', which='major', labelsize=font_small)

    # Adjust layout for colorbar
    fig.subplots_adjust(bottom=0.18, right=0.9)
    # Add colorbar at bottom right corner
    cbar_ax = fig.add_axes([0.65, 0.05, 0.2, 0.05])  # [x, y, width, height]
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=font_small)

    # Standard save pattern
    if use_plot_range:
        file_base = f'x_{plot_range[0]}_{plot_range[1]}_t{plot_range[2]}_{plot_range[3]}'
    else:
        file_base = os.path.splitext(os.path.basename(data_path))[0]
    output_path_png = os.path.join(output_dir, f'{file_base}.png')
    output_path_pdf = os.path.join(output_dir, f'{file_base}.pdf')
    
    plt.savefig(output_path_png, dpi=120)   # Web quality
    plt.savefig(output_path_pdf, dpi=600)   # Publication quality
    print(f"プロットを保存しました: {output_path_png}")
    plt.show()

    return fig

def save_Bscan_size(data, time_zero_idx):
    """Save B-scan size information to a text file"""
    output_path = os.path.join(os.path.dirname(data_path), 'Bscan_size.txt')
    with open(output_path, 'w') as f:
        f.write(f'B-scan size: {data.shape[0]} x {data.shape[1]}\n')
        f.write(f'Time zero index: {time_zero_idx}\n')
        f.write(f'Sample interval: {sample_interval * 1e9:.2f} ns\n')
        f.write(f'Trace interval: {trace_interval:.2f} m\n')
    print(f"B-scan size information saved to: {output_path}")


# 7. Execution
if __name__ == "__main__":
    print('Loading data...')
    data = np.loadtxt(data_path, delimiter=' ')
    print("B-scanの形状:", data.shape)
    # Save B-scan size information
    save_Bscan_size(data, find_time_zero(data))

    # Store original data for NaN handling before any processing
    original_data = data.copy()

    # Find time zero based on first non-NaN value at x=0
    time_zero_idx = find_time_zero(data)
    print(f'Time zero index: {time_zero_idx}')
    print(f'Time zero: {time_zero_idx * sample_interval * 1e9:.2f} ns')


    if use_envelope:
        print('エンベロープを計算中...')
        data = calculate_envelope(data)

    print('プロット作成中...')
    single_plot(data, time_zero_idx)