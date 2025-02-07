import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
from tqdm import tqdm
import gc
from scipy.ndimage import median_filter as medfilt



#* Define the median filter function
def median_filter(data, kernel_size_x, kernel_size_t):
    """Median filter for 2D array
    Args:
        data: 2D array
        kernel_size: size of the kernel
    Returns:
        data_filtered: 2D array
    """
    data_filtered = np.zeros_like(data)
    for i in tqdm(range(data.shape[1]), desc='Applying median filter'):
        for j in range(data.shape[0]):
            i_start = max(0, i - kernel_size_x)
            i_end = min(data.shape[1], i + kernel_size_x + 1)
            j_start = max(0, j - kernel_size_t)
            j_end = min(data.shape[0], j + kernel_size_t + 1)

            data_filtered[j, i] = np.median(data[j_start:j_end, i_start:i_end])

    return data_filtered

#* Difine the function to plot the data
def plot(Bscan_data, max_value):
    fig = plt.figure(figsize=(18, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    im = ax.imshow(Bscan_data, aspect='auto', cmap='viridis',
                    extent=[0, Bscan_data.shape[1]*trace_interval, time[-1], time[0]],
                    vmin=-max_value/7, vmax=max_value/7
                    )

    #* Set labels
    ax.set_xlabel('Distance [m]', fontsize=24)
    ax.set_ylabel('Time [ns]', fontsize=24)
    ax.tick_params(labelsize=20)

    ax.grid(which='both', axis='both', linestyle='-.', color='white')


    delvider = axgrid1.make_axes_locatable(ax)
    cax_im = delvider.append_axes('right', size='3%', pad=0.1)
    cbar_im = plt.colorbar(im, cax=cax_im, orientation = 'vertical')
    cbar_im.ax.tick_params(labelsize=20)

    plt.savefig(output_dir + '/filtered_data.png', format='png', dpi=120)
    plt.savefig(output_dir + '/filtered_data.pdf', format='pdf', dpi=300)
    plt.close()

    return plt


#* Define the function to plot the trimmed data
def plot_trim(Bscan_data, x1, x2, y1, y2, max_value):
    fig = plt.figure(figsize=(18, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    im = ax.imshow(Bscan_data, aspect='auto', cmap='viridis',
                    extent=[x1, x2, y2, y1],
                    vmin=-max_value/7, vmax=max_value/7
                    )

    #* Set labels
    ax.set_xlabel('Distance [m]', fontsize=24)
    ax.set_ylabel('Time [ns]', fontsize=24)
    ax.tick_params(labelsize=20)

    ax.grid(which='both', axis='both', linestyle='-.', color='white')


    delvider = axgrid1.make_axes_locatable(ax)
    cax_im = delvider.append_axes('right', size='3%', pad=0.1)
    cbar_im = plt.colorbar(im, cax=cax_im, orientation = 'vertical')
    cbar_im.ax.tick_params(labelsize=20)


    filename_base = f'/x{x1}_y{y1}'
    fig.savefig(output_dir_trim_png + filename_base + '.png', format='png', dpi=120)
    fig.savefig(output_dir_trim_pdf + filename_base + '.pdf', format='pdf', dpi=300)

    plt.close()
    gc.collect()  # Garbage collection to avoid memory error

    return plt


def run_plot_trim(data, output_dir_trim_png, output_dir_trim_pdf, max_value):
    """グリッド状にB-scanデータを分割してプロット
    Args:
        data: B-scanデータ配列
        grid_size: 分割グリッドのサイズ（デフォルト: 100）, [m]にも[ns]にも使える
    """
    grid_size=100
    x_max = data.shape[1] * trace_interval  # [m]
    y_max = data.shape[0] * sample_interval / 1e-9  # [ns]

    # グリッド開始位置の配列を生成
    x_starts = np.arange(0, x_max, grid_size)  # [m]
    y_starts = np.arange(0, y_max, grid_size)  # [ns]

    total_plots = len(x_starts) * len(y_starts)
    with tqdm(total=total_plots, desc='グリッドプロット作成中') as pbar:
        for x_start in x_starts:
            x_end = min(x_max, x_start + grid_size) # [m]
            # インデックスに変換
            x_start_idx = int(x_start/trace_interval)
            x_end_idx = int(x_end/trace_interval)
            x_slice = slice(x_start_idx, x_end_idx)

            for y_start in y_starts:
                y_end = min(y_max, y_start + grid_size) # [ns]
                # インデックスに変換
                y_start_idx = int(y_start/(sample_interval*1e9))
                y_end_idx = int(y_end/(sample_interval*1e9))
                y_slice = slice(y_start_idx, y_end_idx)

                # データのトリミング
                data_trim = data[y_slice, x_slice]
                plot_trim(data_trim, x_start, x_end, y_start, y_end, max_value)
                pbar.update(1)



#* Main
#* Get input parameters
print('データファイルのパスを入力してください:')
data_path = input().strip()

if not os.path.exists(data_path):
    print('エラー: 指定されたファイルが存在しません')
    exit(1)


#* Parameters
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]


#* Load data
print('Loading data...')
data = np.loadtxt(data_path, delimiter=' ')
time = np.arange(data.shape[0]) * sample_interval / 1e-9 # [ns]
print('Data shape:', data.shape)
print('Finished loading data.')
print(' ')


#* Median filter
print('Meadian filterのカーネルサイズを入力してください（x方向）:')
kernel_size_x = int(input().strip())
print('Meadian filterのカーネルサイズを入力してください（t方向）:')
kernel_size_t = int(input().strip())
#* Define the output directory
output_dir = os.path.join(os.path.dirname(data_path), f'median_filter/kernel_x{kernel_size_x}_t{kernel_size_t}')
os.makedirs(output_dir, exist_ok=True)

print('Applying median filter...')
data_filtered = median_filter(data, kernel_size_x, kernel_size_t)

#* Save the filtered data
output_path = os.path.join(output_dir, 'filtered_data.txt')
np.savetxt(output_path, data_filtered, delimiter=' ')
print('Finished applying median filter.')
print(' ')

max_value = np.amax(np.abs(data_filtered))


output_dir_trim_png = os.path.join(output_dir, 'trim_png')
os.makedirs(output_dir_trim_png, exist_ok=True)
output_dir_trim_pdf = os.path.join(output_dir, 'trim_pdf')
os.makedirs(output_dir_trim_pdf, exist_ok=True)

#* Plot the filtered data
print('Plotting the filtered data...')
plot(data_filtered, max_value)
run_plot_trim(data_filtered, output_dir_trim_png, output_dir_trim_pdf, max_value)
print('Finished plotting the filtered data.')



