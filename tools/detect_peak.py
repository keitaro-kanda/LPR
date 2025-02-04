import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
# import argparse
from tqdm import tqdm
import gc
from scipy.signal import hilbert



#* Get input parameters
print('データファイルのパスを入力してください:')
data_path = input().strip()

if not os.path.exists(data_path):
    print('エラー: 指定されたファイルが存在しません')
    exit(1)

#* Define the output directory
output_dir = os.path.join(os.path.dirname(data_path), 'detect_peak')
os.makedirs(output_dir, exist_ok=True)

output_dir_trim_png = os.path.join(output_dir, 'trim/png')
os.makedirs(output_dir_trim_png, exist_ok=True)
output_dir_trim_pdf = os.path.join(output_dir, 'trim/pdf')
os.makedirs(output_dir_trim_pdf, exist_ok=True)


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


#* Detect the peak
print('Detecting peaks...')
echo_info = []

for trace_idx in tqdm(range(data.shape[1]), desc='Detecting peaks'):
    Ascan = data[:, trace_idx]
    envelope = np.abs(hilbert(Ascan))
    peaks_in_Ascan = []
    for i in range(1, Ascan.shape[0]-1):  # 境界を避けるため1からスタート
        #* Detect local maxima of the envelope
        if envelope[i-1] < envelope[i] > envelope[i+1] and i * sample_interval * 1e9 > 20 and envelope[i] > 1000:  # 20ns以降のピークのみを検出
            peaks_in_Ascan.append(i)

    #* Calculate the FWHM
    for i, peak_idx in enumerate(peaks_in_Ascan):
        peak_amplitude = envelope[peak_idx]
        half_amplitude = peak_amplitude / 2

        # 左側の半値位置を探索
        left_idx = peak_idx
        while left_idx > 0 and envelope[left_idx] > half_amplitude:
            left_idx -= 1

        if left_idx == 0:
            left_half_time = time[0]
        else:
            # 線形補間で正確な半値位置を求める
            left_slope = (envelope[left_idx + 1] - envelope[left_idx]) / (time[left_idx + 1] - time[left_idx])
            left_half_time = time[left_idx] + (half_amplitude - envelope[left_idx]) / left_slope # [ns]

        # 右側の半値位置を探索
        right_idx = peak_idx
        while right_idx < len(envelope) - 1 and envelope[right_idx] > half_amplitude:
            right_idx += 1

        if right_idx == len(envelope) - 1:
            right_half_time = time[-1]
        else:
            # 線形補間で正確な半値位置を求める
            right_slope = (envelope[right_idx] - envelope[right_idx - 1]) / (time[right_idx] - time[right_idx - 1])
            right_half_time = time[right_idx - 1] + (half_amplitude - envelope[right_idx - 1]) / right_slope # [ns]

        # 半値全幅を計算
        #whm = np.min([np.abs(time[peak_idx] - left_half_time), np.abs(time[peak_idx] - right_half_time)]) # [ns], Half width at half maximum
        #fwhm = hwhm * 2 # [ns], Full width at half maximum
        fwhm = right_half_time - left_half_time # [ns], Full width at half maximum



        #* Investigate whether two peaks can be separated or not
        separation_next = None
        separation_prev = None
        distinguishable_prev = True # デフォルトはTrue
        distinguishable_next = True # デフォルトはTrue
        distinguishable = True  # デフォルトはTrue

        # 前のピークとの時間差を計算
        if i > 0:
            prev_peak_idx = peaks_in_Ascan[i - 1]
            separation_prev = time[peak_idx] - time[prev_peak_idx]
            if separation_prev < fwhm:
                distinguishable_prev = False

        # 次のピークとの時間差を計算
        if i < len(peaks_in_Ascan) - 1:
            next_peak_idx = peaks_in_Ascan[i + 1]
            separation_next = time[next_peak_idx] - time[peak_idx]
            if separation_next < fwhm:
                distinguishable_next = False

        # 孤立したピーク（前後にピークがない場合）
        if i == 0 and len(peaks_in_Ascan) == 1:
            separation_prev = None
            separation_next = None
            distinguishable_prev = True
            distinguishable_next = True

        #* Determine whether the peak is distinguishable or not
        if distinguishable_prev and distinguishable_next:
            distinguishable = True
        elif distinguishable_prev and not distinguishable_next:
            if envelope[peak_idx] > envelope[next_peak_idx]:
                distinguishable = True
            else:
                distinguishable = False
        elif not distinguishable_prev and distinguishable_next:
            if envelope[peak_idx] > envelope[prev_peak_idx]:
                distinguishable = True
            else:
                distinguishable = False
        else:
            distinguishable = False

        separation = min(separation_prev, separation_next) if separation_prev is not None and separation_next is not None else (separation_prev or separation_next)



        #* Detect maximum peak in the echo
        data_segment = np.abs(Ascan[int(left_half_time*1e-9/sample_interval):int(right_half_time*1e-9/sample_interval)])
        if len(data_segment) > 0:
            local_max_idxs = []
            local_max_amps = []
            for j in range(1, len(data_segment) - 1):
                if data_segment[j - 1] < data_segment[j] > data_segment[j + 1]:
                    local_max_idxs.append(j)
                    local_max_amps.append(data_segment[j])

            if len(local_max_idxs) >= 2:
                # 振幅の降順でソート
                sorted_indices = np.argsort(local_max_amps)[::-1]
                primary_max_idx = local_max_idxs[sorted_indices[0]]
            elif len(local_max_idxs) == 1:
                primary_max_idx = local_max_idxs[0]
            else:
                primary_max_idx = np.argmax(np.abs(data_segment))

            max_idx = int(left_half_time*1e-9/sample_interval) + primary_max_idx
            max_time = time[max_idx]
            max_amplitude = Ascan[max_idx]

        else:
            max_idx = peak_idx
            max_time = time[peak_idx]
            max_amplitude = Ascan[peak_idx]

        echo_info.append({
            # 'x_idx': j,
            'x': trace_idx * trace_interval,
            # 'env_peak_idx': peak_idx,
            # 'env_peak_time': time[peak_idx],
            # 'env_peak_value': peak_amplitude,
            # 'FWHM': fwhm,
            # 'distinguishable': distinguishable,
            # 'amp_max_idx': max_idx,
            'amp_max_time': max_time,
            'amp_max_value': max_amplitude
        })

peak_x_t_values = np.array([[info['x'], info['amp_max_time'], info['amp_max_value']] for info in echo_info]) # [m], [ns], [Amplitude]
np.savetxt(os.path.join(output_dir, 'peak_x_t_values.txt'), peak_x_t_values, fmt='%.3f')
print('Peak shape:', peak_x_t_values.shape)
print('Finished detecting peaks.')
print(' ')



#* Define the function to plot
def plot(Bscan_data, scatter_data, x1, x2, y1, y2):
    fig = plt.figure(figsize=(18, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    im = ax.imshow(Bscan_data, aspect='auto', cmap='gray',
                    extent=[x1, x2, y2, y1],
                    vmin=-3000, vmax=3000
                    )
    scatter = ax.scatter(scatter_data[:, 0], scatter_data[:, 1], # +50 to compensate the trim
                        c=scatter_data[:, 2], cmap='seismic', s=1,
                        #vmin = -scatter_max/5, vmax = scatter_max/5
                        vmin = -3000, vmax = 3000)


    #* Set labels
    ax.set_xlabel('Distance [m]', fontsize=24)
    ax.set_ylabel('Time [ns]', fontsize=24)
    ax.tick_params(labelsize=20)

    ax.grid(which='both', axis='both', linestyle='-.', color='white')


    delvider = axgrid1.make_axes_locatable(ax)
    cax_im = delvider.append_axes('right', size='3%', pad=0.1)
    cbar_im = plt.colorbar(im, cax=cax_im, orientation = 'vertical')
    cbar_im.ax.tick_params(labelsize=20)

    cax_scatter = delvider.append_axes('right', size='3%', pad=1.5)
    cbar_scatter = plt.colorbar(scatter, cax=cax_scatter, orientation = 'vertical')
    cbar_scatter.ax.tick_params(labelsize=20)


    filename_base = f'/x{x1}_y{y1}'
    fig.savefig(output_dir_trim_png + filename_base + '.png', format='png', dpi=120)
    fig.savefig(output_dir_trim_pdf + filename_base + '.pdf', format='pdf', dpi=300)

    plt.close()
    gc.collect()  # Garbage collection to avoid memory error

    return plt


def plot_grid(data, peak_values, output_dir_trim_png, output_dir_trim_pdf,
                grid_size=100, trace_interval=3.6e-2, sample_interval=0.312500e-9):
    """グリッド状にB-scanデータを分割してプロット
    Args:
        data: B-scanデータ配列
        peak_values: ピーク検出結果の配列 (x, time, amplitude)
        grid_size: 分割グリッドのサイズ（デフォルト: 100）, [m]にも[ns]にも使える
    """
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

                # ピークデータのフィルタリング
                mask = ((peak_values[:, 0] >= x_start) & (peak_values[:, 0] < x_end) &
                       (peak_values[:, 1] >= y_start) & (peak_values[:, 1] < y_end))
                peak_values_trim = peak_values[mask]

                if len(peak_values_trim) > 0:  # ピークが存在する場合のみプロット
                    plot(data_trim, peak_values_trim, x_start, x_end, y_start, y_end)
                pbar.update(1)

#* Plot
print('プロット作成開始...')
plot_grid(data, peak_x_t_values, output_dir_trim_png, output_dir_trim_pdf)
