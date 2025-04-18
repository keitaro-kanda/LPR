"""
This code make hall B-scan plot from resampled ECHO data.
If you want to make B-scan plot of each sequence, you can use resampling.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from natsort import natsorted
from scipy import signal


# インタラクティブな入力
print('B-scanデータファイルのパスを入力してください:')
data_path = input().strip()
if not os.path.exists(data_path):
    print('エラー: 指定されたファイルが存在しません')
    exit(1)

print('データの種類を選択してください（raw, bandpass_filtered, time_zero_corrected, background_filtered, gained）:')
data_type = input().strip().lower()
if data_type not in ['raw', 'bandpass_filtered', 'time_zero_corrected', 'background_filtered', 'gained']:
    print('エラー: 無効なデータ種類です')
    exit(1)

print('エンベロープを計算しますか？（y/n、デフォルト：n）:')
envelope_option = input().strip().lower()
use_envelope = envelope_option.startswith('y') if envelope_option else False

print('プロット範囲を限定しますか？（y/n、デフォルト：n）:')
plot_range_option = input().strip().lower()
use_plot_range = plot_range_option.startswith('y') if plot_range_option else False

plot_range = None
if use_plot_range:
    print('プロット範囲を入力してください（x_start x_end y_start y_end）[m, m, ns, ns]:')
    try:
        x_start, x_end, y_start, y_end = map(float, input().strip().split())
        plot_range = [x_start, x_end, y_start, y_end]
    except ValueError:
        print('エラー: 無効な入力形式です。デフォルトの範囲を使用します。')
        use_plot_range = False




#* パラメータ
sample_interval = 0.312500  # [ns]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]


#* 出力フォルダパスの定義
output_dir = os.path.dirname(data_path)
if use_envelope:
    output_dir = os.path.join(output_dir, 'envelope')
if plot_range_option:
    output_dir = os.path.join(output_dir, 'Trimmed_plot_selected_range')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#* エンベロープ計算
def envelope(data):
    #* データのエンベロープを計算
    envelope = np.abs(signal.hilbert(data, axis=0))
    return envelope


#* プロット
font_large = 20
font_medium = 18
font_small = 16

def single_plot(plot_data):
    plt.figure(figsize=(18, 6), tight_layout=True)
    
    # データタイプに応じた設定
    if data_type == 'gained':
        if use_envelope:
            plt.imshow(plot_data, aspect='auto', cmap='viridis',
                    extent=[0, plot_data.shape[1]*trace_interval, plot_data.shape[0]*sample_interval, 0],
                    vmin=0, vmax=np.amax(np.abs(plot_data))/15
                    )
        else:
            plt.imshow(plot_data, aspect='auto', cmap='viridis',
                    extent=[0, plot_data.shape[1]*trace_interval, plot_data.shape[0]*sample_interval, 0],
                    vmin=-np.amax(np.abs(plot_data))/15, vmax=np.amax(np.abs(plot_data))/15
                    )
    else:
        if use_envelope:
            plt.imshow(plot_data, aspect='auto', cmap='viridis',
                    extent=[0, plot_data.shape[1]*trace_interval, plot_data.shape[0]*sample_interval, 0],
                    vmin=0, vmax=10
                    )
        else:
            plt.imshow(plot_data, aspect='auto', cmap='viridis',
                    extent=[0, plot_data.shape[1]*trace_interval, plot_data.shape[0]*sample_interval, 0],
                    vmin=-10, vmax=10
                    )
    
    # プロット範囲の設定
    if use_plot_range:
        plt.xlim(plot_range[0], plot_range[1])
        plt.ylim(plot_range[3], plot_range[2])  # y軸は逆転しているため、順序を入れ替え
    
    plt.xlabel('Moving distance [m]', fontsize=font_medium)
    plt.ylabel('Time [ns]', fontsize=font_medium)
    plt.tick_params(axis='both', which='major', labelsize=font_small)

    #* カラーバー
    delvider = axgrid1.make_axes_locatable(plt.gca())
    cax = delvider.append_axes('right', size='3%', pad=0.1)
    plt.colorbar(cax=cax, orientation = 'vertical').set_label('Amplitude', fontsize=font_large)
    cax.tick_params(labelsize=font_small)

    # 出力ファイル名
    if plot_range_option:
        file_base = f'x_{plot_range[0]}_{plot_range[1]}_t{plot_range[2]}_{plot_range[3]}'
    else:
        file_base = os.path.splitext(os.path.basename(data_path))[0]
    output_path_png = os.path.join(output_dir, f'{file_base}.png')
    output_path_pdf = os.path.join(output_dir, f'{file_base}.pdf')
    
    plt.savefig(output_path_png, format='png', dpi=120)
    plt.savefig(output_path_pdf, format='pdf', dpi=600)
    print(f"プロットを保存しました: {output_path_png}")
    plt.show()

    return plt


#* メイン処理
print('Loading data...')
resampled_data = np.loadtxt(data_path)
print("B-scanの形状:", resampled_data.shape)

if use_envelope:
    print('エンベロープを計算中...')
    resampled_data = envelope(resampled_data)

print('プロット作成中...')
single_plot(resampled_data)