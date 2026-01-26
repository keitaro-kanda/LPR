"""_summary_
CE-3のバイナリ読み取りコードの出力ファイル名・形式に合わせたコードとして、既存のresampling.pyとは別に作成。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
import cv2
from scipy import ndimage
import re # 正規表現モジュールを追加

# ==========================================
# 設定と入力
# ==========================================
# resampling.pyではユーザー入力にしていたが、ここではハードコーディングに変更。

#* input data folder path
data_folder_path = '/Volumes/SSD_Kanda_SAMSUNG/CE3_LPR/LPR_2B_0005-0009/loaded_data_echo_position' # ←ここに実際のパスを記入してください
if not os.path.exists(data_folder_path):
    raise ValueError('Data folder path does not exist.')

channel_name = '2B'

rover_name = 'CE-3'  # 'CE-3' または 'CE-4'

#* Define output folder path
output_dir = os.path.join(os.path.dirname(data_folder_path), 'Resampled_Data')
os.makedirs(output_dir, exist_ok=True)

txt_output_dir = os.path.join(output_dir, 'txt')
os.makedirs(txt_output_dir, exist_ok=True)

plot_output_dir = os.path.join(output_dir, 'plot')
os.makedirs(plot_output_dir, exist_ok=True)

position_output_dir = os.path.join(output_dir, 'position')
os.makedirs(position_output_dir, exist_ok=True)

idx_dir = os.path.join(output_dir, 'idx')
os.makedirs(idx_dir, exist_ok=True)


# ==========================================
# 関数定義
# ==========================================

def get_sequence_id(filename):
    """
    ファイル名からシーケンスID(例: 0005)を抽出する
    ファイル名パターン: ..._0005_A.txt または ..._0005.txt に対応
    """
    # パターン1: 数字4桁がアンダースコアで囲まれている場合 (..._0005_...)
    match = re.search(r'_(\d{4})_', filename)
    if match:
        return match.group(1)
    
    # パターン2: 数字4桁の後に .txt が来る場合 (..._0005.txt)
    match = re.search(r'_(\d{4})\.txt', filename)
    if match:
        return match.group(1)

    # 上記で見つからない場合、従来の分割ロジック（改良版）
    parts = filename.replace('.txt', '').split('_')
    # 末尾が1文字のアルファベット(A, B...)なら、その一つ前を採用
    if len(parts[-1]) == 1 and parts[-1].isalpha():
        return parts[-2]
    
    return parts[-1]


def resampling(signal_data, position_data, sequence_id, window_num, thres_val, rover_name):
    #* 最初の300サンプルを除外
    start_idx = 300 if signal_data.shape[0] > 300 else 0
    img = signal_data[start_idx:, :]

    #* Sobelフィルタ
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

    #* メディアンフィルタ
    med_denoised = ndimage.median_filter(sobelx, 5)

    #* エネルギー計算
    med = np.sum(np.abs(med_denoised), axis=0)

    #* 移動窓設定
    window = window_num
    thres = thres_val

    # デバッグ表示
    max_energy = np.max(med) if med.size > 0 else 0
    print(f"   [Debug] Max Energy: {max_energy:.0f} (Thres: {thres})")

    medf = np.zeros(med.shape)
    
    if med.shape[0] > window:
        for i in range(med.shape[0] - window):
            if np.all(med[i : i + window] > thres):
                if rover_name == 'CE-3':
                    if np.all(med[i : i + window] < 500000): 
                        medf[i : i + window] = 1
                else:
                    medf[i : i + window] = 1
    
    idx = np.where(medf == 1)[0]

    #* Channel 2Bの場合、idxを保存
    if channel_name == '2B':
        np.savetxt(os.path.join(idx_dir, str(sequence_id) + '_idx.txt'), idx, delimiter=' ', fmt='%d')

    #* フィルタリング実行と保存
    if np.sum(idx) == 0:
        data_filtered = np.zeros((signal_data.shape[0], 0))
        position_data_filtered = np.zeros((position_data.shape[0], 0))
        print('   -> No data selected.')
    else:
        data_filtered = signal_data[:, idx]
        position_data_filtered = position_data[:, idx]
        print(f'   -> Filtered data shape: {data_filtered.shape}')

        # 保存 (ファイル名に sequence_id を使用)
        save_name_txt = f"{sequence_id}_resampled.txt"
        save_name_pos = f"{sequence_id}_resampled_position.txt"

        np.savetxt(os.path.join(txt_output_dir, save_name_txt), data_filtered, delimiter=' ')
        
        header_position = 'velocity, position_x, position_y, position_z, reference_point_x, reference_point_y, reference_point_z'
        np.savetxt(os.path.join(position_output_dir, save_name_pos),
                    position_data_filtered, delimiter=' ', header=header_position)

    return sobelx, med_denoised, med, medf, data_filtered, position_data_filtered


def plot_2B(raw_data, sobelx, med_denoised, med, medf, data_filtered, sequence_id, thres):
    fig, ax = plt.subplots(6, 1, figsize=(12, 20), tight_layout=True, sharex=True)
    fontsize_large = 20
    fontsize_medium = 18
    
    # 描画範囲の決定
    raw_max = np.percentile(np.abs(raw_data), 99) if raw_data.size > 0 else 1
    
    # 1. Raw Data
    ax[0].imshow(raw_data, aspect='auto', cmap='seismic',
                extent=[0, raw_data.shape[1], raw_data.shape[0]*0.3125, 0],
                vmin=-15, vmax=15)
    ax[0].set_title(f'Raw data: {sequence_id}', fontsize=fontsize_large)
    ax[0].set_ylabel('Time [ns]', fontsize=fontsize_medium)

    # 2. Sobel
    ax[1].imshow(sobelx, aspect='auto', cmap='seismic',
                extent=[0, sobelx.shape[1], sobelx.shape[0]*0.3125, 0],
                vmin=-15, vmax=15)
    ax[1].set_title('Sobel filtered', fontsize=fontsize_large)

    # 3. Denoised
    ax[2].imshow(med_denoised, aspect='auto', cmap='seismic',
                extent=[0, med_denoised.shape[1], med_denoised.shape[0]*0.3125, 0],
                vmin=-15, vmax=15)
    ax[2].set_title('Random noise reduction', fontsize=fontsize_large)

    # 4. Amplitude
    ax[3].plot(med, label='Signal Energy')
    ax[3].hlines(thres, 0, len(med), 'r', label=f'Threshold ({thres:.1f})')
    ax[3].set_title('Filtered signals amplitude', fontsize=fontsize_large)
    ax[3].legend(loc='upper right')
    ax[3].set_ylim(0, max(np.max(med)*1.1, thres*1.5) if med.size > 0 else 1)

    # 5. Gate
    ax[4].plot(medf)
    ax[4].set_title('Gate (Selected Traces)', fontsize=fontsize_large)
    ax[4].set_ylim(-0.1, 1.1)

    # 6. Filtered Result
    if data_filtered.shape[1] > 0:
        ax[5].imshow(data_filtered, aspect='auto', cmap='seismic',
                    extent=[0, data_filtered.shape[1], data_filtered.shape[0]*0.3125, 0],
                    vmin=-15, vmax=15)
    else:
        ax[5].text(0.5, 0.5, 'No Data Selected', ha='center', va='center', fontsize=20)
        
    ax[5].set_title('Filtered ECHO data', fontsize=fontsize_large)
    ax[5].set_ylabel('Time [ns]', fontsize=fontsize_medium)

    if raw_data.shape[1] > 0:
        plt.xlim(0, raw_data.shape[1])
    
    fig.supxlabel('Trace number', fontsize=fontsize_medium)

    # 保存 (IDを使って一意な名前で保存)
    save_path = os.path.join(plot_output_dir, f"{sequence_id}_resampling_flow.png")
    plt.savefig(save_path)
    plt.close() # メモリ解放


def plot_2A(raw_data, medf, data_filtered, sequence_id):
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), tight_layout=True, sharex=True)
    fontsize_large = 20
    fontsize_medium = 18

    raw_max = np.percentile(np.abs(raw_data), 99) if raw_data.size > 0 else 1

    ax[0].imshow(raw_data, aspect='auto', cmap='seismic',
                extent=[0, raw_data.shape[1], raw_data.shape[0]*0.3125, 0],
                vmin=-raw_max, vmax=raw_max)
    ax[0].set_title(f'Raw data: {sequence_id}', fontsize=fontsize_large)
    ax[0].set_ylabel('Time [ns]', fontsize=fontsize_medium)

    ax[1].plot(medf)
    ax[1].set_title('Gate with interesting data', fontsize=fontsize_large)

    if data_filtered.shape[1] > 0:
        ax[2].imshow(data_filtered, aspect='auto', cmap='seismic',
                    extent=[0, data_filtered.shape[1], data_filtered.shape[0]*0.3125, 0],
                    vmin=-raw_max, vmax=raw_max)
    else:
        ax[2].text(0.5, 0.5, 'No Data Selected', ha='center', va='center', fontsize=20)

    ax[2].set_title('Filtered ECHO data', fontsize=fontsize_large)
    ax[2].set_ylabel('Time [ns]', fontsize=fontsize_medium)

    if raw_data.shape[1] > 0:
        plt.xlim(0, raw_data.shape[1])
        
    fig.supxlabel('Trace number', fontsize=fontsize_medium)
    
    plt.savefig(os.path.join(plot_output_dir, f"{sequence_id}_resampling_flow.png"))
    plt.close()


# ==========================================
# メイン処理ループ
# ==========================================

total_trace_num = 0
resampled_trace_num = 0

all_files = natsorted(os.listdir(data_folder_path))
target_files = [f for f in all_files if f.startswith('data_') and f.endswith('.txt')]

print(f"Processing {len(target_files)} files...")

for ECHO_data in tqdm(target_files):
    
    ECHO_data_path = os.path.join(data_folder_path, ECHO_data)
    
    try:
        raw_data = np.loadtxt(ECHO_data_path, delimiter=' ', skiprows=1)
    except Exception as e:
        print(f"Error loading {ECHO_data}: {e}")
        continue

    if raw_data.ndim < 2 or raw_data.shape[0] < 8:
        continue

    positions = raw_data[:7, :]
    signals = raw_data[7:, :]
    
    # ID抽出ロジックを変更
    sequence_id = get_sequence_id(ECHO_data)

    print(f'\n--- Processing: {ECHO_data} (ID: {sequence_id}) ---')
    total_trace_num += signals.shape[1]
    
    if channel_name == '2B':
        # 自動閾値調整ロジック (前回と同様)
        signal_level = np.max(np.sum(np.abs(signals[300:, :]), axis=0)) if signals.shape[0] > 300 else 0
        
        if rover_name == 'CE-4':
            base_thres = 25000
            window_num = 16
        else: # CE-3
            base_thres = 13000
            window_num = 6

        if signal_level < 1000 and signal_level > 0:
            thres_val = signal_level * 0.3 
        else:
            thres_val = base_thres
            if rover_name == 'CE-3' and sequence_id == '0004':
                thres_val = 30000

        sobelx, med_denoised, med, medf, data_filtered, positions_filtered = resampling(
            signals, positions, sequence_id, window_num, thres_val, rover_name
        )
        
        plot_2B(signals, sobelx, med_denoised, med, medf, data_filtered, sequence_id, thres_val)


    elif channel_name == '2A':
        idx_file_path = os.path.join(idx_dir, sequence_id + '_idx.txt')
        
        if not os.path.exists(idx_file_path):
            print(f'   Warning: No idx file found for sequence {sequence_id}.')
            medf = np.zeros(signals.shape[1])
            data_filtered = np.zeros((signals.shape[0], 0))
            positions_filtered = np.zeros((positions.shape[0], 0))
        else:
            try:
                if os.path.getsize(idx_file_path) == 0:
                     idx = np.array([])
                else:
                     idx = np.loadtxt(idx_file_path, delimiter=' ', ndmin=1)
                
                if idx.size == 0:
                    medf = np.zeros(signals.shape[1])
                    data_filtered = np.zeros((signals.shape[0], 0))
                    positions_filtered = np.zeros((positions.shape[0], 0))
                else:
                    idx = idx.astype(int)
                    idx = idx[idx < signals.shape[1]]
                    
                    medf = np.zeros(signals.shape[1])
                    medf[idx] = 1

                    data_filtered = signals[:, idx]
                    positions_filtered = positions[:, idx]
                    
                    np.savetxt(os.path.join(txt_output_dir, f"{sequence_id}_resampled.txt"), data_filtered, delimiter=' ')
                    
                    header_position = 'velocity, position_x, position_y, position_z, reference_point_x, reference_point_y, reference_point_z'
                    np.savetxt(os.path.join(position_output_dir, f"{sequence_id}_resampled_position.txt"),
                                positions_filtered, delimiter=' ', header=header_position)

            except Exception as e:
                print(f"   Error reading idx file: {e}")
                continue

        plot_2A(raw_data, medf, data_filtered, sequence_id)

    elif channel_name == '1':
        continue

    resampled_trace_num += data_filtered.shape[1]


with open(os.path.join(output_dir, 'total_trace_num.txt'), 'w') as f:
    f.write('Number of total traces before resampling: ' + str(total_trace_num) + '\n')
    f.write('Number of total traces after resampling: '+ str(resampled_trace_num))

print('\nAll processing complete.')