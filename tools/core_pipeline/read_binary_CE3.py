import struct
import os
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from natsort import natsorted
import matplotlib.pyplot as plt

# ==========================================
# 設定: データフォルダのパス
# ==========================================
#data_folder_path = '/Volumes/SSD_Kanda_SAMSUNG/CE3_LPR/LPR_2B_0005-0009/original_binary'  # ←ここに実際のパスを記入してください
data_folder_path = '/Volumes/SSD_Kanda_SAMSUNG/CE3_LPR/LPR_2B_all/original_binary'

# ==========================================
# 関数定義
# ==========================================

def read_binary_data(file_path, start_byte, record_format):
    """バイナリデータを指定フォーマットで読み込む関数"""
    results = {}
    with open(file_path, 'rb') as file:
        for field_name, field_loc, field_len, fmt in record_format:
            file.seek(start_byte + field_loc - 1)
            data = file.read(field_len)
            if len(data) != field_len:
                continue
            try:
                if fmt.endswith('s'):
                    results[field_name] = data.decode().strip()
                elif 'B' in fmt and len(fmt) > 1 and not fmt[0].isdigit(): 
                     results[field_name] = struct.unpack(fmt, data)
                else:
                    val = struct.unpack(fmt, data)
                    results[field_name] = val[0] if len(val) == 1 else val
            except struct.error:
                continue
    return results

def read_frame_identification(mode_value):
    if isinstance(mode_value, tuple):
        mode_int = int.from_bytes(bytes(mode_value), byteorder='little')
    else:
        mode_int = mode_value
    frame_identification = {0x146F2222: 'Channel 2 data'}
    return frame_identification.get(mode_int, f"Unknown ID: {hex(mode_int) if isinstance(mode_int, int) else mode_value}")

# ==========================================
# レコードフォーマット定義 (Little Endian <)
# ==========================================
record_format_2B_corrected = [
    ("FRAME_IDENTIFICATION", 1, 4, '4B'),
    ("TIME", 5, 6, '6B'), 
    ("VELOCITY", 11, 4, '<f'),
    ("XPOSITION", 15, 4, '<f'), # Rover X = NORTH
    ("YPOSITION", 19, 4, '<f'), # Rover Y = EAST
    ("ZPOSITION", 23, 4, '<f'),
    ("ATT_PITCHING", 27, 4, '<f'),
    ("ATT_ROLLING", 31, 4, '<f'),
    ("ATT_YAWING", 35, 4, '<f'),
    ("REFERENCE_POINT_XPOSITION", 39, 4, '<f'), # Ref X = EAST
    ("REFERENCE_POINT_YPOSITION", 43, 4, '<f'), # Ref Y = NORTH
    ("REFERENCE_POINT_ZPOSITION", 47, 4, '<f'),
    ("REFERENCE_POINT_ATT_PITCHING", 51, 4, '<f'),
    ("REFERENCE_POINT_ATT_ROLLING", 55, 4, '<f'),
    ("REFERENCE_POINT_ATT_YAWING", 59, 4, '<f'),
    ("DATA_BLOCK_COUNT", 63, 2, '<H'), 
    ("VALID_DATA_LENGTH", 108, 2, '<H'),
    ("CHANNEL_1_RECORD_COUNT", 110, 2, '<H'),
    ("CHANNEL_2_RECORD_COUNT", 112, 2, '<H'),
    ("CHANNEL_AND_ANTENNA_MARK", 114, 1, 'B'),
    ("ECHO_DATA", 115, 8192, '<2048f'),
    ("QUALITY_STATE", 8307, 1, 'B')
]

# ==========================================
# メイン処理
# ==========================================

if not os.path.exists(data_folder_path):
    print(f'Error: Folder not found: {data_folder_path}')
    exit(1)

print('Data folder is successfully loaded')
print('Channel: LPR_2B (Fixed format)')

loaded_data_dir = os.path.join(os.path.dirname(data_folder_path), 'loaded_data')
os.makedirs(loaded_data_dir, exist_ok=True)
ECHO_dir = os.path.join(os.path.dirname(data_folder_path), 'loaded_data_echo_position')
os.makedirs(ECHO_dir, exist_ok=True)

# 経路データ蓄積用リスト
all_segments = []

file_list = natsorted([f for f in os.listdir(data_folder_path) if f.endswith('.2B') and not f.startswith('._')])
print(f'Total {len(file_list)} files found to process.')

for filename in tqdm(file_list, desc='Processing Files'):
    full_path = os.path.join(data_folder_path, filename)
    
    file_size = os.path.getsize(full_path)
    RECORD_BYTES = 8307
    LABEL_RECORDS = 5
    HEADER_OFFSET = LABEL_RECORDS * RECORD_BYTES
    
    if file_size < HEADER_OFFSET:
        continue
        
    records = int((file_size - HEADER_OFFSET) / RECORD_BYTES)
    
    parts = filename.split('_')
    sequence_id = parts[-2] if len(parts) >= 2 else "unknown"

    save_data = np.zeros((2055, records))

    for record_index in range(records):
        current_offset = HEADER_OFFSET + (record_index * RECORD_BYTES)
        loaded_data = read_binary_data(full_path, current_offset, record_format_2B_corrected)

        if 'ECHO_DATA' in loaded_data:
            save_data[0, record_index] = loaded_data.get('VELOCITY', 0)
            save_data[1, record_index] = loaded_data.get('XPOSITION', 0)
            save_data[2, record_index] = loaded_data.get('YPOSITION', 0)
            save_data[3, record_index] = loaded_data.get('ZPOSITION', 0)
            save_data[4, record_index] = loaded_data.get('REFERENCE_POINT_XPOSITION', 0)
            save_data[5, record_index] = loaded_data.get('REFERENCE_POINT_YPOSITION', 0)
            save_data[6, record_index] = loaded_data.get('REFERENCE_POINT_ZPOSITION', 0)
            save_data[7:, record_index] = loaded_data['ECHO_DATA']

    # 数値データ保存
    header_txt = ['VELOCITY', 'XPOSITION(North)', 'YPOSITION(East)', 'ZPOSITION',
                  'REFERENCE_POINT_XPOSITION(East)', 'REFERENCE_POINT_YPOSITION(North)', 'REFERENCE_POINT_ZPOSITION',
                  'Observed amplitude (7:)']
    base_name = os.path.splitext(filename)[0]
    np.savetxt(f"{ECHO_dir}/data_{base_name}.txt", save_data, header=' '.join(header_txt), comments='# ')

    # ---------------------------------------------------------
    # 1. 個別ファイルの4パネルプロット作成
    # ---------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # B-scan
    max_val = np.percentile(np.abs(save_data[7:, :]), 98) if records > 0 else 1
    axes[0].imshow(save_data[7:, :], aspect='auto', cmap='seismic', 
                        interpolation='nearest', vmin=-max_val/10, vmax=max_val/10)
    axes[0].set_title(f'B-scan: {filename}')
    axes[0].set_ylabel('Time Sample')

    # Velocity
    axes[1].plot(save_data[0, :], label='Velocity', color='black', linewidth=1)
    axes[1].set_title('Velocity [m/s]') 
    axes[1].set_ylabel('Value')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # Rover Position (X=North, Y=East based on Header)
    axes[2].plot(save_data[1, :], label='X (North)', color='red', linewidth=1)
    axes[2].plot(save_data[2, :], label='Y (East)', color='green', linewidth=1)
    axes[2].plot(save_data[3, :], label='Z', color='blue', linewidth=1)
    axes[2].set_title('Rover Position (Relative)')
    axes[2].set_ylabel('Position (m)')
    axes[2].legend(loc='upper right', fontsize='small')
    axes[2].grid(True, linestyle=':', alpha=0.6)

    # Reference Point Position (X=East, Y=North based on Header)
    axes[3].plot(save_data[4, :], label='Ref X (East)', color='red', linestyle='--', linewidth=1)
    axes[3].plot(save_data[5, :], label='Ref Y (North)', color='green', linestyle='--', linewidth=1)
    axes[3].plot(save_data[6, :], label='Ref Z', color='blue', linestyle='--', linewidth=1)
    axes[3].set_title('Reference Point Position')
    axes[3].set_xlabel('Trace Number (Record Index)')
    axes[3].set_ylabel('Position (m)')
    axes[3].legend(loc='upper right', fontsize='small')
    axes[3].grid(True, linestyle=':', alpha=0.6)

    plt.subplots_adjust(hspace=0.15, left=0.1, right=0.95, top=0.95, bottom=0.05)
    plot_path = os.path.join(ECHO_dir, f"plot_{base_name}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # 経路データの蓄積 (絶対座標計算)
    # 
    # Global East (Map X)  = Ref X (East)  + Rover Y (East)
    # Global North (Map Y) = Ref Y (North) + Rover X (North)
    # ---------------------------------------------------------
    
    # Rover: save_data[2] is Y (East)
    # Ref:   save_data[4] is X (East)
    segment_east  = save_data[4, :] + save_data[2, :]
    
    # Rover: save_data[1] is X (North)
    # Ref:   save_data[5] is Y (North)
    segment_north = save_data[5, :] + save_data[1, :] 
    
    all_segments.append({
        'filename': filename,
        'north': segment_north,
        'east': segment_east
    })

# ---------------------------------------------------------
# 2. 全体経路マップ作成 (RGBCMY色分け)
# ---------------------------------------------------------
print("Generating color-coded trajectory map...")

if all_segments:
    plt.figure(figsize=(8, 6))
    
    # 指定された色リスト
    custom_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    for i, seg in enumerate(all_segments):
        color = custom_colors[i % len(custom_colors)]
        short_name = seg['filename'].split('_')[-2] + '_' + seg['filename'].split('_')[-1].replace('.2B', '')
        
        # 経路プロット (横軸=East, 縦軸=North)
        plt.plot(seg['east'], seg['north'], 
                 marker='.', markersize=4, linewidth=1, 
                 color=color, label=f"{i+1}: {short_name}", alpha=0.8)
        
        # 開始点
        plt.scatter(seg['east'][0], seg['north'][0], color=color, marker='o', s=80, edgecolors='k', zorder=5)
        
        # データが動いていない場合の注釈
        if np.ptp(seg['east']) < 0.1 and np.ptp(seg['north']) < 0.1:
             plt.text(seg['east'][0], seg['north'][0], f" {short_name}", fontsize=9, color=color, fontweight='bold')

    # ミッション全体の開始点・終了点
    first_seg = all_segments[0]
    last_seg = all_segments[-1]
    
    plt.scatter(first_seg['east'][0], first_seg['north'][0], 
                c='lime', marker='^', s=250, label='Mission Start', zorder=10, edgecolors='k', linewidth=1.5)
    plt.scatter(last_seg['east'][-1], last_seg['north'][-1], 
                c='red', marker='X', s=250, label='Mission End', zorder=10, edgecolors='k', linewidth=1.5)
    
    plt.title(f'Combined Rover Trajectory (Color by File)\nX axis: East (Ref_X + Rov_Y), Y axis: North (Ref_Y + Rov_X)')
    plt.xlabel('Global East Position (m)')
    plt.ylabel('Global North Position (m)')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    
    # 保存先を ECHO_dir に変更
    map_path = os.path.join(ECHO_dir, 'trajectory_map_colored.png')
    plt.savefig(map_path, dpi=300)
    print(f"Trajectory map saved to: {map_path}")

else:
    print("No trajectory data found.")

print('All processes finished.')