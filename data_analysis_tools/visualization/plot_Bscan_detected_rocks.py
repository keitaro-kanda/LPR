"""
This code makes full B-scan plot from resampled ECHO data with detected rock labels.
If you want to make B-scan plot of each sequence, you can use resampling.py.
"""
# 1. Imports (standardized order)
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
import json
from glob import glob


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

def detect_json_files(data_dir):
    """Detect all JSON files in the data directory and subdirectories"""
    json_files = []
    
    # Search in common label directories
    possible_dirs = [
        os.path.join(data_dir, 'echo_labels'),
        os.path.join(data_dir, 'labels'),
        os.path.join(data_dir, 'detection_results'),
        data_dir  # Also search in the data directory itself
    ]
    
    for search_dir in possible_dirs:
        if os.path.exists(search_dir) and os.path.isdir(search_dir):
            pattern = os.path.join(search_dir, '*.json')
            found_files = glob(pattern)
            for file_path in found_files:
                if file_path not in json_files:  # Avoid duplicates
                    json_files.append(file_path)
    
    return sorted(json_files)


def select_json_files(json_files):
    """Allow user to select which JSON files to use"""
    if not json_files:
        print("JSONファイルが見つかりませんでした。")
        return [], "no_labels"
    
    print(f"\n検出されたJSONファイル ({len(json_files)}個):")
    for i, json_file in enumerate(json_files, 1):
        rel_path = os.path.relpath(json_file, os.path.dirname(data_path))
        print(f"{i}: {rel_path}")
    
    print("\n使用するJSONファイルを選択してください:")
    print("a: 全て")
    print("n: なし（ラベルなしでプロット）")
    print("数字: 特定のファイル（カンマ区切りで複数選択可能、例: 1,3,5）")
    
    selection = input("選択: ").strip().lower()
    
    if selection == 'n':
        return [], "no_labels"
    elif selection == 'a':
        return json_files, "all_labels"
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_files = [json_files[i] for i in indices if 0 <= i < len(json_files)]
            if not selected_files:
                print("有効な選択がありません。全てのファイルを使用します。")
                return json_files, "all_labels"
            
            # Create output directory suffix based on selected files
            file_names = [os.path.splitext(os.path.basename(f))[0] for f in selected_files]
            suffix = "_".join(file_names) if len(file_names) <= 3 else f"selected_{len(file_names)}_files"
            
            return selected_files, suffix
        except (ValueError, IndexError):
            print("無効な選択です。全てのファイルを使用します。")
            return json_files, "all_labels"


def load_labels_from_files(selected_files):
    """Load labels from selected JSON files"""
    all_x_coords = []
    all_y_coords = []
    all_labels = []
    
    if not selected_files:
        print("ラベルファイルが選択されていません。")
        return np.array([]), np.array([]), np.array([])
    
    print(f"\n--- {len(selected_files)}個のJSONファイルを処理中 ---")
    for labels_path in selected_files:
        print(f"処理中: {os.path.basename(labels_path)}")
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)

            results = labels_data.get('results', {})

            if not results:
                print(f"  警告: '{os.path.basename(labels_path)}' に 'results' キーが見つからないか、空でした。")
                continue

            # Extend the master lists with data from the current JSON file
            all_x_coords.extend([v['x'] for v in results.values()])
            all_y_coords.extend([v['y'] for v in results.values()])
            all_labels.extend([v['label'] for v in results.values()])

            print(f"  完了: {len(results)}個のラベルを読み込み")

        except json.JSONDecodeError:
            print(f"  エラー: '{os.path.basename(labels_path)}' が有効なJSON形式ではありません。")
        except KeyError as e:
            print(f"  エラー: '{os.path.basename(labels_path)}' に期待されるキー '{e}' が見つかりません。")
        except Exception as e:
            print(f"  予期せぬエラー: {e}")

    # Convert collected lists to NumPy arrays
    x = np.array(all_x_coords)
    t = np.array(all_y_coords)
    lab = np.array(all_labels, dtype=int)
    print(f"\n合計 {len(x)} 個のラベルポイントを読み込みました。")
    
    return x, t, lab


# JSON file detection and selection
data_dir = os.path.dirname(data_path)
json_files = detect_json_files(data_dir)
selected_files, output_suffix = select_json_files(json_files)
x, t, lab = load_labels_from_files(selected_files)



# 3. Parameter definitions
sample_interval = 0.312500e-9    # [s] - Time sampling interval
trace_interval = 3.6e-2          # [m] - Spatial trace interval
c = 299792458                    # [m/s] - Speed of light
epsilon_r = 3.0                  # Almost an average value between Chen+2022 (2.3-3.7), Dong+2020 (2-4くらい), Feng+2022 (2.64-3.85).
reciever_time_delay = 28.203e-9  # [s] - Hardware delay


# 4. Path validation and directory creation
# Already done above

# 5. Output directory setup
base_output_dir = os.path.join(os.path.dirname(data_path), 'Bscan_with_detected_rocks')
out_dir = os.path.join(base_output_dir, output_suffix)
if use_plot_range:
    out_dir = os.path.join(out_dir, 'Trimmed_plot_selected_range')
os.makedirs(out_dir, exist_ok=True)
print(f"出力ディレクトリ: {out_dir}")



# 6. Main processing functions

def find_time_zero_index(data):
    """Find time zero based on first non-NaN value at x=0"""
    if data.shape[1] == 0:
        return 0
    
    first_trace = data[:, 0]  # x=0のトレース
    
    # NaNではない最初のインデックスを探す
    valid_indices = np.where(~np.isnan(first_trace))[0]
    
    if len(valid_indices) == 0:
        print("警告: x=0のトレースにNaNではない値が見つかりません。t=0をインデックス0に設定します。")
        return 0
    
    time_zero_index = valid_indices[0]
    print(f"Time zero index: {time_zero_index}")
    print(f"Time zero: {time_zero_index * sample_interval * 1e9:.2f} ns")
    
    return time_zero_index


# Font size standards
font_large = 20      # Titles and labels
font_medium = 18     # Axis labels  
font_small = 16      # Tick labels


def single_plot(plot_data, time_zero_idx, label_keys=None, suffix=''):
    """Create B-scan plot with detected rock labels"""
    fig, ax = plt.subplots(figsize=(18, 9))

    # データタイプに応じたカラーバー範囲設定（run_data_processing.pyと同じロジック）
    if data_type == 'gained':  # Step 4: Gain function
        vmin = -np.nanmax(np.abs(plot_data))/10
        vmax = np.nanmax(np.abs(plot_data))/10
    elif data_type == 'terrain_corrected':  # Step 5: Terrain correction  
        vmin = -np.nanmax(np.abs(plot_data))/10
        vmax = np.nanmax(np.abs(plot_data))/10
    else:  # Steps 0-3: Raw, Bandpass, Time-zero, Background
        vmin = -10
        vmax = 10
    
    # Calculate time array with time-zero correction
    time_array = (np.arange(plot_data.shape[0]) - time_zero_idx) * sample_interval * 1e9  # [ns]
    
    # Create the main plot
    im = ax.imshow(plot_data, aspect='auto', cmap='seismic',
                   extent=[0, plot_data.shape[1]*trace_interval, 
                          time_array[-1], time_array[0]],
                   vmin=vmin, vmax=vmax)

    # Rock label classification standard (from CLAUDE.md)
    label_info = {
        1: ('red',     'Group 1'),
        2: ('green',   'Group 2'),
        3: ('blue',    'Group 3'),
        4: ('cyan',  'Single-peaked PNP'),
        5: ('magenta', 'Double-peaked PNP'),
        6: ('yellow',    'NPN and PNP')
    }
    
    # Plot labels if available
    if len(x) > 0 and len(t) > 0 and len(lab) > 0:
        keys = label_keys if label_keys is not None else list(label_info.keys())
        for L in keys:
            if L in label_info:
                col, _ = label_info[L]
                mask = (lab == L)
                if np.any(mask):
                    ax.scatter(
                        x[mask], t[mask],
                        c=col, s=30, marker='o',
                        edgecolors='k', linewidth=1.0
                    )

    # プロット範囲
    if use_plot_range and plot_range:
        ax.set_xlim(plot_range[0], plot_range[1])
        ax.set_ylim(plot_range[3], plot_range[2])

    ax.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax.set_ylabel('Time [ns]', fontsize=font_medium)
    ax.tick_params(axis='both', which='major', labelsize=font_small)
    
    # 第2縦軸（深さ）を追加
    ax2 = ax.twinx()
    # 第1軸の時間範囲を取得
    t_min, t_max = ax.get_ylim()
    # 対応する深さ範囲を計算 (run_data_processing.pyと同じ式)
    # depth [m] = (time [ns] * 1e-9) * c / sqrt(epsilon_r) / 2
    depth_min = (t_min * 1e-9) * c / np.sqrt(epsilon_r) / 2
    depth_max = (t_max * 1e-9) * c / np.sqrt(epsilon_r) / 2
    # 第2軸に深さの範囲とラベルを設定
    ax2.set_ylim(depth_min, depth_max)
    ax2.set_ylabel(r'Depth [m] ($\varepsilon_r = $' + f'{epsilon_r})', fontsize=font_medium)
    ax2.tick_params(axis='y', which='major', labelsize=font_small)

    # Add legend if labels are present
    if len(x) > 0 and len(t) > 0 and len(lab) > 0:
        keys = label_keys if label_keys is not None else list(label_info.keys())
        # Only include labels that actually exist in the data
        existing_keys = [k for k in keys if k in label_info and np.any(lab == k)]
        
        if existing_keys:
            # Reorder keys for proper legend layout: top row 1-3, bottom row 4-6
            if len(existing_keys) == 6 and all(k in existing_keys for k in [1, 2, 3, 4, 5, 6]):
                # For all 6 labels, arrange as: 1 2 3 (top row), 4 5 6 (bottom row)
                ordered_keys = [1, 4, 2, 5, 3, 6]
                ncol = 3
            elif len(existing_keys) <= 3:
                # For 3 or fewer labels, single row
                ordered_keys = sorted(existing_keys)
                ncol = len(existing_keys)
            else:
                # For 4-5 labels, try to balance rows
                ordered_keys = sorted(existing_keys)
                ncol = 3
            
            patches = [Patch(facecolor=label_info[L][0], edgecolor='white', label=label_info[L][1]) 
                      for L in ordered_keys if L in existing_keys]
            
            # レジェンド配置
            ax.legend(
                handles=patches,
                loc='upper center',
                bbox_to_anchor=(0.35, -0.1),
                ncol=ncol,
                frameon=False,
                fontsize=font_medium
            )
            fig.subplots_adjust(bottom=0.25, right=0.85)
            cbar_y_pos = 0.15
        else:
            fig.subplots_adjust(bottom=0.18, right=0.9)
            cbar_y_pos = 0.10
    else:
        fig.subplots_adjust(bottom=0.18, right=0.9)
        cbar_y_pos = 0.05
    
    # Add colorbar at bottom right corner
    cbar_ax = fig.add_axes([0.68, cbar_y_pos, 0.15, 0.05])  # [x, y, width, height]
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=font_small)

    # Standard save pattern
    base = os.path.splitext(os.path.basename(data_path))[0]
    if suffix:
        base = f"{base}_{suffix}"
    if use_plot_range and plot_range:
        base = f'x_{plot_range[0]}_{plot_range[1]}_t{plot_range[2]}_{plot_range[3]}' + (f'_{suffix}' if suffix else '')
    
    output_path_png = os.path.join(out_dir, base + '.png')
    output_path_pdf = os.path.join(out_dir, base + '.pdf')
    
    plt.savefig(output_path_png, dpi=120)   # Web quality
    plt.savefig(output_path_pdf, dpi=600)   # Publication quality
    print(f'プロットを保存しました: {output_path_png}')
    plt.show()
    
    return fig



# 7. Execution
if __name__ == "__main__":
    print('Loading data...')
    try:
        data = np.loadtxt(data_path, delimiter=' ')
    except ValueError:
        print('エラー: データファイルを読み込めませんでした。数値データ形式の.txtファイルを指定してください。')
        exit(1)
    
    print("B-scanの形状:", data.shape)
    
    # NaN value handling
    nan_count = np.sum(np.isnan(data))
    total_count = data.size
    if nan_count > 0:
        print(f'NaN値検出: {nan_count} / {total_count} ({nan_count/total_count*100:.2f}%)')
    else:
        print('NaN値は検出されませんでした。')
    
    # Find time zero based on first non-NaN value at x=0
    time_zero_index = find_time_zero_index(data)
    
    # Generate plots based on available labels
    if len(x) > 0 and len(t) > 0 and len(lab) > 0:
        unique_labels = np.unique(lab)
        print(f'検出されたラベルタイプ: {unique_labels}')
        
        # Plot all labels
        print('全ラベルをプロット中...')
        single_plot(data, time_zero_index, label_keys=None, suffix='allLabels')
        
        # Plot labels 1-3 if they exist
        rock_labels = [1, 2, 3]
        if any(label in unique_labels for label in rock_labels):
            print('岩石ラベル（1-3）をプロット中...')
            single_plot(data, time_zero_index, label_keys=rock_labels, suffix='labels1-3')
        
        # Plot labels 4-6 if they exist
        other_labels = [4, 5, 6]
        if any(label in unique_labels for label in other_labels):
            print('その他ラベル（4-6）をプロット中...')
            single_plot(data, time_zero_index, label_keys=other_labels, suffix='labels4-6')
    else:
        print('ラベルが見つかりません。ラベルなしでプロットします。')
        single_plot(data, time_zero_index, label_keys=None, suffix='no_labels')
    
    print('プロット完了。')
