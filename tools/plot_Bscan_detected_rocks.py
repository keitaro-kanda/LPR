"""
This code makes full B-scan plot from resampled ECHO data.
If you want to make B-scan plot of each sequence, you can use resampling.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from matplotlib.patches import Patch
from scipy import signal
import json

# --- ユーザ入力 ---
data_path = input('B-scanデータファイル(.txt)のパスを入力してください:').strip()
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

# --- Load Labels from JSON files in 'rock_labels' directory ---
# Initialize lists to store all collected x, t, and lab values
all_x_coords = []
all_y_coords = []
all_labels = []

label_files_dir = os.path.join(os.path.dirname(data_path), 'rock_labels')

json_files = []
if os.path.exists(label_files_dir) and os.path.isdir(label_files_dir):
    for filename in sorted(os.listdir(label_files_dir)):
        if filename.endswith('.json'):
            json_files.append(os.path.join(label_files_dir, filename))
else:
    print(f"エラー: ラベルディレクトリ '{label_files_dir}' が見つからないか、ディレクトリではありません。")
    exit(1) # Exit if the label directory doesn't exist

if not json_files:
    print(f"ディレクトリ '{label_files_dir}' にJSONファイルが見つかりませんでした。")
    # If no JSON files are found, the x, t, lab arrays will be empty, which is acceptable
    # if there are genuinely no labels.
else:
    print(f"ディレクトリ '{label_files_dir}' から以下のJSONファイルを処理します:")
    for json_file in json_files:
        print(f"- {os.path.basename(json_file)}")

    print("\n--- 処理開始 ---")
    for labels_path in json_files:
        print(f"\nファイル: {os.path.basename(labels_path)} を処理中...")
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)

            results = labels_data.get('results', {})

            if not results:
                print(f"  Warning: '{os.path.basename(labels_path)}' に 'results' キーが見つからないか、空でした。スキップします。")
                continue

            # Extend the master lists with data from the current JSON file
            all_x_coords.extend([v['x'] for v in results.values()])
            all_y_coords.extend([v['y'] for v in results.values()])
            all_labels.extend([v['label'] for v in results.values()])

            print(f"  処理完了: {os.path.basename(labels_path)}")

        except json.JSONDecodeError:
                print(f"  Error: '{os.path.basename(labels_path)}' が有効なJSON形式ではありません。")
        except KeyError as e:
            print(f"  Error: '{os.path.basename(labels_path)}' の 'results' 内に期待されるキー '{e}' が見つかりませんでした。")
        except Exception as e:
            print(f"  予期せぬエラーが発生しました: {e} (ファイル: {os.path.basename(labels_path)})")

    # Convert collected lists to NumPy arrays
    x = np.array(all_x_coords)
    t = np.array(all_y_coords) # Renamed to t as per original code's variable
    lab = np.array(all_labels, dtype=int)
    print("\n--- 全てのラベルファイルの処理が終了しました ---")
    print(f"合計 {len(x)} 個のラベルポイントを読み込みました。")

# --- パラメータ ---
sample_interval = 0.312500e-9  # [s] (run_data_processing.pyと同じ単位)
trace_interval = 3.6e-2        # [m]
c = 299792458                  # [m/s] 光速
epsilon_r = 4.5               # 比誘電率 (run_data_processing.pyと同じ値)

# --- 出力フォルダ ---
out_dir = os.path.join(os.path.dirname(data_path), 'Bscan_with_detected_rocks')
if use_plot_range:
    out_dir = os.path.join(out_dir, 'Trimmed_plot_selected_range')
os.makedirs(out_dir, exist_ok=True)

# --- データ読込 ---
print('データを読み込み中...')
try:
    data = np.loadtxt(data_path)
except ValueError:
    print('エラー: データファイルを読み込めませんでした。数値データ形式の.txtファイルを指定してください。')
    exit(1)

# NaN値の統計を表示
nan_count = np.sum(np.isnan(data))
total_count = data.size
if nan_count > 0:
    print(f'NaN値検出: {nan_count} / {total_count} ({nan_count/total_count*100:.2f}%)')
else:
    print('NaN値は検出されませんでした。')

def find_time_zero_index(data):
    """
    x=0（最初のトレース）において最初にNaNではない値をとるインデックスを見つける
    
    Parameters:
    -----------
    data : np.ndarray
        B-scanデータ (time x traces)
    
    Returns:
    --------
    time_zero_index : int
        t=0として設定するインデックス番号
    """
    if data.shape[1] == 0:
        return 0
    
    first_trace = data[:, 0]  # x=0のトレース
    
    # NaNではない最初のインデックスを探す
    valid_indices = np.where(~np.isnan(first_trace))[0]
    
    if len(valid_indices) == 0:
        print("警告: x=0のトレースにNaNではない値が見つかりません。t=0をインデックス0に設定します。")
        return 0
    
    time_zero_index = valid_indices[0]
    print(f"t=0として設定: インデックス {time_zero_index}")
    
    return time_zero_index

# t=0インデックスを取得
time_zero_index = find_time_zero_index(data)

# --- プロット関数 ---
font_large = 20
font_medium = 18
font_small = 16

def single_plot(plot_data, label_keys=None, suffix=''):
    """
    plot_data: 2D numpy array
    label_keys: list of label numbers to plot (e.g. [1,2,3]) or None for all
    suffix: string appended to output file base name
    """
    fig, ax = plt.subplots(figsize=(18, 8), tight_layout=True)

    # B-scan表示
    vmin, vmax = -10, 10
    if data_type == 'gained':
        vmax = np.nanmax(np.abs(plot_data)) / 15  # NaN対応
        vmin = -vmax
    
    # 時間軸の計算（t=0補正を適用）
    x_start, x_end = 0, plot_data.shape[1] * trace_interval
    y_start = -time_zero_index * sample_interval / 1e-9  # 負の時間も含む
    y_end = (plot_data.shape[0] - time_zero_index) * sample_interval / 1e-9
    
    print(f"時間軸範囲: {y_start:.2f} ns ～ {y_end:.2f} ns")
    print(f"空間軸範囲: {x_start:.2f} m ～ {x_end:.2f} m")
    
    # terrain correctedデータの場合の特別な処理
    if data_type == 'terrain_corrected' or 'terrain' in data_type.lower():
        im = ax.imshow(
            plot_data,
            aspect='auto',
            cmap='viridis',
            extent=[x_start, x_end, y_end, y_start],
            vmin=-np.nanmax(np.abs(plot_data))/10 if np.any(~np.isnan(plot_data)) else -1,
            vmax=np.nanmax(np.abs(plot_data))/10 if np.any(~np.isnan(plot_data)) else 1
        )
    else:
        im = ax.imshow(
            plot_data,
            aspect='auto',
            cmap='viridis',
            extent=[x_start, x_end, y_end, y_start],
            vmin=vmin, vmax=vmax
        )

    # ラベル設定
    info = {
        1: ('red',     'Single-peaked NPN'),
        2: ('green',   'Double-peaked NPN'),
        3: ('blue',    'PNP and NPN'),
        4: ('yellow',  'Single-peaked PNP'),
        5: ('magenta', 'Double-peaked PNP'),
        6: ('cyan',    'NPN and PNP')
    }
    keys = label_keys if label_keys is not None else list(info.keys())
    for L in keys:
        col, _ = info[L]  # desc is used only in legend, not here
        mask = (lab == L)
        if np.any(mask):
            ax.scatter(
                x[mask], t[mask],
                c=col, s=50, marker='o',
                edgecolors='white'
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
    ax2.set_ylabel(r'Depth [m] ($\varepsilon_r = 4.5$)', fontsize=font_medium)
    ax2.tick_params(axis='y', which='major', labelsize=font_small)

    # カラーバーを右下に配置 (run_data_processing.pyと同じスタイル)
    fig.subplots_adjust(bottom=0.25, right=0.85)  # レジェンドとカラーバーのスペースを確保
    # Figureのサイズを基準とした位置とサイズ [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.65, 0.05, 0.2, 0.05])  # [x, y, 幅, 高さ]
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Amplitude', fontsize=font_medium)
    cbar.ax.tick_params(labelsize=font_small)

    # レジェンド（2行3列）
    patches = [Patch(facecolor=info[L][0], edgecolor='white', label=info[L][1]) for L in keys]
    ax.legend(
        handles=patches,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        frameon=False,
        fontsize=font_medium
    )
    # レイアウト調整はカラーバー設定の部分で実施済み

    # 保存
    base = os.path.splitext(os.path.basename(data_path))[0]
    if suffix:
        base = f"{base}_{suffix}"
    if use_plot_range and plot_range:
        base = f'x_{plot_range[0]}_{plot_range[1]}_t{plot_range[2]}_{plot_range[3]}' + (f'_{suffix}' if suffix else '')
    png = os.path.join(out_dir, base + '.png')
    pdf = os.path.join(out_dir, base + '.pdf')
    plt.savefig(png, dpi=120)
    plt.savefig(pdf, dpi=600)
    print('Saved:', png)
    plt.show()

# --- 実行 ---
print('B-scan shape:', data.shape)
# 全ラベルプロット
print('Plotting all labels...')
single_plot(data, label_keys=None, suffix='allLabels')
# ラベル1-3のみプロット
print('Plotting labels 1-3...')
single_plot(data, label_keys=[1,2,3], suffix='labels1-3')
