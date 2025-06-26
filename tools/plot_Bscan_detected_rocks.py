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

data_type = input('データの種類を選択してください（raw, bandpass_filtered, time_zero_corrected, background_filtered, gained）:').strip()
if data_type not in ['raw', 'bandpass_filtered', 'time_zero_corrected', 'background_filtered', 'gained']:
    print('エラー: 無効なデータ種類です')
    exit(1)

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
sample_interval = 0.312500  # [ns]
trace_interval = 3.6e-2     # [m]

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
        vmax = np.max(np.abs(plot_data)) / 15
        vmin = -vmax
    im = ax.imshow(
        plot_data,
        aspect='auto',
        cmap='viridis',
        extent=[0, plot_data.shape[1]*trace_interval,
                plot_data.shape[0]*sample_interval, 0],
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
        col, desc = info[L]
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

    # カラーバー
    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Amplitude', fontsize=font_large)
    cax.tick_params(labelsize=font_small)

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
    plt.subplots_adjust(bottom=0.2)

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
