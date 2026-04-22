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
import matplotlib.cm as cm

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
    
    possible_dirs = [
        os.path.join(data_dir, 'echo_labels'),
        os.path.join(data_dir, 'labels'),
        os.path.join(data_dir, 'detection_results'),
        data_dir
    ]
    
    for search_dir in possible_dirs:
        if os.path.exists(search_dir) and os.path.isdir(search_dir):
            pattern = os.path.join(search_dir, '*.json')
            found_files = glob(pattern)
            for file_path in found_files:
                if file_path not in json_files:
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
            
            file_names = [os.path.splitext(os.path.basename(f))[0] for f in selected_files]
            suffix = "_".join(file_names) if len(file_names) <= 3 else f"selected_{len(file_names)}_files"
            
            return selected_files, suffix
        except (ValueError, IndexError):
            print("無効な選択です。全てのファイルを使用します。")
            return json_files, "all_labels"


def load_labels_from_files(selected_files):
    """Load labels and size info from selected JSON files"""
    all_x_coords = []
    all_y_coords = []
    all_labels = []
    all_time_tops = []
    all_time_bottoms = []
    
    if not selected_files:
        print("ラベルファイルが選択されていません。")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
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

            all_x_coords.extend([v.get('x') for v in results.values()])
            all_y_coords.extend([v.get('y') for v in results.values()])
            all_labels.extend([v.get('label') for v in results.values()])
            all_time_tops.extend([v.get('time_top') for v in results.values()])
            all_time_bottoms.extend([v.get('time_bottom') for v in results.values()])

            print(f"  完了: {len(results)}個のラベルを読み込み")

        except json.JSONDecodeError:
            print(f"  エラー: '{os.path.basename(labels_path)}' が有効なJSON形式ではありません。")
        except Exception as e:
            print(f"  予期せぬエラー: {e}")

    x = np.array(all_x_coords, dtype=float)
    t = np.array(all_y_coords, dtype=float)
    lab = np.array(all_labels, dtype=int)
    
    # None (null) を考慮して配列化
    t_top = np.array([v if v is not None else np.nan for v in all_time_tops], dtype=float)
    t_bot = np.array([v if v is not None else np.nan for v in all_time_bottoms], dtype=float)
    
    print(f"\n合計 {len(x)} 個のラベルポイントを読み込みました。")
    
    return x, t, lab, t_top, t_bot


# JSON file detection and selection
data_dir = os.path.dirname(data_path)
json_files = detect_json_files(data_dir)
selected_files, output_suffix = select_json_files(json_files)
x, t, lab, t_top, t_bot = load_labels_from_files(selected_files)


# 3. Parameter definitions
sample_interval = 0.312500e-9    # [s]
trace_interval = 3.6e-2          # [m]
c = 299792458                    # [m/s]
epsilon_r = 3.0                  # Average value
reciever_time_delay = 28.203e-9  # [s]


# 5. Output directory setup
base_output_dir = os.path.join(os.path.dirname(data_path), 'Bscan_with_detected_rocks')
out_dir = os.path.join(base_output_dir, output_suffix)
if use_plot_range:
    out_dir = os.path.join(out_dir, 'Trimmed_plot_selected_range')
os.makedirs(out_dir, exist_ok=True)
print(f"出力ディレクトリ: {out_dir}")


# 6. Main processing functions
def find_time_zero_index(data):
    if data.shape[1] == 0:
        return 0
    first_trace = data[:, 0]
    valid_indices = np.where(~np.isnan(first_trace))[0]
    
    if len(valid_indices) == 0:
        return 0
    
    time_zero_index = valid_indices[0]
    return time_zero_index


font_large = 20
font_medium = 18
font_small = 16

def single_plot(plot_data, time_zero_idx, plot_mode='all', suffix=''):
    """Create B-scan plot with size-colored labels"""
    fig, ax = plt.subplots(figsize=(18, 9))

    if data_type == 'gained':
        vmin = -np.nanmax(np.abs(plot_data))/10
        vmax = np.nanmax(np.abs(plot_data))/10
    elif data_type == 'terrain_corrected':  
        vmin = -np.nanmax(np.abs(plot_data))/10
        vmax = np.nanmax(np.abs(plot_data))/10
    else:
        vmin = -10
        vmax = 10
    
    time_array = (np.arange(plot_data.shape[0]) - time_zero_idx) * sample_interval * 1e9  # [ns]
    im = ax.imshow(plot_data, aspect='auto', cmap='seismic',
                   extent=[0, plot_data.shape[1]*trace_interval, 
                          time_array[-1], time_array[0]],
                   vmin=vmin, vmax=vmax)

    scatter_im = None
    cbar_label = ""
    legend_handles = []

    if len(x) > 0 and len(t) > 0 and len(lab) > 0:
        if plot_mode == 'all':
            # 岩石(1-3)と空洞(4-6)の2グループとしてのみ表示
            mask_rocks = np.isin(lab, [1, 2, 3])
            mask_cavs = np.isin(lab, [4, 5, 6])
            
            if np.any(mask_rocks):
                ax.scatter(x[mask_rocks], t[mask_rocks], c='red', s=40, edgecolors='k', linewidths=1.0)
                legend_handles.append(Patch(facecolor='red', edgecolor='white', label='Rocks'))
            if np.any(mask_cavs):
                ax.scatter(x[mask_cavs], t[mask_cavs], c='blue', s=40, edgecolors='k', linewidths=1.0)
                legend_handles.append(Patch(facecolor='blue', edgecolor='white', label='voids'))

        elif plot_mode == 'rocks':
            # 岩石サイズの計算 (time_top/bottomはnsなので秒に変換)
            valid_size_mask = ~np.isnan(t_top) & ~np.isnan(t_bot)
            mask_measured = np.isin(lab, [2, 3]) & valid_size_mask
            mask_unmeasured = (lab == 1) | (np.isin(lab, [2, 3]) & ~valid_size_mask)

            # サイズ不明またはラベル1 (グレー固定)
            if np.any(mask_unmeasured):
                ax.scatter(x[mask_unmeasured], t[mask_unmeasured], c='gray', s=40, edgecolors='k', linewidths=1.0)
                legend_handles.append(Patch(facecolor='gray', edgecolor='white', label='Rock (Size unmeasured)'))

            # サイズ計算可能 (カラーマップでサイズ表示)
            if np.any(mask_measured):
                # 岩石サイズ = (time_bottom - time_top) * c / √9 * 0.5
                sizes = (t_bot[mask_measured] - t_top[mask_measured]) * 1e-9 * c / np.sqrt(9.0) * 0.5 * 100 # [cm]
                scatter_im = ax.scatter(x[mask_measured], t[mask_measured], c=sizes, cmap='autumn', s=60, edgecolors='k', linewidths=1.0, vmin=0, vmax=50)
                cbar_label = 'Rock Size [cm]'

        elif plot_mode == 'voids':
            # 空洞サイズの計算 (time_top/bottomはnsなので秒に変換)
            valid_size_mask = ~np.isnan(t_top) & ~np.isnan(t_bot)
            mask_measured = np.isin(lab, [5, 6]) & valid_size_mask
            mask_unmeasured = (lab == 4) | (np.isin(lab, [5, 6]) & ~valid_size_mask)

            # サイズ不明またはラベル4 (グレー固定)
            if np.any(mask_unmeasured):
                ax.scatter(x[mask_unmeasured], t[mask_unmeasured], c='gray', s=40, edgecolors='k', linewidths=1.0)
                legend_handles.append(Patch(facecolor='gray', edgecolor='white', label='Void (Size unmeasured)'))

            # サイズ計算可能 (カラーマップでサイズ表示)
            if np.any(mask_measured):
                # 空洞サイズ = (time_bottom - time_top) * c * 0.5
                sizes = (t_bot[mask_measured] - t_top[mask_measured]) * 1e-9 * c * 0.5 * 100 # [cm]
                scatter_im = ax.scatter(x[mask_measured], t[mask_measured], c=sizes, cmap='winter', s=60, edgecolors='k', linewidths=1.0)
                cbar_label = 'Void Size [m]'

    if use_plot_range and plot_range:
        ax.set_xlim(plot_range[0], plot_range[1])
        ax.set_ylim(plot_range[3], plot_range[2])

    ax.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax.set_ylabel('Time [ns]', fontsize=font_medium)
    ax.tick_params(axis='both', which='major', labelsize=font_small)
    
    # 深度の第2軸
    ax2 = ax.twinx()
    t_min, t_max = ax.get_ylim()
    depth_min = (t_min * 1e-9) * c / np.sqrt(epsilon_r) / 2
    depth_max = (t_max * 1e-9) * c / np.sqrt(epsilon_r) / 2
    ax2.set_ylim(depth_min, depth_max)
    ax2.set_ylabel(r'Depth [m] ($\varepsilon_r = $' + f'{epsilon_r})', fontsize=font_medium)
    ax2.tick_params(axis='y', which='major', labelsize=font_small)

    # レジェンドの設定（左側のカラーバーと被らないよう、中央寄りに配置）
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.15, -0.10), ncol=2, frameon=False, fontsize=font_medium)
    
    # 図の余白を調整
    fig.subplots_adjust(bottom=0.25, right=0.82)
    
    # 振幅カラーバー (右下に配置・従来通り)
    cbar_ax_img = fig.add_axes([0.68, 0.145, 0.12, 0.03])
    cbar_img = plt.colorbar(im, cax=cbar_ax_img, orientation='horizontal')
    cbar_img.set_label('Amplitude', fontsize=font_small)
    cbar_img.ax.tick_params(labelsize=font_small)

    # サイズのカラーバーが必要な場合は左下に配置（振幅カラーバーと高さ・幅を揃える）
    if scatter_im is not None:
        cbar_ax_scat = fig.add_axes([0.35, 0.145, 0.12, 0.03])
        cbar_scat = plt.colorbar(scatter_im, cax=cbar_ax_scat, orientation='horizontal')
        cbar_scat.set_label(cbar_label, fontsize=font_small)
        cbar_scat.ax.tick_params(labelsize=font_small)

    base = os.path.splitext(os.path.basename(data_path))[0]
    if suffix:
        base = f"{base}_{suffix}"
    if use_plot_range and plot_range:
        base = f'x_{plot_range[0]}_{plot_range[1]}_t{plot_range[2]}_{plot_range[3]}' + (f'_{suffix}' if suffix else '')
    
    output_path_png = os.path.join(out_dir, base + '.png')
    output_path_pdf = os.path.join(out_dir, base + '.pdf')
    
    plt.savefig(output_path_png, dpi=120, bbox_inches='tight')
    plt.savefig(output_path_pdf, dpi=600, bbox_inches='tight')
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
    
    time_zero_index = find_time_zero_index(data)
    
    if len(x) > 0 and len(t) > 0 and len(lab) > 0:
        unique_labels = np.unique(lab)
        print(f'検出されたラベルタイプ: {unique_labels}')
        
        # 全ラベルプロット
        print('全ラベルをプロット中 (Rocks vs voids)...')
        single_plot(data, time_zero_index, plot_mode='all', suffix='allLabels')
        
        # 岩石(1-3)が存在すればプロット
        if any(label in unique_labels for label in [1, 2, 3]):
            print('岩石ラベルをプロット中 (サイズマッピング)...')
            single_plot(data, time_zero_index, plot_mode='rocks', suffix='rocks_sized')
        
        # 空洞(4-6)が存在すればプロット
        if any(label in unique_labels for label in [4, 5, 6]):
            print('空洞ラベルをプロット中 (サイズマッピング)...')
            single_plot(data, time_zero_index, plot_mode='voids', suffix='voids_sized')
    else:
        print('ラベルが見つかりません。ラベルなしでプロットします。')
        single_plot(data, time_zero_index, plot_mode='none', suffix='no_labels')
    
    print('プロット完了。')