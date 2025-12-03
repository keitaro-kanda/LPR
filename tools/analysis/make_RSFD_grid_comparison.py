#!/usr/bin/env python3
# make_RSFD_grid_comparison.py
# ------------------------------------------------------------
# ラベル JSON から時間方向・距離方向のグリッド分割を行い、
# 各グリッド範囲内のRSFDプロットを作成し、
# 全範囲を1枚のプロットで比較する機能を持つツール
# ------------------------------------------------------------

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import MultipleLocator

# ------------------------------------------------------------------
# 補助関数定義
# ------------------------------------------------------------------
def none_to_nan(v):
    """None値をnp.nanに変換"""
    return np.nan if v is None else v

def format_p_value(p):
    """p値のフォーマットを補助する"""
    if p < 0.001:
        return "p < 0.001"
    else:
        return f"p={p:.3f}"

def save_legend_info_to_txt(output_path, legend_entries):
    """
    Legend情報をTXTファイルに保存

    Parameters:
    -----------
    output_path : str
        出力パス（拡張子なし）
    legend_entries : list of dict
        Legendエントリのリスト [{'label': str, 'color': str, 'linestyle': str}, ...]
    """
    with open(f'{output_path}_legend.txt', 'w', encoding='utf-8') as f:
        f.write('# Legend Information\n')
        f.write('# -------------------\n')
        for entry in legend_entries:
            f.write(f"Label: {entry['label']}\n")
            if 'color' in entry:
                f.write(f"  Color: {entry['color']}\n")
            if 'linestyle' in entry:
                f.write(f"  Linestyle: {entry['linestyle']}\n")
            if 'marker' in entry:
                f.write(f"  Marker: {entry['marker']}\n")
            if 'alpha' in entry:
                f.write(f"  Alpha: {entry['alpha']}\n")
            f.write('\n')
    print(f'Legend情報保存: {output_path}_legend.txt')

def save_legend_only_pdf(output_path, legend_entries):
    """
    Legend専用のPDFファイルを作成

    Parameters:
    -----------
    output_path : str
        出力パス（拡張子なし）
    legend_entries : list of dict
        Legendエントリのリスト [{'label': str, 'color': str, 'linestyle': str, 'marker': str}, ...]
    """
    fig = plt.figure(figsize=(8, len(legend_entries) * 0.5 + 1))
    ax = fig.add_subplot(111)

    # 空のプロットを作成し、legend用のハンドルを生成
    handles = []
    labels = []
    for entry in legend_entries:
        label = entry['label']
        color = entry.get('color', 'black')
        linestyle = entry.get('linestyle', '-')
        marker = entry.get('marker', '')
        linewidth = entry.get('linewidth', 1.5)
        alpha = entry.get('alpha', 1.0)

        # ダミーのプロット（表示されない）
        if marker and linestyle and linestyle != '':
            line, = ax.plot([], [], color=color, linestyle=linestyle,
                           marker=marker, linewidth=linewidth, alpha=alpha, label=label)
        elif marker:
            line, = ax.plot([], [], color=color, marker=marker,
                           linestyle='', alpha=alpha, label=label)
        else:
            line, = ax.plot([], [], color=color, linestyle=linestyle,
                           linewidth=linewidth, alpha=alpha, label=label)
        handles.append(line)
        labels.append(label)

    # 軸を非表示にする
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Legendのみを表示
    ax.legend(handles, labels, loc='center', fontsize=14,
             frameon=True, fancybox=True, shadow=False)

    plt.tight_layout()
    plt.savefig(f'{output_path}_legend.pdf', dpi=600, bbox_inches='tight')
    plt.close()

    print(f'Legend専用PDF保存: {output_path}_legend.pdf')

def calc_fitting_power_law(sizes, counts):
    """べき則フィッティングのみを実行"""
    # 対数変換
    mask = sizes > 0
    log_D = np.log(sizes[mask])
    log_N = np.log(counts[mask])

    # べき則フィッティング (Power-law: log N = r log D + log k)
    X_pow = sm.add_constant(log_D)
    model_pow = sm.OLS(log_N, X_pow)
    results_pow = model_pow.fit()

    log_k_pow, r_pow = results_pow.params
    k_pow = np.exp(log_k_pow)
    R2_pow = results_pow.rsquared
    r_pow_se = results_pow.bse[1]
    r_pow_t = results_pow.tvalues[1]
    r_pow_p = results_pow.pvalues[1]
    dof_pow = results_pow.df_resid
    n_pow = int(results_pow.nobs)

    # フィット曲線用に滑らかなサンプル点を生成
    D_fit = np.linspace(sizes.min(), sizes.max(), 200)
    N_pow_fit = k_pow * D_fit**r_pow

    # べき則の結果, D_fit
    return (k_pow, np.abs(r_pow), R2_pow, N_pow_fit, r_pow_t, r_pow_p, r_pow_se, n_pow, dof_pow), D_fit

def calc_fitting_power_law_area_normalized(sizes, counts, area):
    """面積規格化されたべき則フィッティングを実行"""
    # 岩石数を面積で規格化
    counts_normalized = counts / area

    # 対数変換
    mask = sizes > 0
    log_D = np.log(sizes[mask])
    log_N = np.log(counts_normalized[mask])

    # べき則フィッティング (Power-law: log N = r log D + log k)
    X_pow = sm.add_constant(log_D)
    model_pow = sm.OLS(log_N, X_pow)
    results_pow = model_pow.fit()

    log_k_pow, r_pow = results_pow.params
    k_pow = np.exp(log_k_pow)
    R2_pow = results_pow.rsquared
    r_pow_se = results_pow.bse[1]
    r_pow_t = results_pow.tvalues[1]
    r_pow_p = results_pow.pvalues[1]
    dof_pow = results_pow.df_resid
    n_pow = int(results_pow.nobs)

    # フィット曲線用に滑らかなサンプル点を生成
    D_fit = np.linspace(sizes.min(), sizes.max(), 200)
    N_pow_fit = k_pow * D_fit**r_pow

    # べき則の結果, D_fit, 規格化されたカウント
    return (k_pow, np.abs(r_pow), R2_pow, N_pow_fit, r_pow_t, r_pow_p, r_pow_se, n_pow, dof_pow), \
           D_fit, counts_normalized

def create_individual_rsfd_plot(x_data, y_data, xlabel, ylabel, output_path,
                               scale_type='linear', fit_line=None,
                               dpi_png=300, dpi_pdf=600):
    """
    個別範囲のRSFDプロットを作成・保存する関数

    Parameters:
    -----------
    x_data, y_data : array
        プロットするデータ
    xlabel, ylabel : str
        軸ラベル
    output_path : str
        出力パス（拡張子なし）
    scale_type : str
        'linear' or 'loglog'
    fit_line : dict, optional
        フィット曲線 {'x': x, 'y': y, 'label': label, 'params': dict}
    dpi_png, dpi_pdf : int
        解像度
    """
    plt.figure(figsize=(10, 8))

    # データプロット
    plt.plot(x_data, y_data, marker='o', linestyle='-', linewidth=1.5, color='blue')

    # フィット曲線の追加
    if fit_line:
        plt.plot(fit_line['x'], fit_line['y'],
                linestyle='--', linewidth=1.5, color='red',
                label=fit_line.get('label', ''))

    # 軸スケール設定
    if scale_type == 'loglog':
        plt.xscale('log')
        plt.yscale('log')

    # 軸ラベルとグリッド
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(labelsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)

    # y軸のtick設定（最大値1-20の場合は2刻みに固定）
    ax = plt.gca()
    ylim = ax.get_ylim()
    if scale_type == 'linear' and 1 <= ylim[1] <= 20:
        ax.yaxis.set_major_locator(MultipleLocator(2))

    # Legendの表示
    if fit_line:
        plt.legend(fontsize=14)

    plt.tight_layout()

    # 保存
    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)
    plt.close()

    print(f'個別プロット保存: {output_path}.png')

def create_comparison_plot(ranges_data_list, xlabel, ylabel, output_path,
                          scale_type='linear', dpi_png=300, dpi_pdf=600):
    """
    複数範囲のRSFDを1つのプロットに重ねて表示する関数（Legend別出力版）

    Parameters:
    -----------
    ranges_data_list : list of dict
        各範囲のデータリスト。各要素は以下のキーを持つ辞書:
        {
            'x_data': array, 'y_data': array,
            'label': str, 'color': str, 'marker': str, 'alpha': float,
            'fit_x': array, 'fit_y': array
        }
    xlabel, ylabel : str
        軸ラベル
    output_path : str
        出力パス（拡張子なし）
    scale_type : str
        'linear' or 'loglog'
    dpi_png, dpi_pdf : int
        解像度
    """
    plt.figure(figsize=(10, 8))

    # Legend情報を格納するリスト
    legend_entries = []

    # 各範囲のデータとフィット曲線をプロット
    for range_data in ranges_data_list:
        color = range_data['color']
        marker = range_data['marker']
        alpha = range_data['alpha']
        label = range_data['label']

        # データプロット
        plt.plot(range_data['x_data'], range_data['y_data'],
                marker=marker, linestyle='', color=color, alpha=alpha)

        # フィット曲線プロット
        plt.plot(range_data['fit_x'], range_data['fit_y'],
                linestyle='--', linewidth=1.5, color=color, alpha=alpha)

        # Legend情報を保存
        legend_entries.append({
            'label': label,
            'color': color,
            'marker': marker,
            'linestyle': '--',
            'alpha': alpha,
            'linewidth': 1.5
        })

    # 軸スケール設定
    if scale_type == 'loglog':
        plt.xscale('log')
        plt.yscale('log')

    # 軸ラベルとグリッド
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(labelsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)

    # y軸のtick設定
    ax = plt.gca()
    ylim = ax.get_ylim()
    if scale_type == 'linear' and 1 <= ylim[1] <= 20:
        ax.yaxis.set_major_locator(MultipleLocator(2))

    plt.tight_layout()

    # 保存
    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)
    plt.close()

    print(f'比較プロット保存: {output_path}.png')

    # Legend情報をTXT・PDF形式で別途保存
    save_legend_info_to_txt(output_path, legend_entries)
    save_legend_only_pdf(output_path, legend_entries)

def save_statistics_to_txt(output_path, grid_statistics):
    """
    各グリッド範囲の統計情報をTXTファイルに保存

    Parameters:
    -----------
    output_path : str
        出力ファイルパス
    grid_statistics : list of dict
        各グリッドの統計情報
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('# Grid RSFD Statistics\n')
        f.write('# =====================\n\n')

        for stat in grid_statistics:
            f.write(f"Grid: {stat['label']}\n")
            f.write(f"  Time range: {stat['time_min']:.1f} - {stat['time_max']:.1f} ns\n")
            f.write(f"  Distance range: {stat['dist_min']:.1f} - {stat['dist_max']:.1f} m\n")
            f.write(f"  Area: {stat['area']:.2f} m²\n")
            f.write(f"  Total rocks: {stat['total_rocks']}\n")
            f.write(f"  Group 1 rocks: {stat['group1_rocks']}\n")
            f.write(f"  Group 2 rocks: {stat['group2_rocks']}\n")
            f.write(f"  Group 3 rocks: {stat['group3_rocks']}\n")
            f.write(f"  Power-law fit (non-normalized):\n")
            f.write(f"    k = {stat['k_pow']:.4e}\n")
            f.write(f"    r = {stat['r_pow']:.4f}\n")
            f.write(f"    R² = {stat['R2_pow']:.4f}\n")
            f.write(f"    p-value: {stat['p_value_pow']}\n")
            if 'k_pow_norm' in stat:
                f.write(f"  Power-law fit (area-normalized):\n")
                f.write(f"    k = {stat['k_pow_norm']:.4e}\n")
                f.write(f"    r = {stat['r_pow_norm']:.4f}\n")
                f.write(f"    R² = {stat['R2_pow_norm']:.4f}\n")
                f.write(f"    p-value: {stat['p_value_pow_norm']}\n")
            f.write('\n')

    print(f'統計情報保存: {output_path}')

def create_grid_subplot_comparison(grid_data_dict, fit_params_dict, num_time_bins, num_dist_bins,
                                   xlabel, ylabel, output_path, scale_type='linear',
                                   dpi_png=300, dpi_pdf=600):
    """
    グリッド配置でsubplotを作成し、各グリッドのRSFDを表示

    Parameters:
    -----------
    grid_data_dict : dict
        キーが(time_idx, dist_idx)のタプル、値がグリッドデータの辞書
    fit_params_dict : dict
        キーが(time_idx, dist_idx)のタプル、値がフィッティングパラメータの辞書
    num_time_bins : int
        時間方向の分割数（行数）
    num_dist_bins : int
        距離方向の分割数（列数）
    xlabel, ylabel : str
        軸ラベル
    output_path : str
        出力パス（拡張子なし）
    scale_type : str
        'linear' or 'loglog'
    dpi_png, dpi_pdf : int
        解像度
    """
    # 全データから軸範囲を計算
    all_x_data = []
    all_y_data = []
    all_fit_y = []

    for grid_data in grid_data_dict.values():
        all_x_data.extend(grid_data['x_data'])
        all_y_data.extend(grid_data['y_data'])
        all_fit_y.extend(grid_data['fit_y'])

    if len(all_x_data) > 0:
        if scale_type == 'loglog':
            # logスケールの場合は正の値のみ考慮
            x_positive = [x for x in all_x_data if x > 0]
            y_positive = [y for y in all_y_data + all_fit_y if y > 0]

            if len(x_positive) > 0 and len(y_positive) > 0:
                x_min, x_max = min(x_positive) * 0.8, max(x_positive) * 1.2
                y_min, y_max = min(y_positive) * 0.5, max(y_positive) * 2.0
            else:
                x_min, x_max, y_min, y_max = None, None, None, None
        else:
            # linearスケールの場合
            x_min, x_max = min(all_x_data) * 0.95, max(all_x_data) * 1.05
            y_min, y_max = min(all_y_data + all_fit_y) * 0.95, max(all_y_data + all_fit_y) * 1.05
    else:
        x_min, x_max, y_min, y_max = None, None, None, None

    # Figure作成
    _, axes = plt.subplots(num_time_bins, num_dist_bins,
                           figsize=(4 * num_dist_bins, 3.5 * num_time_bins),
                           squeeze=False)

    # 各subplotにデータをプロット
    for i in range(num_time_bins):
        for j in range(num_dist_bins):
            ax = axes[i, j]

            # グリッドデータの取得
            if (i, j) in grid_data_dict:
                grid_data = grid_data_dict[(i, j)]
                fit_params = fit_params_dict.get((i, j), {})

                # データプロット
                ax.plot(grid_data['x_data'], grid_data['y_data'],
                       marker='o', linestyle='-', linewidth=1.5, color='blue', markersize=4)

                # フィット曲線プロット
                ax.plot(grid_data['fit_x'], grid_data['fit_y'],
                       linestyle='--', linewidth=1.5, color='red')

                # 軸スケール設定
                if scale_type == 'loglog':
                    ax.set_xscale('log')
                    ax.set_yscale('log')

                # 軸範囲を統一
                if x_min is not None and x_max is not None:
                    ax.set_xlim(x_min, x_max)
                if y_min is not None and y_max is not None:
                    ax.set_ylim(y_min, y_max)

                # グリッドラベルを表示（右上）
                ax.text(0.85, 0.95, f'T{i+1}D{j+1}',
                       transform=ax.transAxes, fontsize=10, fontweight='bold',
                       verticalalignment='top', bbox=dict(boxstyle='round',
                       facecolor='wheat', alpha=0.5))

                # フィッティングパラメータを表示（右中央）
                if fit_params:
                    k = fit_params.get('k', 0)
                    r = fit_params.get('r', 0)
                    p_str = fit_params.get('p_str', '')

                    param_text = f'k={k:.3e}\nr={r:.3f}\n{p_str}'
                    ax.text(0.60, 0.70, param_text,
                           transform=ax.transAxes, fontsize=10,
                           verticalalignment='bottom', horizontalalignment='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            else:
                # データがない場合は空のプロット
                ax.text(0.5, 0.5, 'No Data',
                       transform=ax.transAxes, fontsize=12,
                       horizontalalignment='center', verticalalignment='center',
                       color='gray')
                ax.set_xticks([])
                ax.set_yticks([])

            # グリッド表示
            ax.grid(True, linestyle='--', alpha=0.3)

            # 軸ラベル（外側のみ）
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=14)
            if i == num_time_bins - 1:
                ax.set_xlabel(xlabel, fontsize=14)

            # Tick labelサイズ
            ax.tick_params(labelsize=10)

    plt.tight_layout()

    # 保存
    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)
    plt.close()

    print(f'グリッドsubplotプロット保存: {output_path}.png')

def generate_color_marker_combinations(num_time_bins, num_dist_bins):
    """
    時間範囲と距離範囲に応じた色とマーカーの組み合わせを生成

    Parameters:
    -----------
    num_time_bins : int
        時間方向の分割数
    num_dist_bins : int
        距離方向の分割数

    Returns:
    --------
    list of dict
        各グリッドに対応する色・マーカー・透明度の組み合わせ
    """
    # 基本色のリスト（時間範囲ごとに異なる色）
    base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
                   'pink', 'olive', 'cyan', 'magenta']

    # マーカーのリスト（距離範囲ごとに異なるマーカー）
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # 色の濃度（距離範囲ごとに変化）
    alphas = np.linspace(1.0, 0.4, num_dist_bins)

    combinations = []
    for i in range(num_time_bins):
        base_color = base_colors[i % len(base_colors)]
        for j in range(num_dist_bins):
            marker = markers[j % len(markers)]
            alpha = alphas[j]
            combinations.append({
                'color': base_color,
                'marker': marker,
                'alpha': alpha
            })

    return combinations

# ------------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------------
print('=== RSFD Grid Comparison Tool ===')
print('時間・距離方向のグリッド分割によるRSFD比較ツール\n')

# ------------------------------------------------------------------
# 1. 入力ファイルチェック
# ------------------------------------------------------------------
print('検出された岩石のラベルデータファイル(.json)のパスを入力してください:')
data_path = input().strip()
if not (os.path.exists(data_path) and data_path.lower().endswith('.json')):
    raise FileNotFoundError('正しい .json ファイルを指定してください。')

# ------------------------------------------------------------------
# 2. グリッド分割パラメータの入力
# ------------------------------------------------------------------
print('\n=== グリッド分割パラメータ ===')
time_bin_width = float(input('時間方向の分割幅 [ns] を入力してください: ').strip())
dist_bin_width = float(input('距離方向の分割幅 [m] を入力してください: ').strip())

if time_bin_width <= 0 or dist_bin_width <= 0:
    raise ValueError('分割幅は正の値を指定してください。')

# ------------------------------------------------------------------
# 3. JSONデータ読み込み
# ------------------------------------------------------------------
print('\nデータ読み込み中...')
with open(data_path, 'r') as f:
    results = json.load(f).get('results', {})

x_all = np.array([v['x'] for v in results.values()])
y_all = np.array([v['y'] for v in results.values()])
lab_all = np.array([v['label'] for v in results.values()], dtype=int)
time_top_all = np.array([none_to_nan(v['time_top']) for v in results.values()], dtype=float)
time_bottom_all = np.array([none_to_nan(v['time_bottom']) for v in results.values()], dtype=float)

print(f'ラベルデータ読み込み完了: {len(lab_all)}個')

# ------------------------------------------------------------------
# 4. グリッド範囲の計算
# ------------------------------------------------------------------
# 時間方向：データ全体の最小値から開始
# Group1はy座標、Group2-3はtime_topを使用
time_values_group1 = y_all[lab_all == 1]
time_values_others = time_top_all[(lab_all != 1) & (~np.isnan(time_top_all))]
time_values_all = np.concatenate([time_values_group1, time_values_others])

time_min_data = np.min(time_values_all)
time_max_data = np.max(time_values_all)

# 距離方向：x=0から開始
dist_min_data = 0.0
dist_max_data = np.max(x_all)

# グリッド数の計算（端数を含める）
num_time_bins = int(np.ceil((time_max_data - time_min_data) / time_bin_width))
num_dist_bins = int(np.ceil((dist_max_data - dist_min_data) / dist_bin_width))

print(f'\n時間範囲: {time_min_data:.2f} - {time_max_data:.2f} ns')
print(f'距離範囲: {dist_min_data:.2f} - {dist_max_data:.2f} m')
print(f'時間方向分割数: {num_time_bins}')
print(f'距離方向分割数: {num_dist_bins}')
print(f'総グリッド数: {num_time_bins * num_dist_bins}')

# 色・マーカーの組み合わせを生成
color_marker_combinations = generate_color_marker_combinations(num_time_bins, num_dist_bins)

# ------------------------------------------------------------------
# 5. 出力ディレクトリの作成
# ------------------------------------------------------------------

base_dir = os.path.join(os.path.dirname(os.path.dirname(data_path)),
                        f'RSFD_grid_time{time_bin_width:.0f}ns_dist{dist_bin_width:.0f}m')
os.makedirs(base_dir, exist_ok=True)

individual_dir = os.path.join(base_dir, 'individual_plots')
comparison_dir = os.path.join(base_dir, 'comparison_plots')
os.makedirs(individual_dir, exist_ok=True)
os.makedirs(comparison_dir, exist_ok=True)

print(f'\n出力ディレクトリ: {base_dir}')

# ------------------------------------------------------------------
# 6. 各グリッドの処理
# ------------------------------------------------------------------
print('\n=== グリッド処理開始 ===')

# 物理定数
epsilon_regolith = 4.5  # 月面レゴリスの比誘電率
epsilon_rock = 9.0     # 岩石の比誘電率
c = 299_792_458  # [m/s]

# 各グリッドのデータを格納するリスト
all_grids_data_non_normalized = []  # 非規格化
all_grids_data_area_normalized = []  # 面積規格化
grid_statistics = []

# subplot用のグリッドデータ辞書（キー: (time_idx, dist_idx)）
grid_data_dict_non_normalized = {}
grid_data_dict_area_normalized = {}

# subplot用のフィッティングパラメータ辞書（キー: (time_idx, dist_idx)）
fit_params_dict_non_normalized = {}
fit_params_dict_area_normalized = {}

grid_idx = 0
for i in range(num_time_bins):
    for j in range(num_dist_bins):
        grid_idx += 1

        # グリッド範囲の計算
        time_min = time_min_data + i * time_bin_width
        time_max = time_min_data + (i + 1) * time_bin_width
        dist_min = dist_min_data + j * dist_bin_width
        dist_max = dist_min_data + (j + 1) * dist_bin_width

        # 最後の区間の調整
        if i == num_time_bins - 1:
            time_max = time_max_data
        if j == num_dist_bins - 1:
            dist_max = dist_max_data

        print(f'\nGrid {grid_idx}/{num_time_bins * num_dist_bins}: '
              f'Time {time_min:.1f}-{time_max:.1f} ns, Dist {dist_min:.1f}-{dist_max:.1f} m')

        # データのコピー
        x = x_all.copy()
        y = y_all.copy()
        lab = lab_all.copy()
        time_top = time_top_all.copy()
        time_bottom = time_bottom_all.copy()

        # データ範囲フィルタリング
        # 時間範囲フィルタ
        mask_group1 = (lab == 1) & (y >= time_min) & (y <= time_max)
        mask_others = (lab != 1) & (time_top >= time_min) & (time_top <= time_max)
        time_mask = mask_group1 | mask_others

        x = x[time_mask]
        y = y[time_mask]
        lab = lab[time_mask]
        time_top = time_top[time_mask]
        time_bottom = time_bottom[time_mask]

        # 距離範囲フィルタ
        dist_mask = (x >= dist_min) & (x <= dist_max)
        x = x[dist_mask]
        y = y[dist_mask]
        lab = lab[dist_mask]
        time_top = time_top[dist_mask]
        time_bottom = time_bottom[dist_mask]

        num_rocks = len(lab)
        print(f'  岩石数: {num_rocks}')

        # データが少なすぎる場合はスキップ
        if num_rocks < 3:
            print(f'  警告: データ数が不足しているためスキップします')
            continue

        # 面積計算（時間範囲×距離範囲）
        # 時間を距離に変換: depth = time * c / (2 * sqrt(epsilon_r))
        depth_min = time_min * 1e-9 * c / (2 * np.sqrt(epsilon_regolith))
        depth_max = time_max * 1e-9 * c / (2 * np.sqrt(epsilon_regolith))
        area = (depth_max - depth_min) * (dist_max - dist_min)
        print(f'  面積: {area:.2f} m²')

        # サイズ配列を作成（Group1, Group2, Group3全て含める）
        # Group1: 1cm固定
        num_group1 = int(np.sum(lab == 1))
        size_label1 = np.full(num_group1, 1.0)

        # Group2: 6cm固定
        mask2_valid = (lab == 2) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
        sizes_group2 = (time_bottom[mask2_valid] - time_top[mask2_valid]) * 1e-9 * c / np.sqrt(epsilon_rock) * 0.5 * 100

        # 浮動小数点誤差を排除
        sizes_group2 = np.round(sizes_group2, decimals=3)
        num_group2 = len(sizes_group2)

        # Group3: 計算値
        mask3_valid = (lab == 3) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
        sizes_group3 = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(epsilon_rock) * 0.5 * 100

        # 浮動小数点誤差を排除
        sizes_group3 = np.round(sizes_group3, decimals=3)
        num_group3 = len(sizes_group3)

        print(f'  Group1: {num_group1}, Group2: {num_group2}, Group3: {num_group3}')

        # データが不足している場合はスキップ
        total_samples = num_group1 + num_group2 + num_group3
        if total_samples < 3:
            print(f'  警告: データ数が不足しているためスキップします')
            continue

        # Group1-2-3統合データ
        all_sizes = np.concatenate([size_label1, sizes_group2, sizes_group3])
        unique_sizes = np.sort(np.unique(all_sizes))
        cum_counts = np.array([np.sum(all_sizes >= s) for s in unique_sizes])

        # フィッティング（非規格化）
        (k_pow, r_pow, R2_pow, N_pow_fit, t_pow, p_pow, se_pow, n_pow, dof_pow), D_fit = \
            calc_fitting_power_law(unique_sizes, cum_counts)

        # フィッティング（面積規格化）
        (k_pow_norm, r_pow_norm, R2_pow_norm, N_pow_fit_norm, t_pow_norm, p_pow_norm,
         se_pow_norm, n_pow_norm, dof_pow_norm), D_fit_norm, cum_counts_normalized = \
            calc_fitting_power_law_area_normalized(unique_sizes, cum_counts, area)

        # グリッドラベル
        grid_label = f'T{i+1}D{j+1} ({time_min:.0f}-{time_max:.0f}ns, {dist_min:.0f}-{dist_max:.0f}m)'

        # 色・マーカー・透明度の取得
        style = color_marker_combinations[(i * num_dist_bins + j) % len(color_marker_combinations)]

        # 非規格化データを保存（比較プロット用リスト）
        all_grids_data_non_normalized.append({
            'x_data': unique_sizes,
            'y_data': cum_counts,
            'fit_x': D_fit,
            'fit_y': N_pow_fit,
            'label': grid_label,
            'color': style['color'],
            'marker': style['marker'],
            'alpha': style['alpha'],
            'fit_params': {
                'k': k_pow,
                'r': r_pow,
                'R2': R2_pow,
                'p_str': format_p_value(p_pow)
            }
        })

        # 非規格化データを保存（subplot用辞書）
        grid_data_dict_non_normalized[(i, j)] = {
            'x_data': unique_sizes,
            'y_data': cum_counts,
            'fit_x': D_fit,
            'fit_y': N_pow_fit
        }

        # 非規格化フィッティングパラメータを保存（subplot用辞書）
        fit_params_dict_non_normalized[(i, j)] = {
            'k': k_pow,
            'r': r_pow,
            'R2': R2_pow,
            'p_str': format_p_value(p_pow)
        }

        # 面積規格化データを保存（比較プロット用リスト）
        all_grids_data_area_normalized.append({
            'x_data': unique_sizes,
            'y_data': cum_counts_normalized,
            'fit_x': D_fit_norm,
            'fit_y': N_pow_fit_norm,
            'label': grid_label,
            'color': style['color'],
            'marker': style['marker'],
            'alpha': style['alpha'],
            'fit_params': {
                'k': k_pow_norm,
                'r': r_pow_norm,
                'R2': R2_pow_norm,
                'p_str': format_p_value(p_pow_norm)
            }
        })

        # 面積規格化データを保存（subplot用辞書）
        grid_data_dict_area_normalized[(i, j)] = {
            'x_data': unique_sizes,
            'y_data': cum_counts_normalized,
            'fit_x': D_fit_norm,
            'fit_y': N_pow_fit_norm
        }

        # 面積規格化フィッティングパラメータを保存（subplot用辞書）
        fit_params_dict_area_normalized[(i, j)] = {
            'k': k_pow_norm,
            'r': r_pow_norm,
            'R2': R2_pow_norm,
            'p_str': format_p_value(p_pow_norm)
        }

        # 統計情報を保存
        grid_statistics.append({
            'label': grid_label,
            'time_min': time_min,
            'time_max': time_max,
            'dist_min': dist_min,
            'dist_max': dist_max,
            'area': area,
            'total_rocks': num_rocks,
            'group1_rocks': num_group1,
            'group2_rocks': num_group2,
            'group3_rocks': num_group3,
            'k_pow': k_pow,
            'r_pow': r_pow,
            'R2_pow': R2_pow,
            'p_value_pow': format_p_value(p_pow),
            'k_pow_norm': k_pow_norm,
            'r_pow_norm': r_pow_norm,
            'R2_pow_norm': R2_pow_norm,
            'p_value_pow_norm': format_p_value(p_pow_norm)
        })

        # 個別プロットの作成
        # 非規格化 linear-linear
        output_path_individual = os.path.join(individual_dir,
            f'grid_{i+1:02d}_{j+1:02d}_linear_non_normalized')
        create_individual_rsfd_plot(
            unique_sizes, cum_counts,
            'Rock Size D [cm]', 'Cumulative Number N',
            output_path_individual,
            scale_type='linear',
            fit_line={
                'x': D_fit, 'y': N_pow_fit,
                'label': f'Power-law: k={k_pow:.2e}, r={r_pow:.3f}, R²={R2_pow:.4f}'
            }
        )

        # 非規格化 log-log
        output_path_individual = os.path.join(individual_dir,
            f'grid_{i+1:02d}_{j+1:02d}_loglog_non_normalized')
        create_individual_rsfd_plot(
            unique_sizes, cum_counts,
            'Rock Size D [cm]', 'Cumulative Number N',
            output_path_individual,
            scale_type='loglog',
            fit_line={
                'x': D_fit, 'y': N_pow_fit,
                'label': f'Power-law: k={k_pow:.2e}, r={r_pow:.3f}, R²={R2_pow:.4f}'
            }
        )

        # 面積規格化 linear-linear
        output_path_individual = os.path.join(individual_dir,
            f'grid_{i+1:02d}_{j+1:02d}_linear_area_normalized')
        create_individual_rsfd_plot(
            unique_sizes, cum_counts_normalized,
            'Rock Size D [cm]', 'Cumulative number of rocks /m²',
            output_path_individual,
            scale_type='linear',
            fit_line={
                'x': D_fit_norm, 'y': N_pow_fit_norm,
                'label': f'Power-law: k={k_pow_norm:.2e}, r={r_pow_norm:.3f}, R²={R2_pow_norm:.4f}'
            }
        )

        # 面積規格化 log-log
        output_path_individual = os.path.join(individual_dir,
            f'grid_{i+1:02d}_{j+1:02d}_loglog_area_normalized')
        create_individual_rsfd_plot(
            unique_sizes, cum_counts_normalized,
            'Rock Size D [cm]', 'Cumulative number of rocks /m²',
            output_path_individual,
            scale_type='loglog',
            fit_line={
                'x': D_fit_norm, 'y': N_pow_fit_norm,
                'label': f'Power-law: k={k_pow_norm:.2e}, r={r_pow_norm:.3f}, R²={R2_pow_norm:.4f}'
            }
        )

# ------------------------------------------------------------------
# 7. 比較プロットの作成
# ------------------------------------------------------------------
if len(all_grids_data_non_normalized) > 0:
    print('\n=== 比較プロット作成 ===')

    # 非規格化 linear-linear
    output_path_comparison = os.path.join(comparison_dir, 'comparison_linear_non_normalized')
    create_comparison_plot(
        all_grids_data_non_normalized,
        'Rock Size D [cm]', 'Cumulative Number N',
        output_path_comparison,
        scale_type='linear'
    )

    # 非規格化 log-log
    output_path_comparison = os.path.join(comparison_dir, 'comparison_loglog_non_normalized')
    create_comparison_plot(
        all_grids_data_non_normalized,
        'Rock Size D [cm]', 'Cumulative Number N',
        output_path_comparison,
        scale_type='loglog'
    )

    # 面積規格化 linear-linear
    output_path_comparison = os.path.join(comparison_dir, 'comparison_linear_area_normalized')
    create_comparison_plot(
        all_grids_data_area_normalized,
        'Rock Size D [cm]', 'Cumulative number of rocks /m²',
        output_path_comparison,
        scale_type='linear'
    )

    # 面積規格化 log-log
    output_path_comparison = os.path.join(comparison_dir, 'comparison_loglog_area_normalized')
    create_comparison_plot(
        all_grids_data_area_normalized,
        'Rock Size D [cm]', 'Cumulative number of rocks /m²',
        output_path_comparison,
        scale_type='loglog'
    )

    # ------------------------------------------------------------------
    # グリッドsubplot比較プロットの作成
    # ------------------------------------------------------------------
    print('\n=== グリッドsubplot比較プロット作成 ===')

    # 非規格化 log-log
    output_path_subplot = os.path.join(comparison_dir, 'grid_subplot_loglog_non_normalized')
    create_grid_subplot_comparison(
        grid_data_dict_non_normalized, fit_params_dict_non_normalized,
        num_time_bins, num_dist_bins,
        'Rock Size D [cm]', 'Cumulative Number N',
        output_path_subplot,
        scale_type='loglog'
    )

    # 面積規格化 log-log
    output_path_subplot = os.path.join(comparison_dir, 'grid_subplot_loglog_area_normalized')
    create_grid_subplot_comparison(
        grid_data_dict_area_normalized, fit_params_dict_area_normalized,
        num_time_bins, num_dist_bins,
        'Rock Size D [cm]', 'Cumulative Number Density N [/m²]',
        output_path_subplot,
        scale_type='loglog'
    )
else:
    print('\n警告: 有効なグリッドデータが見つかりませんでした。')

# ------------------------------------------------------------------
# 8. 統計情報の保存
# ------------------------------------------------------------------
if len(grid_statistics) > 0:
    stats_path = os.path.join(base_dir, 'grid_statistics.txt')
    save_statistics_to_txt(stats_path, grid_statistics)

print('\n=== 処理完了 ===')
print(f'出力ディレクトリ: {base_dir}')
