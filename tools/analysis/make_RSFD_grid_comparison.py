#!/usr/bin/env python3
# make_RSFD_grid_comparison.py
# ------------------------------------------------------------
# ラベル JSON から時間方向・距離方向のグリッド分割を行い、
# 各グリッド範囲内のRSFDプロットを作成し、
# 全範囲を1枚のプロットで比較する機能を持つツール
# ------------------------------------------------------------

import json
import os
import glob
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

def find_label_json_files(bscan_dir):
    """
    B-scanと同階層のecho_labelsディレクトリからJSONファイルを検索

    Parameters:
    -----------
    bscan_dir : str
        B-scanファイルのディレクトリパス

    Returns:
    --------
    list: JSONファイルのパスリスト
    """
    echo_labels_dir = os.path.join(bscan_dir, 'echo_labels')

    if not os.path.exists(echo_labels_dir):
        return []

    # echo_labelsディレクトリ内の全JSONファイルを検索
    label_files = glob.glob(os.path.join(echo_labels_dir, '*.json'))

    return sorted(label_files)

def select_label_file(label_files):
    """
    複数のlabel.jsonファイルがある場合、ユーザーに選択させる

    Parameters:
    -----------
    label_files : list
        label.jsonファイルのパスリスト

    Returns:
    --------
    str: 選択されたファイルパス
    """
    if len(label_files) == 0:
        raise FileNotFoundError('echo_labelsディレクトリにJSONファイルが見つかりません。')

    if len(label_files) == 1:
        print(f'label.jsonファイルを自動検出: {label_files[0]}')
        return label_files[0]

    print('\n複数のlabel.jsonファイルが見つかりました。使用するファイルを選択してください:')
    for i, path in enumerate(label_files):
        # 相対パスで表示
        rel_path = os.path.relpath(path, os.path.dirname(os.path.dirname(path)))
        print(f'  {i + 1}: {rel_path}')

    while True:
        try:
            choice = int(input(f'選択 (1-{len(label_files)}): ').strip())
            if 1 <= choice <= len(label_files):
                return label_files[choice - 1]
            else:
                print(f'1から{len(label_files)}の数字を入力してください。')
        except ValueError:
            print('数字を入力してください。')


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
        plt.legend(fontsize=16)

    plt.tight_layout()

    # 保存
    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)
    plt.close()

    print(f'個別プロット保存: {output_path}.png')

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
        f.write('# Grid RSFD Statistics (Area-Normalized)\n')
        f.write('# ======================================\n\n')

        for stat in grid_statistics:
            f.write(f"Grid: {stat['label']}\n")
            f.write(f"  Time range: {stat['time_min']:.1f} - {stat['time_max']:.1f} ns\n")
            f.write(f"  Distance range: {stat['dist_min']:.1f} - {stat['dist_max']:.1f} m\n")
            f.write(f"  Area: {stat['area']:.2f} m²\n")
            f.write(f"  Total rocks: {stat['total_rocks']}\n")
            f.write(f"  Group 1 rocks: {stat['group1_rocks']}\n")
            f.write(f"  Group 2 rocks: {stat['group2_rocks']}\n")
            f.write(f"  Group 3 rocks: {stat['group3_rocks']}\n")
            f.write(f"  Power-law fit (area-normalized):\n")
            f.write(f"    k = {stat['k_pow_norm']:.4e}\n")
            f.write(f"    r = {stat['r_pow_norm']:.4f}\n")
            f.write(f"    R² = {stat['R2_pow_norm']:.4f}\n")
            f.write(f"    p-value: {stat['p_value_pow_norm']}\n")
            f.write('\n')

    print(f'統計情報保存: {output_path}')

def create_grid_subplot_comparison(grid_data_dict, fit_params_dict, num_time_bins, num_dist_bins,
                                   xlabel, ylabel, output_path, scale_type='loglog',
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
    fig, axes = plt.subplots(num_time_bins, num_dist_bins,
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
                ax.text(0.95, 0.95, f'T{i+1}D{j+1}',
                       transform=ax.transAxes, fontsize=16, fontweight='bold',
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                # フィッティング式を表示（左下）
                if fit_params:
                    k = fit_params.get('k', 0)
                    r = fit_params.get('r', 0)
                    R2 = fit_params.get('R2', 0)
                    p_str = fit_params.get('p_str', '')

                    # N = k D^(-r) 形式で表示
                    fit_eq = f'$N = {k:.2e} \\cdot D^{{-{r:.2f}}}$'
                    stats_text = f'$R^2 = {R2:.3f}$, {p_str}'
                    param_text = f'{fit_eq}\n{stats_text}'
                    ax.text(0.05, 0.05, param_text,
                            transform=ax.transAxes, fontsize=14, color='red')

            else:
                # データがない場合は空のプロット
                ax.text(0.5, 0.5, 'No Data',
                       transform=ax.transAxes, fontsize=16,
                       horizontalalignment='center', verticalalignment='center',
                       color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(labelsize=14)

            # グリッド表示
            ax.grid(True, linestyle='--', alpha=0.3)

            # Tick labelサイズ
            ax.tick_params(labelsize=10)

    # 全体で1つのxlabel/ylabelを配置
    fig.text(0.5, 0.02, xlabel, ha='center', fontsize=18, fontweight='bold')
    fig.text(0.02, 0.5, ylabel, va='center', rotation='vertical', fontsize=18, fontweight='bold')

    plt.tight_layout(rect=[0.03, 0.03, 1, 1])

    # 保存
    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)
    plt.close()

    print(f'グリッドsubplotプロット保存: {output_path}.png')

def create_grid_subplot_rock_counts(grid_data_dict, rock_counts_dict, num_time_bins, num_dist_bins,
                                     xlabel, ylabel, output_path, scale_type='loglog',
                                     dpi_png=300, dpi_pdf=600):
    """
    グリッド配置でsubplotを作成し、各グリッドのRSFDと岩石数を表示

    Parameters:
    -----------
    grid_data_dict : dict
        キーが(time_idx, dist_idx)のタプル、値がグリッドデータの辞書
    rock_counts_dict : dict
        キーが(time_idx, dist_idx)のタプル、値が岩石数の辞書
        {'total': int, 'group1': int, 'group2': int, 'group3': int}
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
    fig, axes = plt.subplots(num_time_bins, num_dist_bins,
                             figsize=(4 * num_dist_bins, 3.5 * num_time_bins),
                             squeeze=False)

    # 各subplotにデータをプロット
    for i in range(num_time_bins):
        for j in range(num_dist_bins):
            ax = axes[i, j]

            # グリッドデータの取得
            if (i, j) in grid_data_dict:
                grid_data = grid_data_dict[(i, j)]
                rock_counts = rock_counts_dict.get((i, j), {})

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
                ax.text(0.95, 0.95, f'T{i+1}D{j+1}',
                       transform=ax.transAxes, fontsize=16, fontweight='bold',
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                # 岩石数を表示（左下）
                if rock_counts:
                    total = rock_counts.get('total', 0)
                    g1 = rock_counts.get('group1', 0)
                    g2 = rock_counts.get('group2', 0)
                    g3 = rock_counts.get('group3', 0)

                    count_text = f'Total: {total}\nG1: {g1}, G2: {g2}, G3: {g3}'
                    ax.text(0.05, 0.05, count_text,
                           transform=ax.transAxes, fontsize=14,
                           verticalalignment='bottom', horizontalalignment='left',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            else:
                # データがない場合は空のプロット
                ax.text(0.5, 0.5, 'No Data',
                       transform=ax.transAxes, fontsize=16,
                       horizontalalignment='center', verticalalignment='center',
                       color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(labelsize=14)

            # グリッド表示
            ax.grid(True, linestyle='--', alpha=0.3)

            # Tick labelサイズ
            ax.tick_params(labelsize=10)

    # 全体で1つのxlabel/ylabelを配置
    fig.text(0.5, 0.02, xlabel, ha='center', fontsize=18, fontweight='bold')
    fig.text(0.02, 0.5, ylabel, va='center', rotation='vertical', fontsize=18, fontweight='bold')

    plt.tight_layout(rect=[0.03, 0.03, 1, 1])

    # 保存
    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)
    plt.close()

    print(f'グリッドsubplotプロット（岩石数表示版）保存: {output_path}.png')

def create_bscan_with_grid_lines(bscan_data, time_bins, dist_bins, output_path,
                                  fit_params_dict=None, rock_counts_dict=None,
                                  sample_interval=0.312500e-9, trace_interval=3.6e-2,
                                  epsilon_r=4.5, c=299792458, dpi_png=300, dpi_pdf=600):
    """
    B-scanプロット上にグリッド分割の境界線を表示する関数

    Parameters:
    -----------
    bscan_data : ndarray
        B-scanデータ（2D配列、行がサンプル、列がトレース）
    time_bins : list of tuple
        時間方向の分割境界 [(time_min1, time_max1), (time_min2, time_max2), ...]
    dist_bins : list of tuple
        距離方向の分割境界 [(dist_min1, dist_max1), (dist_min2, dist_max2), ...]
    output_path : str
        出力パス（拡張子なし）
    fit_params_dict : dict, optional
        フィッティングパラメータの辞書。キーは(time_idx, dist_idx)のタプル。
        各値は{'k': float, 'r': float, 'p_value': float}を含む辞書。
    rock_counts_dict : dict, optional
        岩石数の辞書。キーは(time_idx, dist_idx)のタプル。
        各値は{'total': int, 'group1': int, 'group2': int, 'group3': int}を含む辞書。
    sample_interval : float
        サンプル間隔 [s]
    trace_interval : float
        トレース間隔 [m]
    epsilon_r : float
        比誘電率
    c : float
        光速 [m/s]
    dpi_png, dpi_pdf : int
        解像度
    """
    # Font size standards (plot_Bscan.pyと同じ)
    font_medium = 18
    font_small = 16

    # Time zero検出（最初のトレースで最初の非NaN値を探す）
    first_trace = bscan_data[:, 0]
    first_non_nan_idx = np.where(~np.isnan(first_trace))[0]
    time_zero_idx = first_non_nan_idx[0] if len(first_non_nan_idx) > 0 else 0

    # 時間配列の計算
    time_array = (np.arange(bscan_data.shape[0]) - time_zero_idx) * sample_interval * 1e9  # [ns]

    # vmin/vmaxの設定（plot_Bscan.pyと同じロジック）
    vmin = -np.nanmax(np.abs(bscan_data)) / 10
    vmax = np.nanmax(np.abs(bscan_data)) / 10

    # Figure作成
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(111)

    # B-scanプロット
    im = ax.imshow(bscan_data, aspect='auto', cmap='seismic',
                   extent=[0, bscan_data.shape[1] * trace_interval,
                          time_array[-1], time_array[0]],
                   vmin=vmin, vmax=vmax)

    # グリッド範囲の取得
    time_min_grid = min([t[0] for t in time_bins])
    time_max_grid = max([t[1] for t in time_bins])
    dist_min_grid = min([d[0] for d in dist_bins])
    dist_max_grid = max([d[1] for d in dist_bins])

    # プロット範囲をグリッド範囲に設定（少し余裕を持たせる）
    time_margin = (time_max_grid - time_min_grid) * 0.05
    dist_margin = (dist_max_grid - dist_min_grid) * 0.05
    ax.set_xlim(dist_min_grid - dist_margin, dist_max_grid + dist_margin)
    ax.set_ylim(time_max_grid + time_margin, time_min_grid - time_margin)

    # グリッド境界線の描画（黒点線）
    # 時間方向の境界線（水平線）
    time_boundaries = set()
    for t_min, t_max in time_bins:
        time_boundaries.add(t_min)
        time_boundaries.add(t_max)

    for t_boundary in sorted(time_boundaries):
        ax.axhline(y=t_boundary, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

    # 距離方向の境界線（垂直線）
    dist_boundaries = set()
    for d_min, d_max in dist_bins:
        dist_boundaries.add(d_min)
        dist_boundaries.add(d_max)

    for d_boundary in sorted(dist_boundaries):
        ax.axvline(x=d_boundary, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

    # グリッドラベルの表示（岩石数 + RSFD Params）
    for i, (t_min, t_max) in enumerate(time_bins):
        for j, (d_min, d_max) in enumerate(dist_bins):
            # ラベル位置（グリッドの中央）
            label_x = (d_min + d_max) / 2
            label_y = (t_min + t_max) / 2

            lines = []
            
            # デフォルトの枠線色と太さ
            box_edge_color = 'black'
            box_linewidth = 1.0

            # 1. 岩石数の取得とラベルテキスト生成
            if rock_counts_dict and (i, j) in rock_counts_dict:
                counts = rock_counts_dict[(i, j)]
                g1 = counts.get('group1', 0)
                g2 = counts.get('group2', 0)
                g3 = counts.get('group3', 0)

                # スペース節約のため空白を詰める
                lines.append(f'Gr1:{g1} Gr2:{g2} Gr3:{g3}')

            # 2. RSFDパラメータの取得 (r, k, p)
            if fit_params_dict and (i, j) in fit_params_dict:
                params = fit_params_dict[(i, j)]
                r_val = params.get('r', 0)
                k_val = params.get('k', 0)
                p_str = params.get('p_str', '')
                p_val = params.get('p_value', 1.0) # 判定用数値p値

                # 2行目: rとk
                lines.append(f'r={r_val:.2f}, k={k_val:.1e}')
                # 3行目: p-value
                lines.append(f'{p_str}')
                
                # p-valueが0.05以下なら赤枠に変更し、線を少し太くする
                if p_val <= 0.05:
                    box_edge_color = 'red'
                    box_linewidth = 2.0

            if lines:
                label_text = '\n'.join(lines)
                text_color = 'black'
            else:
                label_text = 'No Data'
                text_color = 'gray'

            # ラベル描画（白背景付き）
            # 情報量が増えるのでフォントを少し小さく設定 (12 -> 10)
            ax.text(label_x, label_y, label_text,
                   fontsize=10, fontweight='bold', color=text_color,
                   horizontalalignment='center', verticalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=box_edge_color, linewidth=box_linewidth, alpha=0.8))

    # 軸ラベルの設定
    ax.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax.set_ylabel('Time [ns]', fontsize=font_medium)
    ax.tick_params(axis='both', which='major', labelsize=font_small)

    # 第2Y軸（深度）の追加
    ax2 = ax.twinx()
    t_min_plot, t_max_plot = ax.get_ylim()
    depth_min = (t_min_plot * 1e-9) * c / np.sqrt(epsilon_r) / 2
    depth_max = (t_max_plot * 1e-9) * c / np.sqrt(epsilon_r) / 2
    ax2.set_ylim(depth_min, depth_max)
    ax2.set_ylabel(r'Depth [m] ($\varepsilon_r = 4.5$)', fontsize=font_medium)
    ax2.tick_params(axis='y', which='major', labelsize=font_small)

    # レイアウト調整とカラーバー
    fig.subplots_adjust(bottom=0.18, right=0.9)
    cbar_ax = fig.add_axes([0.65, 0.05, 0.2, 0.05])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=font_small)

    plt.tight_layout(rect=[0, 0.1, 0.9, 1])

    # 保存
    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)
    plt.close()

    print(f'B-scanグリッドプロット保存: {output_path}.png')


# ------------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------------
if __name__ == '__main__':
    print('=== RSFD Grid Comparison Tool ===')
    print('時間・距離方向のグリッド分割によるRSFD比較ツール\n')

    # ------------------------------------------------------------------
    # 1. 入力ファイルチェック
    # ------------------------------------------------------------------
    print('B-scanデータファイル(.txt)のパスを入力してください:')
    bscan_path = input().strip()
    if not (os.path.exists(bscan_path) and bscan_path.lower().endswith('.txt')):
        raise FileNotFoundError('正しい .txt ファイルを指定してください。')

    bscan_dir = os.path.dirname(bscan_path)

    # ------------------------------------------------------------------
    # 2. label.jsonファイルの自動検索と選択
    # ------------------------------------------------------------------
    print('\nlabel.jsonファイルを検索中...')
    label_files = find_label_json_files(bscan_dir)
    data_path = select_label_file(label_files)

    # ------------------------------------------------------------------
    # 3. グリッド分割パラメータの入力
    # ------------------------------------------------------------------
    print('\n=== グリッド分割パラメータ ===')
    print('グリッド分割の入力方法を選択してください:')
    print('1: 分割幅を入力 (時間[ns]、距離[m])')
    print('2: 分割数を入力 (時間方向の分割数、距離方向の分割数)')
    grid_input_mode = input('選択 (1 or 2): ').strip()

    if grid_input_mode not in ['1', '2']:
        raise ValueError('1 または 2 を入力してください。')

    if grid_input_mode == '1':
        # モード1: 分割幅を入力
        time_bin_width = float(input('時間方向の分割幅 [ns] を入力してください: ').strip())
        dist_bin_width = float(input('距離方向の分割幅 [m] を入力してください: ').strip())

        if time_bin_width <= 0 or dist_bin_width <= 0:
            raise ValueError('分割幅は正の値を指定してください。')

        # 分割数は後で計算される
        num_time_bins_input = None
        num_dist_bins_input = None
    else:
        # モード2: 分割数を入力
        num_time_bins_input = int(input('時間方向の分割数を入力してください: ').strip())
        num_dist_bins_input = int(input('距離方向の分割数を入力してください: ').strip())

        if num_time_bins_input <= 0 or num_dist_bins_input <= 0:
            raise ValueError('分割数は正の整数を指定してください。')

        # 分割幅は後で計算される
        time_bin_width = None
        dist_bin_width = None

    # ------------------------------------------------------------------
    # 4. JSONデータ読み込み
    # ------------------------------------------------------------------
    print('\nデータ読み込み中...')
    with open(data_path, 'r') as f:
        results = json.load(f).get('results', {})

    # B-scanデータの読み込み
    print('B-scanデータ読み込み中...')
    bscan_data = np.loadtxt(bscan_path, delimiter=' ')
    print(f'B-scan形状: {bscan_data.shape}')

    x_all = np.array([v['x'] for v in results.values()])
    y_all = np.array([v['y'] for v in results.values()])
    lab_all = np.array([v['label'] for v in results.values()], dtype=int)
    time_top_all = np.array([none_to_nan(v['time_top']) for v in results.values()], dtype=float)
    time_bottom_all = np.array([none_to_nan(v['time_bottom']) for v in results.values()], dtype=float)

    print(f'ラベルデータ読み込み完了: {len(lab_all)}個')

    # ------------------------------------------------------------------
    # 5. グリッド範囲の計算
    # ------------------------------------------------------------------
    # 物理定数
    sample_interval = 0.312500e-9  # [s] - サンプル間隔
    trace_interval = 3.6e-2        # [m] - トレース間隔

    # 時間方向：最小値はB-scanから、最大値はlabel.jsonから取得
    # B-scanから時間の最小値を取得（地形補正対応、t<0も含む）
    print('B-scanから時間範囲を計算中...')

    # time_zero_idx を検出（最初のトレースの最初の非NaN値）
    first_trace = bscan_data[:, 0]
    first_non_nan_idx = np.where(~np.isnan(first_trace))[0]
    time_zero_idx = first_non_nan_idx[0] if len(first_non_nan_idx) > 0 else 0

    # 各トレースの最初の非NaN値の時間を計算（t=0未満も含む）
    first_valid_times = []
    for col in range(bscan_data.shape[1]):
        non_nan_idx = np.where(~np.isnan(bscan_data[:, col]))[0]
        if len(non_nan_idx) > 0:
            # time_zero_idxを基準とした相対時間（負の値も含む）
            time_ns = (non_nan_idx[0] - time_zero_idx) * sample_interval * 1e9
            first_valid_times.append(time_ns)

    if first_valid_times:
        time_min_data = min(first_valid_times)  # 負の値も含めて最小値を採用
    else:
        time_min_data = 0.0
        print('警告: B-scanに有効なデータが見つかりませんでした。time_min=0として処理を続行します。')

    # 最大値はlabel.jsonから取得
    # Group1はy座標、Group2-3はtime_topを使用
    time_values_group1 = y_all[lab_all == 1]
    time_values_others = time_top_all[(lab_all != 1) & (~np.isnan(time_top_all))]
    time_values_all = np.concatenate([time_values_group1, time_values_others])
    time_max_data = np.max(time_values_all)

    # 距離方向：B-scanから取得
    dist_min_data = 0.0  # B-scanの開始位置
    dist_max_data = bscan_data.shape[1] * trace_interval  # [m]

    # 入力モードに応じて分割数または分割幅を計算
    if grid_input_mode == '1':
        # モード1: 分割幅から分割数を計算
        num_time_bins = int(np.ceil((time_max_data - time_min_data) / time_bin_width))
        num_dist_bins = int(np.ceil((dist_max_data - dist_min_data) / dist_bin_width))
    else:
        # モード2: 分割数から分割幅を計算
        num_time_bins = num_time_bins_input
        num_dist_bins = num_dist_bins_input
        time_bin_width = (time_max_data - time_min_data) / num_time_bins
        dist_bin_width = (dist_max_data - dist_min_data) / num_dist_bins

    print(f'\n時間範囲: {time_min_data:.2f} - {time_max_data:.2f} ns')
    print(f'距離範囲: {dist_min_data:.2f} - {dist_max_data:.2f} m')
    print(f'時間方向分割数: {num_time_bins}')
    print(f'距離方向分割数: {num_dist_bins}')
    print(f'時間方向分割幅: {time_bin_width:.2f} ns')
    print(f'距離方向分割幅: {dist_bin_width:.2f} m')
    print(f'総グリッド数: {num_time_bins * num_dist_bins}')

    # ------------------------------------------------------------------
    # 6. 出力ディレクトリの作成
    # ------------------------------------------------------------------

    # 親ディレクトリ: RSFD_grid_comparison（B-scanファイルと同じディレクトリに作成）
    parent_dir = os.path.join(bscan_dir, 'RSFD_grid_comparison')
    os.makedirs(parent_dir, exist_ok=True)

    # サブディレクトリ名の決定（入力モードに応じて命名を変更）
    if grid_input_mode == '1':
        # モード1（分割幅入力）: timeXXns_distOOm
        sub_dir_name = f'time{time_bin_width:.0f}ns_dist{dist_bin_width:.0f}m'
    else:
        # モード2（分割数入力）: timeX_distY
        sub_dir_name = f'time{num_time_bins}_dist{num_dist_bins}'

    base_dir = os.path.join(parent_dir, sub_dir_name)
    os.makedirs(base_dir, exist_ok=True)

    individual_dir = os.path.join(base_dir, 'individual_plots')
    # comparison_dir = os.path.join(base_dir, 'comparison_plots')
    os.makedirs(individual_dir, exist_ok=True)
    os.makedirs(parent_dir, exist_ok=True)

    print(f'\n出力ディレクトリ: {base_dir}')

    # ------------------------------------------------------------------
    # 7. 各グリッドの処理
    # ------------------------------------------------------------------
    print('\n=== グリッド処理開始 ===')

    # 物理定数
    epsilon_regolith = 4.5  # 月面レゴリスの比誘電率
    epsilon_rock = 9.0     # 岩石の比誘電率
    c = 299_792_458  # [m/s]

    # 各グリッドのデータを格納するリスト
    grid_statistics = []

    # B-scanプロット用のグリッド範囲情報
    time_bins_for_bscan = []  # [(time_min, time_max), ...]
    dist_bins_for_bscan = []  # [(dist_min, dist_max), ...]

    # subplot用のグリッドデータ辞書（キー: (time_idx, dist_idx)）
    grid_data_dict_area_normalized = {}

    # subplot用のフィッティングパラメータ辞書（キー: (time_idx, dist_idx)）
    fit_params_dict_area_normalized = {}

    # 岩石数カウント辞書（キー: (time_idx, dist_idx)）
    rock_counts_dict = {}

    # B-scanプロット用のグリッド範囲情報を事前計算
    for i in range(num_time_bins):
        t_min = time_min_data + i * time_bin_width
        t_max = time_min_data + (i + 1) * time_bin_width
        if i == num_time_bins - 1:
            t_max = time_max_data
        time_bins_for_bscan.append((t_min, t_max))

    for j in range(num_dist_bins):
        d_min = dist_min_data + j * dist_bin_width
        d_max = dist_min_data + (j + 1) * dist_bin_width
        if j == num_dist_bins - 1:
            d_max = dist_max_data
        dist_bins_for_bscan.append((d_min, d_max))

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

            # フィッティング（面積規格化）
            (k_pow_norm, r_pow_norm, R2_pow_norm, N_pow_fit_norm, t_pow_norm, p_pow_norm,
             se_pow_norm, n_pow_norm, dof_pow_norm), D_fit_norm, cum_counts_normalized = \
                calc_fitting_power_law_area_normalized(unique_sizes, cum_counts, area)

            # グリッドラベル
            grid_label = f'T{i+1}D{j+1} ({time_min:.0f}-{time_max:.0f}ns, {dist_min:.0f}-{dist_max:.0f}m)'

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
                'p_str': format_p_value(p_pow_norm),
                'p_value': p_pow_norm  # p値（数値）を追加
            }

            # 岩石数カウントを保存
            rock_counts_dict[(i, j)] = {
                'total': num_rocks,
                'group1': num_group1,
                'group2': num_group2,
                'group3': num_group3
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
                'k_pow_norm': k_pow_norm,
                'r_pow_norm': r_pow_norm,
                'R2_pow_norm': R2_pow_norm,
                'p_value_pow_norm': format_p_value(p_pow_norm)
            })

            # 個別プロットの作成（面積規格化のみ）
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
    # 8. グリッドsubplot比較プロットの作成
    # ------------------------------------------------------------------
    if len(grid_data_dict_area_normalized) > 0:
        print('\n=== グリッドsubplot比較プロット作成 ===')

        # 面積規格化 log-log（フィッティング式表示版）
        output_path_subplot = os.path.join(base_dir, 'grid_subplot_loglog_area_normalized')
        create_grid_subplot_comparison(
            grid_data_dict_area_normalized, fit_params_dict_area_normalized,
            num_time_bins, num_dist_bins,
            'Rock Size D [cm]', 'Cumulative Number Density N [/m²]',
            output_path_subplot,
            scale_type='loglog'
        )

        # 面積規格化 log-log（岩石数表示版）
        output_path_subplot_rocks = os.path.join(base_dir, 'grid_subplot_loglog_area_normalized_rock_counts')
        create_grid_subplot_rock_counts(
            grid_data_dict_area_normalized, rock_counts_dict,
            num_time_bins, num_dist_bins,
            'Rock Size D [cm]', 'Cumulative Number Density N [/m²]',
            output_path_subplot_rocks,
            scale_type='loglog'
        )

        # ------------------------------------------------------------------
        # B-scanプロットにグリッド境界線を表示
        # ------------------------------------------------------------------
        print('\n=== B-scanグリッドプロット作成 ===')

        output_path_bscan = os.path.join(base_dir, 'bscan_with_grid')
        create_bscan_with_grid_lines(
            bscan_data, time_bins_for_bscan, dist_bins_for_bscan,
            output_path_bscan,
            fit_params_dict=fit_params_dict_area_normalized,
            rock_counts_dict=rock_counts_dict,
            sample_interval=sample_interval,
            trace_interval=trace_interval,
            epsilon_r=epsilon_regolith,
            c=c
        )
    else:
        print('\n警告: 有効なグリッドデータが見つかりませんでした。')

    # ------------------------------------------------------------------
    # 9. 統計情報の保存
    # ------------------------------------------------------------------
    if len(grid_statistics) > 0:
        stats_path = os.path.join(base_dir, 'grid_statistics.txt')
        save_statistics_to_txt(stats_path, grid_statistics)

    print('\n=== 処理完了 ===')
    print(f'出力ディレクトリ: {base_dir}')