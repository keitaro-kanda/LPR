#!/usr/bin/env python3
# make_RSFD_moving_window.py
# ------------------------------------------------------------
# 移動ウィンドウによるRSFD解析ツール
# 水平方向または深さ方向に移動しながらRSFDパラメータ(r, k)を計算し、
# B-scan背景上にプロットする
# ------------------------------------------------------------

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm

# ------------------------------------------------------------------
# 補助関数定義
# ------------------------------------------------------------------
def none_to_nan(v):
    """None値をnp.nanに変換"""
    return np.nan if v is None else v

def calc_fitting_power_law(sizes, counts):
    """
    べき則フィッティングを実行

    Returns:
    --------
    tuple: (k, r, R2, p_value) または データ不足時は (np.nan, np.nan, np.nan, np.nan)
    """
    if len(sizes) < 3 or len(counts) < 3:
        return np.nan, np.nan, np.nan, np.nan

    # 対数変換
    mask = (sizes > 0) & (counts > 0)
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan, np.nan

    log_D = np.log(sizes[mask])
    log_N = np.log(counts[mask])

    try:
        # べき則フィッティング (Power-law: log N = r log D + log k)
        X_pow = sm.add_constant(log_D)
        model_pow = sm.OLS(log_N, X_pow)
        results_pow = model_pow.fit()

        log_k_pow, r_pow = results_pow.params
        k_pow = np.exp(log_k_pow)
        R2_pow = results_pow.rsquared
        p_pow = results_pow.pvalues[1]

        return k_pow, np.abs(r_pow), R2_pow, p_pow
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

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

def load_rock_data(label_path):
    """
    label.jsonから岩石データを読み込む

    Parameters:
    -----------
    label_path : str
        label.jsonファイルのパス

    Returns:
    --------
    dict: 岩石データ（x, y, label, time_top, time_bottom）
    """
    with open(label_path, 'r') as f:
        results = json.load(f).get('results', {})

    x_all = np.array([v['x'] for v in results.values()])
    y_all = np.array([v['y'] for v in results.values()])
    lab_all = np.array([v['label'] for v in results.values()], dtype=int)
    time_top_all = np.array([none_to_nan(v['time_top']) for v in results.values()], dtype=float)
    time_bottom_all = np.array([none_to_nan(v['time_bottom']) for v in results.values()], dtype=float)

    return {
        'x': x_all,
        'y': y_all,
        'label': lab_all,
        'time_top': time_top_all,
        'time_bottom': time_bottom_all
    }

def calculate_rock_sizes(rock_data, epsilon_rock=9.0, c=299792458):
    """
    各岩石のサイズを計算

    Parameters:
    -----------
    rock_data : dict
        岩石データ
    epsilon_rock : float
        岩石の比誘電率
    c : float
        光速 [m/s]

    Returns:
    --------
    ndarray: 各岩石のサイズ [cm]
    """
    lab = rock_data['label']
    time_top = rock_data['time_top']
    time_bottom = rock_data['time_bottom']

    sizes = np.full(len(lab), np.nan)

    # Group1: 1cm固定
    sizes[lab == 1] = 1.0

    # Group2, Group3: time_top/time_bottomから計算
    for group in [2, 3]:
        mask = (lab == group) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
        if np.any(mask):
            size_calc = (time_bottom[mask] - time_top[mask]) * 1e-9 * c / np.sqrt(epsilon_rock) * 0.5 * 100
            sizes[mask] = np.round(size_calc, decimals=3)

    return sizes

def get_time_position(rock_data):
    """
    各岩石の時間位置を取得（Group1はy座標、Group2-3はtime_top）

    Parameters:
    -----------
    rock_data : dict
        岩石データ

    Returns:
    --------
    ndarray: 各岩石の時間位置 [ns]
    """
    lab = rock_data['label']
    y = rock_data['y']
    time_top = rock_data['time_top']

    time_pos = np.full(len(lab), np.nan)

    # Group1: y座標を使用
    time_pos[lab == 1] = y[lab == 1]

    # Group2, Group3: time_topを使用
    for group in [2, 3]:
        mask = lab == group
        time_pos[mask] = time_top[mask]

    return time_pos

def calculate_rsfd_in_window(rock_data, sizes, time_positions,
                              dist_min, dist_max, time_min, time_max,
                              epsilon_r, c):
    """
    指定された範囲内の岩石から面積規格化されたRSFDパラメータを計算

    Parameters:
    -----------
    rock_data : dict
        岩石データ
    sizes : ndarray
        各岩石のサイズ [cm]
    time_positions : ndarray
        各岩石の時間位置 [ns]
    dist_min, dist_max : float
        距離範囲 [m]
    time_min, time_max : float
        時間範囲 [ns]
    epsilon_r : float
        比誘電率
    c : float
        光速 [m/s]

    Returns:
    --------
    tuple: (k, r, R2, p_value, num_rocks, num_rocks_by_label, area, unique_sizes, cum_counts_normalized)
           num_rocks_by_label: dict with keys 1, 2, 3 containing counts per label
    """
    x = rock_data['x']
    lab = rock_data['label']

    # 範囲内の岩石をフィルタリング
    mask = (x >= dist_min) & (x <= dist_max) & \
           (time_positions >= time_min) & (time_positions <= time_max) & \
           (~np.isnan(sizes))

    filtered_sizes = sizes[mask]
    filtered_labels = lab[mask]
    num_rocks = len(filtered_sizes)

    # ラベル別岩石数をカウント
    num_rocks_by_label = {
        1: np.sum(filtered_labels == 1),
        2: np.sum(filtered_labels == 2),
        3: np.sum(filtered_labels == 3)
    }

    # 面積計算（時間範囲を深度に変換）
    depth_min = time_min * 1e-9 * c / (2 * np.sqrt(epsilon_r))
    depth_max = time_max * 1e-9 * c / (2 * np.sqrt(epsilon_r))
    area = (depth_max - depth_min) * (dist_max - dist_min)  # [m²]

    if num_rocks < 3:
        return np.nan, np.nan, np.nan, np.nan, num_rocks, num_rocks_by_label, area, np.array([]), np.array([])

    # 累積サイズ分布を計算
    unique_sizes = np.sort(np.unique(filtered_sizes))
    cum_counts = np.array([np.sum(filtered_sizes >= s) for s in unique_sizes])

    # 面積で規格化
    cum_counts_normalized = cum_counts / area

    # べき則フィッティング（規格化されたカウントを使用）
    k, r, R2, p_value = calc_fitting_power_law(unique_sizes, cum_counts_normalized)

    return k, r, R2, p_value, num_rocks, num_rocks_by_label, area, unique_sizes, cum_counts_normalized

def plot_individual_rsfd(unique_sizes, cum_counts_normalized, k, r, R2, p_value,
                         num_rocks, area, window_range, output_path,
                         dpi_png=300, dpi_pdf=600, xlim=None, ylim=None):
    """
    個別ウィンドウのRSFDプロットを作成

    Parameters:
    -----------
    unique_sizes : ndarray
        ユニークなサイズ配列 [cm]
    cum_counts_normalized : ndarray
        面積規格化された累積カウント [/m²]
    k, r, R2, p_value : float
        フィッティングパラメータ
    num_rocks : int
        岩石数
    area : float
        面積 [m²]
    window_range : str
        ウィンドウ範囲の説明文字列
    output_path : str
        出力パス（拡張子なし）
    """
    # Font size standards
    font_medium = 18
    font_small = 16

    fig, ax = plt.subplots(figsize=(10, 8))

    # データプロット
    if len(unique_sizes) > 0:
        ax.scatter(unique_sizes, cum_counts_normalized,
                  s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                  label='Observed data')

        # べき則フィッティング曲線
        if not np.isnan(k) and not np.isnan(r):
            if xlim is not None:
                size_fit = np.logspace(np.log10(xlim[0]),
                                       np.log10(xlim[1]), 100)
            else:
                size_fit = np.logspace(np.log10(unique_sizes.min()),
                                       np.log10(unique_sizes.max()), 100)
            count_fit = k * size_fit ** (-r)
            ax.plot(size_fit, count_fit, 'r-', linewidth=2,
                   label=f'Power-law fit: N = {k:.4e} × D^(-{r:.3f})')

    ax.set_title(f'Window: {window_range}', fontsize=font_medium)
    ax.set_xlabel('Rock size D [cm]', fontsize=font_medium)
    ax.set_ylabel('Cumulative count N(≥D) [/m²]', fontsize=font_medium)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=font_small)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=font_small)

    # パラメータ情報をテキストボックスで表示
    info_text = f'Window: {window_range}\n'
    info_text += f'k = {k:.4e} [/m²]\n'
    info_text += f'r = {r:.4f}\n'
    info_text += f'R² = {R2:.4f}\n'
    info_text += f'p-value = {p_value:.4e}\n'
    info_text += f'Num rocks = {num_rocks}\n'
    info_text += f'Area = {area:.4f} m²'

    ax.text(0.05, 0.05, info_text, transform=ax.transAxes,
           fontsize=font_small, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # 保存
    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)
    plt.close()

def create_horizontal_moving_window_plot(bscan_data, rock_data, sizes, time_positions,
                                          window_width, step_size,
                                          time_min_data, time_max_data,
                                          dist_min_data, dist_max_data,
                                          sample_interval, trace_interval,
                                          epsilon_r, c, output_path,
                                          dpi_png=300, dpi_pdf=600):
    """
    水平方向移動ウィンドウのRSFDプロットを作成

    Parameters:
    -----------
    bscan_data : ndarray
        B-scanデータ
    rock_data : dict
        岩石データ
    sizes : ndarray
        岩石サイズ [cm]
    time_positions : ndarray
        時間位置 [ns]
    window_width : float
        ウィンドウ幅 [m]
    step_size : float
        ステップサイズ [m]
    time_min_data, time_max_data : float
        時間範囲 [ns]
    dist_min_data, dist_max_data : float
        距離範囲 [m]
    sample_interval : float
        サンプル間隔 [s]
    trace_interval : float
        トレース間隔 [m]
    epsilon_r : float
        比誘電率
    c : float
        光速 [m/s]
    output_path : str
        出力パス（拡張子なし）
    """
    # Font size standards
    font_large = 20
    font_medium = 18
    font_small = 16

    # ウィンドウ中心位置を計算
    window_centers = []
    current_center = dist_min_data + window_width / 2
    while current_center + window_width / 2 <= dist_max_data:
        window_centers.append(current_center)
        current_center += step_size

    # RSFD個別プロット用サブディレクトリ作成
    rsfd_plots_dir = os.path.join(os.path.dirname(output_path), 'RSFD_plots')
    os.makedirs(rsfd_plots_dir, exist_ok=True)

    # 各ウィンドウでRSFDパラメータを計算
    k_values = []
    r_values = []
    R2_values = []
    p_values = []
    num_rocks_list = []
    num_rocks_label1_list = []
    num_rocks_label2_list = []
    num_rocks_label3_list = []
    area_list = []

    print(f'\n水平方向移動ウィンドウ解析: {len(window_centers)}個のウィンドウ')

    # 個別RSFDプロット用データ格納リスト
    rsfd_plot_data = []

    for center in tqdm(window_centers, desc='RSFD計算中'):
        d_min = center - window_width / 2
        d_max = center + window_width / 2

        k, r, R2, p, num_rocks, num_rocks_by_label, area, unique_sizes, cum_counts_normalized = calculate_rsfd_in_window(
            rock_data, sizes, time_positions,
            d_min, d_max, time_min_data, time_max_data,
            epsilon_r, c
        )

        k_values.append(k)
        r_values.append(r)
        R2_values.append(R2)
        p_values.append(p)
        num_rocks_list.append(num_rocks)
        num_rocks_label1_list.append(num_rocks_by_label[1])
        num_rocks_label2_list.append(num_rocks_by_label[2])
        num_rocks_label3_list.append(num_rocks_by_label[3])
        area_list.append(area)

        # 個別RSFDプロット用データを格納
        window_range = f'{d_min:.2f} - {d_max:.2f} m'
        rsfd_output_path = os.path.join(rsfd_plots_dir, f'rsfd_x{d_min:.2f}-{d_max:.2f}m')
        rsfd_plot_data.append({
            'unique_sizes': unique_sizes,
            'cum_counts_normalized': cum_counts_normalized,
            'k': k, 'r': r, 'R2': R2, 'p': p,
            'num_rocks': num_rocks, 'area': area,
            'window_range': window_range,
            'output_path': rsfd_output_path
        })

    # 全ウィンドウのデータから共通軸範囲を算出
    all_sizes_list = [d['unique_sizes'] for d in rsfd_plot_data if len(d['unique_sizes']) > 0]
    all_counts_list = [d['cum_counts_normalized'] for d in rsfd_plot_data if len(d['cum_counts_normalized']) > 0]
    if all_sizes_list and all_counts_list:
        all_sizes_concat = np.concatenate(all_sizes_list)
        all_counts_concat = np.concatenate(all_counts_list)
        positive_counts = all_counts_concat[all_counts_concat > 0]
        if len(all_sizes_concat) > 0 and len(positive_counts) > 0:
            global_xlim = (all_sizes_concat.min() * 0.5, all_sizes_concat.max() * 2.0)
            global_ylim = (positive_counts.min() * 0.5, all_counts_concat.max() * 2.0)
        else:
            global_xlim = None
            global_ylim = None
    else:
        global_xlim = None
        global_ylim = None

    # 個別RSFDプロット作成（共通軸範囲で統一）
    for d in rsfd_plot_data:
        plot_individual_rsfd(d['unique_sizes'], d['cum_counts_normalized'],
                           d['k'], d['r'], d['R2'], d['p'],
                           d['num_rocks'], d['area'],
                           d['window_range'], d['output_path'],
                           xlim=global_xlim, ylim=global_ylim)

    window_centers = np.array(window_centers)
    k_values = np.array(k_values)
    r_values = np.array(r_values)

    # Time zero検出
    first_trace = bscan_data[:, 0]
    first_non_nan_idx = np.where(~np.isnan(first_trace))[0]
    time_zero_idx = first_non_nan_idx[0] if len(first_non_nan_idx) > 0 else 0

    # 時間配列の計算
    time_array = (np.arange(bscan_data.shape[0]) - time_zero_idx) * sample_interval * 1e9  # [ns]

    # B-scan表示範囲
    vmin = -np.nanmax(np.abs(bscan_data)) / 10
    vmax = np.nanmax(np.abs(bscan_data)) / 10

    # p値の配列化
    p_values_array = np.array(p_values)

    # 岩石数の配列化（ラベル別）
    num_rocks_array = np.array(num_rocks_list)
    num_label1_array = np.array(num_rocks_label1_list)
    num_label2_array = np.array(num_rocks_label2_list)
    num_label3_array = np.array(num_rocks_label3_list)

    # 棒グラフの幅（ステップサイズの80%）
    bar_width = step_size * 0.8

    # Figure作成（4つのサブプロット：上から岩石数、r、k、p値）
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(18, 20), sharex=True)

    # === 1段目: 岩石数のプロット（ラベル別積み上げ棒グラフ） ===
    # B-scan背景
    ax1.imshow(bscan_data, aspect='auto', cmap='seismic',
               extent=[0, bscan_data.shape[1] * trace_interval,
                      time_array[-1], time_array[0]],
               vmin=vmin, vmax=vmax, alpha=0.5)

    # 岩石数を第2軸に積み上げ棒グラフでプロット（縦向き）
    ax1_twin = ax1.twinx()
    # ラベル1（赤）を一番下に
    ax1_twin.bar(window_centers, num_label1_array, width=bar_width,
                 color='red', alpha=0.7, label='Group 1')
    # ラベル2（緑）をラベル1の上に積み上げ
    ax1_twin.bar(window_centers, num_label2_array, width=bar_width,
                 bottom=num_label1_array, color='green', alpha=0.7, label='Group 2')
    # ラベル3（青）をラベル1+2の上に積み上げ
    ax1_twin.bar(window_centers, num_label3_array, width=bar_width,
                 bottom=num_label1_array + num_label2_array, color='blue', alpha=0.7, label='Group 3')
    ax1_twin.set_ylabel('Number of detected rocks', fontsize=font_medium, color='black')
    ax1_twin.tick_params(axis='y', labelcolor='black', labelsize=font_small)
    ax1_twin.legend(loc='lower right', fontsize=font_small - 2)

    ax1.set_ylabel('Time [ns]', fontsize=font_medium)
    ax1.set_ylim(time_max_data, time_min_data)
    ax1.tick_params(axis='both', which='major', labelsize=font_small)
    ax1.grid(True, ls='--', alpha=1.0, axis='both')

    # === 2段目: rのプロット ===
    # B-scan背景
    ax2.imshow(bscan_data, aspect='auto', cmap='seismic',
               extent=[0, bscan_data.shape[1] * trace_interval,
                      time_array[-1], time_array[0]],
               vmin=vmin, vmax=vmax, alpha=0.5)

    # rの値を第2軸にプロット（NaN値は1.6としてプロット）
    ax2_twin = ax2.twinx()
    r_nan_value = 1.6  # NaN値の代替値
    r_values_plot = np.where(np.isnan(r_values), r_nan_value, r_values)
    ax2_twin.plot(window_centers, r_values_plot,
                  'b-', linewidth=2, marker='o', markersize=10, label='r (power-law exponent)')
    ax2_twin.set_ylabel('r (power-law exponent)', fontsize=font_medium, color='blue')
    ax2_twin.tick_params(axis='y', labelcolor='blue', labelsize=font_small)

    # y軸範囲設定（r値）: 0.3〜1.6に拡張（1.6はNaN用）
    ax2_twin.set_ylim(0.3, 1.6)
    # カスタム目盛り設定（1.6を「nan」と表示）
    r_ticks = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    r_tick_labels = ['0.4', '0.6', '0.8', '1.0', '1.2', '1.4', 'nan']
    ax2_twin.set_yticks(r_ticks)
    ax2_twin.set_yticklabels(r_tick_labels)

    ax2.set_ylabel('Time [ns]', fontsize=font_medium)
    ax2.set_ylim(time_max_data, time_min_data)
    ax2.tick_params(axis='both', which='major', labelsize=font_small)
    ax2.grid(True, ls='--', alpha=1.0, axis='both')

    # === 3段目: kのプロット ===
    # B-scan背景
    ax3.imshow(bscan_data, aspect='auto', cmap='seismic',
               extent=[0, bscan_data.shape[1] * trace_interval,
                      time_array[-1], time_array[0]],
               vmin=vmin, vmax=vmax, alpha=0.5)

    # kの値を第2軸にプロット（リニアスケール、NaN値は8e-3としてプロット）
    ax3_twin = ax3.twinx()
    k_nan_value = 8e-3  # NaN値の代替値
    k_values_plot = np.where(np.isnan(k_values), k_nan_value, k_values)
    ax3_twin.plot(window_centers, k_values_plot,
                  'r-', linewidth=2, marker='s', markersize=10, label='k (scaling factor)')
    ax3_twin.tick_params(axis='y', labelcolor='red', labelsize=font_small)
    # y軸範囲設定（k値）: 1e-3〜8e-3に拡張（8e-3はNaN用）
    ax3_twin.set_ylim(1e-3, 8e-3)
    # カスタム目盛り設定（8e-3を「nan」と表示）
    k_ticks = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3]
    k_tick_labels = ['1', '2', '3', '4', '5', '6', '7', 'nan']
    ax3_twin.set_yticks(k_ticks)
    ax3_twin.set_yticklabels(k_tick_labels)
    ax3_twin.set_ylabel('k [×10⁻³ /m²]', fontsize=font_medium, color='red')

    ax3.set_ylabel('Time [ns]', fontsize=font_medium)
    ax3.set_ylim(time_max_data, time_min_data)
    ax3.tick_params(axis='both', which='major', labelsize=font_small)
    ax3.grid(True, ls='--', alpha=1.0, axis='both')

    # === 4段目: p値のプロット ===
    # B-scan背景
    ax4.imshow(bscan_data, aspect='auto', cmap='seismic',
               extent=[0, bscan_data.shape[1] * trace_interval,
                      time_array[-1], time_array[0]],
               vmin=vmin, vmax=vmax, alpha=0.5)

    # p値を第2軸にプロット（対数スケール、NaN値は1.5としてプロット）
    ax4_twin = ax4.twinx()
    p_nan_value = 1.5  # NaN値の代替値
    p_values_plot = np.where(np.isnan(p_values_array) | (p_values_array <= 0), p_nan_value, p_values_array)
    ax4_twin.semilogy(window_centers, p_values_plot,
                      'g-', linewidth=2, marker='^', markersize=10, label='p-value')
    # p=0.05の横線を追加
    ax4_twin.axhline(y=0.05, color='black', linestyle='--', linewidth=1.5, label='p = 0.05')
    ax4_twin.set_ylabel('p-value', fontsize=font_medium, color='green')
    ax4_twin.tick_params(axis='y', labelcolor='green', labelsize=font_small)
    # y軸範囲設定（p値）: 1e-4〜1.5に拡張（1.5はNaN用）
    ax4_twin.set_ylim(1e-4, 1.5)
    # カスタム目盛り設定（1.5を「nan」と表示）
    p_ticks = [1e-4, 1e-3, 1e-2, 0.05, 1e-1, 1.0, 1.5]
    p_tick_labels = ['10⁻⁴', '10⁻³', '10⁻²', '0.05', '10⁻¹', '1', 'nan']
    ax4_twin.set_yticks(p_ticks)
    ax4_twin.set_yticklabels(p_tick_labels)

    ax4.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax4.set_ylabel('Time [ns]', fontsize=font_medium)
    ax4.set_ylim(time_max_data, time_min_data)
    ax4.tick_params(axis='both', which='major', labelsize=font_small)
    ax4.grid(True, ls='--', alpha=1.0, axis='both')

    plt.tight_layout()

    # 保存
    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)
    plt.close()

    print(f'水平方向移動ウィンドウプロット保存: {output_path}.png')

    # === 個別プロット作成 ===
    # 個別プロット用サブディレクトリ
    individual_plots_dir = os.path.join(os.path.dirname(output_path), 'individual_plots')
    os.makedirs(individual_plots_dir, exist_ok=True)

    # --- 岩石数の個別プロット（ラベル別積み上げ棒グラフ） ---
    fig_num, ax_num = plt.subplots(figsize=(18, 6))
    ax_num.imshow(bscan_data, aspect='auto', cmap='seismic',
                  extent=[0, bscan_data.shape[1] * trace_interval,
                         time_array[-1], time_array[0]],
                  vmin=vmin, vmax=vmax, alpha=0.5)
    ax_num_twin = ax_num.twinx()
    # ラベル別積み上げ棒グラフ（縦向き）
    ax_num_twin.bar(window_centers, num_label1_array, width=bar_width,
                    color='red', alpha=0.7, label='Group 1')
    ax_num_twin.bar(window_centers, num_label2_array, width=bar_width,
                    bottom=num_label1_array, color='green', alpha=0.7, label='Group 2')
    ax_num_twin.bar(window_centers, num_label3_array, width=bar_width,
                    bottom=num_label1_array + num_label2_array, color='blue', alpha=0.7, label='Group 3')
    ax_num_twin.set_ylabel('Number of detected rocks', fontsize=font_medium, color='black')
    ax_num_twin.tick_params(axis='y', labelcolor='black', labelsize=font_small)
    ax_num_twin.legend(loc='lower right', fontsize=font_small - 2)
    ax_num.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax_num.set_ylabel('Time [ns]', fontsize=font_medium)
    ax_num.set_ylim(time_max_data, time_min_data)
    ax_num.tick_params(axis='both', which='major', labelsize=font_small)
    ax_num.grid(True, ls='--', alpha=1.0, axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(individual_plots_dir, 'num_rocks_plot.png'), dpi=dpi_png)
    plt.savefig(os.path.join(individual_plots_dir, 'num_rocks_plot.pdf'), dpi=dpi_pdf)
    plt.close()

    # --- rの個別プロット ---
    fig_r, ax_r = plt.subplots(figsize=(18, 6))
    ax_r.imshow(bscan_data, aspect='auto', cmap='seismic',
                extent=[0, bscan_data.shape[1] * trace_interval,
                       time_array[-1], time_array[0]],
                vmin=vmin, vmax=vmax, alpha=0.5)
    ax_r_twin = ax_r.twinx()
    ax_r_twin.plot(window_centers, r_values_plot,
                   'b-', linewidth=2, marker='o', markersize=10, label='r (power-law exponent)')
    ax_r_twin.set_ylabel('r (power-law exponent)', fontsize=font_medium, color='blue')
    ax_r_twin.tick_params(axis='y', labelcolor='blue', labelsize=font_small)
    ax_r_twin.set_ylim(0.3, 1.6)
    ax_r_twin.set_yticks(r_ticks)
    ax_r_twin.set_yticklabels(r_tick_labels)
    ax_r.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax_r.set_ylabel('Time [ns]', fontsize=font_medium)
    ax_r.set_ylim(time_max_data, time_min_data)
    ax_r.tick_params(axis='both', which='major', labelsize=font_small)
    ax_r.grid(True, ls='--', alpha=1.0, axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(individual_plots_dir, 'r_plot.png'), dpi=dpi_png)
    plt.savefig(os.path.join(individual_plots_dir, 'r_plot.pdf'), dpi=dpi_pdf)
    plt.close()

    # --- kの個別プロット ---
    fig_k, ax_k = plt.subplots(figsize=(18, 6))
    ax_k.imshow(bscan_data, aspect='auto', cmap='seismic',
                extent=[0, bscan_data.shape[1] * trace_interval,
                       time_array[-1], time_array[0]],
                vmin=vmin, vmax=vmax, alpha=0.5)
    ax_k_twin = ax_k.twinx()
    ax_k_twin.plot(window_centers, k_values_plot,
                   'r-', linewidth=2, marker='s', markersize=10, label='k (scaling factor)')
    ax_k_twin.tick_params(axis='y', labelcolor='red', labelsize=font_small)
    ax_k_twin.set_ylim(1e-3, 8e-3)
    ax_k_twin.set_yticks(k_ticks)
    ax_k_twin.set_yticklabels(k_tick_labels)
    ax_k_twin.set_ylabel('k [×10⁻³ /m²]', fontsize=font_medium, color='red')
    ax_k.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax_k.set_ylabel('Time [ns]', fontsize=font_medium)
    ax_k.set_ylim(time_max_data, time_min_data)
    ax_k.tick_params(axis='both', which='major', labelsize=font_small)
    ax_k.grid(True, ls='--', alpha=1.0, axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(individual_plots_dir, 'k_plot.png'), dpi=dpi_png)
    plt.savefig(os.path.join(individual_plots_dir, 'k_plot.pdf'), dpi=dpi_pdf)
    plt.close()

    # --- p値の個別プロット ---
    fig_p, ax_p = plt.subplots(figsize=(18, 6))
    ax_p.imshow(bscan_data, aspect='auto', cmap='seismic',
                extent=[0, bscan_data.shape[1] * trace_interval,
                       time_array[-1], time_array[0]],
                vmin=vmin, vmax=vmax, alpha=0.5)
    ax_p_twin = ax_p.twinx()
    ax_p_twin.semilogy(window_centers, p_values_plot,
                       'g-', linewidth=2, marker='^', markersize=10, label='p-value')
    ax_p_twin.axhline(y=0.05, color='black', linestyle='--', linewidth=1.5, label='p = 0.05')
    ax_p_twin.set_ylabel('p-value', fontsize=font_medium, color='green')
    ax_p_twin.tick_params(axis='y', labelcolor='green', labelsize=font_small)
    ax_p_twin.set_ylim(1e-4, 1.5)
    ax_p_twin.set_yticks(p_ticks)
    ax_p_twin.set_yticklabels(p_tick_labels)
    ax_p.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax_p.set_ylabel('Time [ns]', fontsize=font_medium)
    ax_p.set_ylim(time_max_data, time_min_data)
    ax_p.tick_params(axis='both', which='major', labelsize=font_small)
    ax_p.grid(True, ls='--', alpha=1.0, axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(individual_plots_dir, 'p_value_plot.png'), dpi=dpi_png)
    plt.savefig(os.path.join(individual_plots_dir, 'p_value_plot.pdf'), dpi=dpi_pdf)
    plt.close()

    print(f'個別プロット保存: {individual_plots_dir}')

    # p値統計を計算（p値がnanの場合はp > 0.05とみなす）
    p_values_array = np.array(p_values)
    valid_p_count = 0
    significant_count = 0
    for p in p_values_array:
        if np.isnan(p):
            valid_p_count += 1  # nanはp > 0.05とみなす
        else:
            valid_p_count += 1
            if p <= 0.05:
                significant_count += 1

    significant_ratio = (significant_count / valid_p_count * 100) if valid_p_count > 0 else 0.0

    # 統計情報をテキストファイルに保存
    stats_path = f'{output_path}_statistics.txt'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('# Horizontal Moving Window RSFD Statistics (Area-Normalized)\n')
        f.write('# ==========================================================\n\n')
        f.write(f'Window width: {window_width:.2f} m\n')
        f.write(f'Step size: {step_size:.2f} m\n')
        f.write(f'Number of windows: {len(window_centers)}\n')
        f.write(f'Note: k values are area-normalized [/m²]\n\n')
        f.write('# P-value Statistics\n')
        f.write(f'Windows with p ≤ 0.05: {significant_count} / {valid_p_count} ({significant_ratio:.2f}%)\n')
        f.write(f'Note: NaN p-values are treated as p > 0.05\n\n')
        f.write('Window\tCenter [m]\tk [/m²]\tr\tR2\tp-value\tNum rocks\tArea [m²]\n')
        for i, center in enumerate(window_centers):
            f.write(f'{i+1}\t{center:.2f}\t{k_values[i]:.4e}\t{r_values[i]:.4f}\t'
                   f'{R2_values[i]:.4f}\t{p_values[i]:.4e}\t{num_rocks_list[i]}\t{area_list[i]:.4f}\n')

    print(f'統計情報保存: {stats_path}')
    print(f'p値統計: {significant_count}/{valid_p_count}個のウィンドウでp ≤ 0.05 ({significant_ratio:.2f}%)')

def create_vertical_moving_window_plot(bscan_data, rock_data, sizes, time_positions,
                                        window_width, step_size,
                                        time_min_data, time_max_data,
                                        dist_min_data, dist_max_data,
                                        sample_interval, trace_interval,
                                        epsilon_r, c, output_path,
                                        dpi_png=300, dpi_pdf=600):
    """
    深さ方向移動ウィンドウのRSFDプロットを作成

    Parameters:
    -----------
    window_width : float
        ウィンドウ幅 [ns]
    step_size : float
        ステップサイズ [ns]
    (その他のパラメータは水平方向と同じ)
    """
    # Font size standards
    font_large = 20
    font_medium = 18
    font_small = 16

    # RSFD個別プロット用サブディレクトリ作成
    rsfd_plots_dir = os.path.join(os.path.dirname(output_path), 'RSFD_plots')
    os.makedirs(rsfd_plots_dir, exist_ok=True)

    # ウィンドウ中心位置を計算（時間方向）
    window_centers = []
    current_center = time_min_data + window_width / 2
    while current_center + window_width / 2 <= time_max_data:
        window_centers.append(current_center)
        current_center += step_size

    # 各ウィンドウでRSFDパラメータを計算
    k_values = []
    r_values = []
    R2_values = []
    p_values = []
    num_rocks_list = []
    num_rocks_label1_list = []
    num_rocks_label2_list = []
    num_rocks_label3_list = []
    area_list = []

    print(f'\n深さ方向移動ウィンドウ解析: {len(window_centers)}個のウィンドウ')

    # 個別RSFDプロット用データ格納リスト
    rsfd_plot_data = []

    for center in tqdm(window_centers, desc='RSFD計算中'):
        t_min = center - window_width / 2
        t_max = center + window_width / 2

        k, r, R2, p, num_rocks, num_rocks_by_label, area, unique_sizes, cum_counts_normalized = calculate_rsfd_in_window(
            rock_data, sizes, time_positions,
            dist_min_data, dist_max_data, t_min, t_max,
            epsilon_r, c
        )

        k_values.append(k)
        r_values.append(r)
        R2_values.append(R2)
        p_values.append(p)
        num_rocks_list.append(num_rocks)
        num_rocks_label1_list.append(num_rocks_by_label[1])
        num_rocks_label2_list.append(num_rocks_by_label[2])
        num_rocks_label3_list.append(num_rocks_by_label[3])
        area_list.append(area)

        # 個別RSFDプロット用データを格納
        window_range = f'{t_min:.2f} - {t_max:.2f} ns'
        rsfd_output_path = os.path.join(rsfd_plots_dir, f'rsfd_t{t_min:.2f}-{t_max:.2f}ns')
        rsfd_plot_data.append({
            'unique_sizes': unique_sizes,
            'cum_counts_normalized': cum_counts_normalized,
            'k': k, 'r': r, 'R2': R2, 'p': p,
            'num_rocks': num_rocks, 'area': area,
            'window_range': window_range,
            'output_path': rsfd_output_path
        })

    # 全ウィンドウのデータから共通軸範囲を算出
    all_sizes_list = [d['unique_sizes'] for d in rsfd_plot_data if len(d['unique_sizes']) > 0]
    all_counts_list = [d['cum_counts_normalized'] for d in rsfd_plot_data if len(d['cum_counts_normalized']) > 0]
    if all_sizes_list and all_counts_list:
        all_sizes_concat = np.concatenate(all_sizes_list)
        all_counts_concat = np.concatenate(all_counts_list)
        positive_counts = all_counts_concat[all_counts_concat > 0]
        if len(all_sizes_concat) > 0 and len(positive_counts) > 0:
            global_xlim = (all_sizes_concat.min() * 0.5, all_sizes_concat.max() * 2.0)
            global_ylim = (positive_counts.min() * 0.5, all_counts_concat.max() * 2.0)
        else:
            global_xlim = None
            global_ylim = None
    else:
        global_xlim = None
        global_ylim = None

    # 個別RSFDプロット作成（共通軸範囲で統一）
    for d in rsfd_plot_data:
        plot_individual_rsfd(d['unique_sizes'], d['cum_counts_normalized'],
                           d['k'], d['r'], d['R2'], d['p'],
                           d['num_rocks'], d['area'],
                           d['window_range'], d['output_path'],
                           xlim=global_xlim, ylim=global_ylim)

    window_centers = np.array(window_centers)
    k_values = np.array(k_values)
    r_values = np.array(r_values)

    # Time zero検出
    first_trace = bscan_data[:, 0]
    first_non_nan_idx = np.where(~np.isnan(first_trace))[0]
    time_zero_idx = first_non_nan_idx[0] if len(first_non_nan_idx) > 0 else 0

    # 時間配列の計算
    time_array = (np.arange(bscan_data.shape[0]) - time_zero_idx) * sample_interval * 1e9  # [ns]

    # B-scan表示範囲
    vmin = -np.nanmax(np.abs(bscan_data)) / 10
    vmax = np.nanmax(np.abs(bscan_data)) / 10

    # p値の配列化
    p_values_array = np.array(p_values)

    # 岩石数の配列化（ラベル別）
    num_rocks_array = np.array(num_rocks_list)
    num_label1_array = np.array(num_rocks_label1_list)
    num_label2_array = np.array(num_rocks_label2_list)
    num_label3_array = np.array(num_rocks_label3_list)

    # 棒グラフの高さ（ステップサイズの80%）
    bar_height = step_size * 0.8

    # Figure作成（4つのサブプロット：上から岩石数、r、k、p値）
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(18, 20), sharey=True)

    # === 1段目: 岩石数のプロット（ラベル別積み上げ棒グラフ） ===
    # B-scan背景
    ax1.imshow(bscan_data, aspect='auto', cmap='seismic',
               extent=[0, bscan_data.shape[1] * trace_interval,
                      time_array[-1], time_array[0]],
               vmin=vmin, vmax=vmax, alpha=0.5)

    # 岩石数を第2横軸に積み上げ棒グラフでプロット（横向き）
    ax1_twin = ax1.twiny()
    # ラベル1（赤）を一番左に
    ax1_twin.barh(window_centers, num_label1_array, height=bar_height,
                  color='red', alpha=0.7, label='Group 1')
    # ラベル2（緑）をラベル1の右に積み上げ
    ax1_twin.barh(window_centers, num_label2_array, height=bar_height,
                  left=num_label1_array, color='green', alpha=0.7, label='Group 2')
    # ラベル3（青）をラベル1+2の右に積み上げ
    ax1_twin.barh(window_centers, num_label3_array, height=bar_height,
                  left=num_label1_array + num_label2_array, color='blue', alpha=0.7, label='Group 3')

    # フィッティングデータ数を第2横軸に折れ線グラフでプロット（横向き）
    num_fitting_points = num_label2_array + num_label3_array + 1 # グループ１は１つのデータ点にまとめて取り扱う
    ax1_twin.plot(num_fitting_points, window_centers,
                'k-', linewidth=2, marker='D', markersize=10, label='Number of fitting points')

    # 第２横軸の設定
    ax1_twin.set_xlabel('Number of rocks', fontsize=font_medium, color='black')
    ax1_twin.tick_params(axis='x', labelcolor='black', labelsize=font_small)
    ax1_twin.legend(loc='lower right', fontsize=font_small - 2)

    # グラフの軸ラベルと範囲設定
    ax1.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax1.set_ylabel('Time [ns]', fontsize=font_medium)
    ax1.set_ylim(time_max_data, time_min_data)
    ax1.tick_params(axis='both', which='major', labelsize=font_small)
    ax1.grid(True, ls='--', alpha=1.0, axis='both')

    # === 2段目: rのプロット ===
    # B-scan背景
    ax2.imshow(bscan_data, aspect='auto', cmap='seismic',
               extent=[0, bscan_data.shape[1] * trace_interval,
                      time_array[-1], time_array[0]],
               vmin=vmin, vmax=vmax, alpha=0.5)

    # rの値を第2横軸にプロット（縦方向に沿って、NaN値は1.6としてプロット）
    ax2_twin = ax2.twiny()
    r_nan_value = 1.6  # NaN値の代替値
    r_values_plot = np.where(np.isnan(r_values), r_nan_value, r_values)
    ax2_twin.plot(r_values_plot, window_centers,
                  'b-', linewidth=2, marker='o', markersize=10, label='r (power-law exponent)')
    ax2_twin.set_xlabel('r (power-law exponent)', fontsize=font_medium, color='blue')
    ax2_twin.tick_params(axis='x', labelcolor='blue', labelsize=font_small)

    # x軸範囲設定（r値）: 0.3〜1.6に拡張（1.6はNaN用）
    ax2_twin.set_xlim(0.3, 1.6)
    # カスタム目盛り設定（1.6を「nan」と表示）
    r_ticks = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    r_tick_labels = ['0.4', '0.6', '0.8', '1.0', '1.2', '1.4', 'nan']
    ax2_twin.set_xticks(r_ticks)
    ax2_twin.set_xticklabels(r_tick_labels)

    ax2.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax2.set_ylabel('Time [ns]', fontsize=font_medium)
    ax2.set_ylim(time_max_data, time_min_data)
    ax2.tick_params(axis='both', which='major', labelsize=font_small)
    ax2.grid(True, ls='--', alpha=1.0, axis='both')

    # === 3段目: kのプロット ===
    # B-scan背景
    ax3.imshow(bscan_data, aspect='auto', cmap='seismic',
               extent=[0, bscan_data.shape[1] * trace_interval,
                      time_array[-1], time_array[0]],
               vmin=vmin, vmax=vmax, alpha=0.5)

    # kの値を第2横軸にプロット（リニアスケール、縦方向に沿って、NaN値は22e-3としてプロット）
    ax3_twin = ax3.twiny()
    k_nan_value = 22e-3  # NaN値の代替値
    k_values_plot = np.where(np.isnan(k_values), k_nan_value, k_values)
    ax3_twin.plot(k_values_plot, window_centers,
                  'r-', linewidth=2, marker='s', markersize=10, label='k (scaling factor)')
    ax3_twin.tick_params(axis='x', labelcolor='red', labelsize=font_small)
    # x軸範囲設定（k値）: 0〜22e-3に拡張（22e-3はNaN用）
    ax3_twin.set_xlim(0, 22e-3)
    # カスタム目盛り設定（22e-3を「nan」と表示）
    k_ticks = [0, 5e-3, 10e-3, 15e-3, 20e-3, 22e-3]
    k_tick_labels = ['0', '5', '10', '15', '20', 'nan']
    ax3_twin.set_xticks(k_ticks)
    ax3_twin.set_xticklabels(k_tick_labels)
    ax3_twin.set_xlabel('k [×10⁻³ /m²]', fontsize=font_medium, color='red')

    ax3.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax3.set_ylabel('Time [ns]', fontsize=font_medium)
    ax3.set_ylim(time_max_data, time_min_data)
    ax3.tick_params(axis='both', which='major', labelsize=font_small)
    ax3.grid(True, ls='--', alpha=1.0, axis='both')

    # === 4段目: p値のプロット ===
    # B-scan背景
    ax4.imshow(bscan_data, aspect='auto', cmap='seismic',
               extent=[0, bscan_data.shape[1] * trace_interval,
                      time_array[-1], time_array[0]],
               vmin=vmin, vmax=vmax, alpha=0.5)

    # p値を第2横軸にプロット（対数スケール、縦方向に沿って、NaN値は1.5としてプロット）
    ax4_twin = ax4.twiny()
    p_nan_value = 1.5  # NaN値の代替値
    p_values_plot = np.where(np.isnan(p_values_array) | (p_values_array <= 0), p_nan_value, p_values_array)
    ax4_twin.semilogx(p_values_plot, window_centers,
                      'g-', linewidth=2, marker='^', markersize=10, label='p-value')
    # p=0.05の縦線を追加
    ax4_twin.axvline(x=0.05, color='black', linestyle='--', linewidth=1.5, label='p = 0.05')
    ax4_twin.set_xlabel('p-value', fontsize=font_medium, color='green')
    ax4_twin.tick_params(axis='x', labelcolor='green', labelsize=font_small)
    # x軸範囲設定（p値）: 1e-4〜1.5に拡張（1.5はNaN用）
    ax4_twin.set_xlim(1e-4, 1.5)
    # カスタム目盛り設定（1.5を「nan」と表示）
    p_ticks = [1e-4, 1e-3, 1e-2, 0.05, 1e-1, 1.0, 1.5]
    p_tick_labels = ['10⁻⁴', '10⁻³', '10⁻²', '0.05', '10⁻¹', '1', 'nan']
    ax4_twin.set_xticks(p_ticks)
    ax4_twin.set_xticklabels(p_tick_labels)

    ax4.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax4.set_ylabel('Time [ns]', fontsize=font_medium)
    ax4.set_ylim(time_max_data, time_min_data)
    ax4.tick_params(axis='both', which='major', labelsize=font_small)
    ax4.grid(True, ls='--', alpha=1.0, axis='both')

    plt.tight_layout()

    # 保存
    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)
    plt.close()

    print(f'深さ方向移動ウィンドウプロット保存: {output_path}.png')

    # === 個別プロット作成 ===
    # 個別プロット用サブディレクトリ
    individual_plots_dir = os.path.join(os.path.dirname(output_path), 'individual_plots')
    os.makedirs(individual_plots_dir, exist_ok=True)

    # --- 岩石数の個別プロット（ラベル別積み上げ棒グラフ） ---
    fig_num, ax_num = plt.subplots(figsize=(18, 6))
    ax_num.imshow(bscan_data, aspect='auto', cmap='seismic',
                  extent=[0, bscan_data.shape[1] * trace_interval,
                         time_array[-1], time_array[0]],
                  vmin=vmin, vmax=vmax, alpha=0.5)
    ax_num_twin = ax_num.twiny()
    # ラベル別積み上げ棒グラフ（横向き）
    ax_num_twin.barh(window_centers, num_label1_array, height=bar_height,
                     color='red', alpha=0.7, label='Group 1')
    ax_num_twin.barh(window_centers, num_label2_array, height=bar_height,
                     left=num_label1_array, color='green', alpha=0.7, label='Group 2')
    ax_num_twin.barh(window_centers, num_label3_array, height=bar_height,
                     left=num_label1_array + num_label2_array, color='blue', alpha=0.7, label='Group 3')
    ax_num_twin.set_xlabel('Number of rocks', fontsize=font_medium, color='black')
    ax_num_twin.tick_params(axis='x', labelcolor='black', labelsize=font_small)
    # フィッティングデータ数を第2横軸に折れ線グラフでプロット（横向き）
    num_fitting_points = num_label2_array + num_label3_array + 1 # グループ１は１つのデータ点にまとめて取り扱う
    ax_num_twin.plot(num_fitting_points, window_centers,
                'k-', linewidth=2, marker='D', markersize=10, label='Number of fitting points')
    ax_num_twin.set_xlabel('Number of fitting points', fontsize=font_medium, color='black')
    ax_num_twin.legend(loc='lower right', fontsize=font_small - 2)
    ax_num.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax_num.set_ylabel('Time [ns]', fontsize=font_medium)
    ax_num.set_ylim(time_max_data, time_min_data)
    ax_num.tick_params(axis='both', which='major', labelsize=font_small)
    ax_num.grid(True, ls='--', alpha=1.0, axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(individual_plots_dir, 'num_rocks_plot.png'), dpi=dpi_png)
    plt.savefig(os.path.join(individual_plots_dir, 'num_rocks_plot.pdf'), dpi=dpi_pdf)
    plt.close()

    # --- rの個別プロット ---
    fig_r, ax_r = plt.subplots(figsize=(18, 6))
    ax_r.imshow(bscan_data, aspect='auto', cmap='seismic',
                extent=[0, bscan_data.shape[1] * trace_interval,
                       time_array[-1], time_array[0]],
                vmin=vmin, vmax=vmax, alpha=0.5)
    ax_r_twin = ax_r.twiny()
    ax_r_twin.plot(r_values_plot, window_centers,
                   'b-', linewidth=2, marker='o', markersize=10, label='r (power-law exponent)')
    ax_r_twin.set_xlabel('r (power-law exponent)', fontsize=font_medium, color='blue')
    ax_r_twin.tick_params(axis='x', labelcolor='blue', labelsize=font_small)
    ax_r_twin.set_xlim(0.3, 1.6)
    ax_r_twin.set_xticks(r_ticks)
    ax_r_twin.set_xticklabels(r_tick_labels)
    ax_r.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax_r.set_ylabel('Time [ns]', fontsize=font_medium)
    ax_r.set_ylim(time_max_data, time_min_data)
    ax_r.tick_params(axis='both', which='major', labelsize=font_small)
    ax_r.grid(True, ls='--', alpha=1.0, axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(individual_plots_dir, 'r_plot.png'), dpi=dpi_png)
    plt.savefig(os.path.join(individual_plots_dir, 'r_plot.pdf'), dpi=dpi_pdf)
    plt.close()

    # --- kの個別プロット ---
    fig_k, ax_k = plt.subplots(figsize=(18, 6))
    ax_k.imshow(bscan_data, aspect='auto', cmap='seismic',
                extent=[0, bscan_data.shape[1] * trace_interval,
                       time_array[-1], time_array[0]],
                vmin=vmin, vmax=vmax, alpha=0.5)
    ax_k_twin = ax_k.twiny()
    ax_k_twin.plot(k_values_plot, window_centers,
                   'r-', linewidth=2, marker='s', markersize=10, label='k (scaling factor)')
    ax_k_twin.tick_params(axis='x', labelcolor='red', labelsize=font_small)
    ax_k_twin.set_xlim(0, 22e-3)
    ax_k_twin.set_xticks(k_ticks)
    ax_k_twin.set_xticklabels(k_tick_labels)
    ax_k_twin.set_xlabel('k [×10⁻³ /m²]', fontsize=font_medium, color='red')
    ax_k.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax_k.set_ylabel('Time [ns]', fontsize=font_medium)
    ax_k.set_ylim(time_max_data, time_min_data)
    ax_k.tick_params(axis='both', which='major', labelsize=font_small)
    ax_k.grid(True, ls='--', alpha=1.0, axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(individual_plots_dir, 'k_plot.png'), dpi=dpi_png)
    plt.savefig(os.path.join(individual_plots_dir, 'k_plot.pdf'), dpi=dpi_pdf)
    plt.close()

    # --- p値の個別プロット ---
    fig_p, ax_p = plt.subplots(figsize=(18, 6))
    ax_p.imshow(bscan_data, aspect='auto', cmap='seismic',
                extent=[0, bscan_data.shape[1] * trace_interval,
                       time_array[-1], time_array[0]],
                vmin=vmin, vmax=vmax, alpha=0.5)
    ax_p_twin = ax_p.twiny()
    ax_p_twin.semilogx(p_values_plot, window_centers,
                       'g-', linewidth=2, marker='^', markersize=10, label='p-value')
    ax_p_twin.axvline(x=0.05, color='black', linestyle='--', linewidth=1.5, label='p = 0.05')
    ax_p_twin.set_xlabel('p-value', fontsize=font_medium, color='green')
    ax_p_twin.tick_params(axis='x', labelcolor='green', labelsize=font_small)
    ax_p_twin.set_xlim(1e-4, 1.5)
    ax_p_twin.set_xticks(p_ticks)
    ax_p_twin.set_xticklabels(p_tick_labels)
    ax_p.set_xlabel('Moving distance [m]', fontsize=font_medium)
    ax_p.set_ylabel('Time [ns]', fontsize=font_medium)
    ax_p.set_ylim(time_max_data, time_min_data)
    ax_p.tick_params(axis='both', which='major', labelsize=font_small)
    ax_p.grid(True, ls='--', alpha=1.0, axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(individual_plots_dir, 'p_value_plot.png'), dpi=dpi_png)
    plt.savefig(os.path.join(individual_plots_dir, 'p_value_plot.pdf'), dpi=dpi_pdf)
    plt.close()

    print(f'個別プロット保存: {individual_plots_dir}')

    # p値統計を計算（p値がnanの場合はp > 0.05とみなす）
    p_values_array = np.array(p_values)
    valid_p_count = 0
    significant_count = 0
    for p in p_values_array:
        if np.isnan(p):
            valid_p_count += 1  # nanはp > 0.05とみなす
        else:
            valid_p_count += 1
            if p <= 0.05:
                significant_count += 1

    significant_ratio = (significant_count / valid_p_count * 100) if valid_p_count > 0 else 0.0

    # 統計情報をテキストファイルに保存
    stats_path = f'{output_path}_statistics.txt'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('# Vertical Moving Window RSFD Statistics (Area-Normalized)\n')
        f.write('# ========================================================\n\n')
        f.write(f'Window width: {window_width:.2f} ns\n')
        f.write(f'Step size: {step_size:.2f} ns\n')
        f.write(f'Number of windows: {len(window_centers)}\n')
        f.write(f'Note: k values are area-normalized [/m²]\n\n')
        f.write('# P-value Statistics\n')
        f.write(f'Windows with p ≤ 0.05: {significant_count} / {valid_p_count} ({significant_ratio:.2f}%)\n')
        f.write(f'Note: NaN p-values are treated as p > 0.05\n\n')
        f.write('Window\tCenter [ns]\t Time range [ns]\tk [/m²]\tr\tR2\tp-value\tNum rocks\tArea [m²]\n')
        for i, center in enumerate(window_centers):
            f.write(f'{i+1}\t{center:.2f}\t{center - window_width/2:.2f} - {center + window_width/2:.2f}\t{k_values[i]:.4e}\t{r_values[i]:.4f}\t'
                   f'{R2_values[i]:.4f}\t{p_values[i]:.4e}\t{num_rocks_list[i]}\t{area_list[i]:.4f}\n')

    print(f'統計情報保存: {stats_path}')
    print(f'p値統計: {significant_count}/{valid_p_count}個のウィンドウでp ≤ 0.05 ({significant_ratio:.2f}%)')

# ------------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------------
print('=== RSFD Moving Window Analysis Tool ===')
print('移動ウィンドウによるRSFD解析ツール\n')

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
label_path = select_label_file(label_files)

# ------------------------------------------------------------------
# 3. 解析方向とウィンドウ幅の入力
# ------------------------------------------------------------------
print('\n=== 解析パラメータ ===')
print('解析方向を選択してください:')
print('1: 水平方向 (Moving distance)')
print('2: 深さ方向 (Depth / Time)')
direction_choice = input('選択 (1 or 2): ').strip()

if direction_choice not in ['1', '2']:
    raise ValueError('1 または 2 を入力してください。')

if direction_choice == '1':
    analysis_direction = 'horizontal'
    window_width = float(input('ウィンドウ幅 [m] を入力してください: ').strip())
    if window_width <= 0:
        raise ValueError('ウィンドウ幅は正の値を指定してください。')
else:
    analysis_direction = 'vertical'
    window_width = float(input('ウィンドウ幅 [ns] を入力してください: ').strip())
    if window_width <= 0:
        raise ValueError('ウィンドウ幅は正の値を指定してください。')

# ステップサイズ（ウィンドウ幅の20%）
step_size = window_width * 0.2
print(f'ステップサイズ: {step_size:.2f} {"m" if analysis_direction == "horizontal" else "ns"} (ウィンドウ幅の20%)')

# ------------------------------------------------------------------
# 4. データ読み込み
# ------------------------------------------------------------------
print('\nデータ読み込み中...')

# B-scanデータの読み込み
print('B-scanデータ読み込み中...')
bscan_data = np.loadtxt(bscan_path, delimiter=' ')
print(f'B-scan形状: {bscan_data.shape}')

# 岩石データの読み込み
print('岩石ラベルデータ読み込み中...')
rock_data = load_rock_data(label_path)
print(f'岩石数: {len(rock_data["label"])}')

# 岩石サイズと時間位置の計算
sizes = calculate_rock_sizes(rock_data)
time_positions = get_time_position(rock_data)

# ------------------------------------------------------------------
# 5. データ範囲の計算（make_RSFD_grid_comparison.pyと同じロジック）
# ------------------------------------------------------------------
# 物理定数
sample_interval = 0.312500e-9  # [s] - サンプル間隔
trace_interval = 3.6e-2        # [m] - トレース間隔
epsilon_r = 4.5                # 比誘電率
c = 299792458                  # [m/s]

# 時間方向：最小値はB-scanから、最大値はlabel.jsonから取得
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
        time_ns = (non_nan_idx[0] - time_zero_idx) * sample_interval * 1e9
        first_valid_times.append(time_ns)

if first_valid_times:
    time_min_data = min(first_valid_times)
else:
    time_min_data = 0.0
    print('警告: B-scanに有効なデータが見つかりませんでした。time_min=0として処理を続行します。')

# 最大値はlabel.jsonから取得
lab = rock_data['label']
y = rock_data['y']
time_top = rock_data['time_top']

time_values_group1 = y[lab == 1]
time_values_others = time_top[(lab != 1) & (~np.isnan(time_top))]
time_values_all = np.concatenate([time_values_group1, time_values_others])
time_max_data = np.max(time_values_all)

# 距離方向：B-scanから取得
dist_min_data = 0.0
dist_max_data = bscan_data.shape[1] * trace_interval

print(f'\n時間範囲: {time_min_data:.2f} - {time_max_data:.2f} ns')
print(f'距離範囲: {dist_min_data:.2f} - {dist_max_data:.2f} m')

# ------------------------------------------------------------------
# 6. 出力ディレクトリの作成
# ------------------------------------------------------------------
output_dir = os.path.join(bscan_dir, 'RSFD_moving_window_comparison')
os.makedirs(output_dir, exist_ok=True)

# サブディレクトリ名
if analysis_direction == 'horizontal':
    sub_dir_name = f'horizontal_window{window_width:.0f}m'
else:
    sub_dir_name = f'vertical_window{window_width:.1f}ns'

sub_dir = os.path.join(output_dir, sub_dir_name)
os.makedirs(sub_dir, exist_ok=True)

print(f'\n出力ディレクトリ: {sub_dir}')

# ------------------------------------------------------------------
# 7. プロット作成
# ------------------------------------------------------------------
print('\n=== プロット作成開始 ===')

if analysis_direction == 'horizontal':
    output_path = os.path.join(sub_dir, 'horizontal_moving_window_rsfd')
    create_horizontal_moving_window_plot(
        bscan_data, rock_data, sizes, time_positions,
        window_width, step_size,
        time_min_data, time_max_data,
        dist_min_data, dist_max_data,
        sample_interval, trace_interval,
        epsilon_r, c, output_path
    )
else:
    output_path = os.path.join(sub_dir, 'vertical_moving_window_rsfd')
    create_vertical_moving_window_plot(
        bscan_data, rock_data, sizes, time_positions,
        window_width, step_size,
        time_min_data, time_max_data,
        dist_min_data, dist_max_data,
        sample_interval, trace_interval,
        epsilon_r, c, output_path
    )

print('\n=== 処理完了 ===')
print(f'出力ディレクトリ: {sub_dir}')
