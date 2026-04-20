#!/usr/bin/env python3
# make_RSFD_compare_multiple_region.py
# ---------------------------------------------------------------
# 複数範囲のRSFDを比較するツール
# 比較する区間はJSONファイルで指定する
#
# 入力JSONフォーマット:
# {
#   "label_file": "/path/to/rock_labels.json",
#   "group1_size": 1.0,
#   "output_dir": "/path/to/output",
#   "ranges": [
#     {
#       "label": "Range 1: near crater",
#       "time_min": 50.0,
#       "time_max": 100.0,
#       "horizontal_min": null,
#       "horizontal_max": null,
#       "exclude": false,
#       "area": 5000.0
#     }
#   ]
# }
#
# label: 任意の範囲ラベル（省略時は自動生成）
# time_min, time_max: 時間範囲 [ns]（指定なし→ null）
# horizontal_min, horizontal_max: 水平位置範囲 [m]（指定なし→ null）
# exclude: true=指定範囲を除外, false=指定範囲のみ使用
# area: 面積 [m²]
# ---------------------------------------------------------------

import json
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime

# NaNデータ点をプロットする際のy位置（ylim下限より小さい値、軸の下に描画）
NAN_Y_MARKER = 1e-5
# フィッティングに必要な最小データ点数
MIN_FIT_POINTS = 3


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
        return f"p = {p:.3f}"


def format_k_value(k):
    """
    kを10^-3オーダーで統一して表記する
    例: 1.2e-2 → "12×10⁻³", 5.0e-4 → "0.5×10⁻³"
    """
    coefficient = k / 1e-3
    if coefficient >= 10:
        return f"{coefficient:.1f}×10⁻³"
    elif coefficient >= 1:
        return f"{coefficient:.2f}×10⁻³"
    else:
        return f"{coefficient:.2f}×10⁻³"


def calc_fitting(sizes, counts):
    """べき則と指数関数のフィッティングを実行"""
    mask = sizes > 0
    if len(sizes) == 0 or mask.sum() < MIN_FIT_POINTS:
        nan_arr = np.full(200, np.nan)
        nan_val = np.nan
        d_fit = np.linspace(sizes.min(), sizes.max(), 200) if len(sizes) >= 2 else np.linspace(1.0, 10.0, 200)
        return (nan_val, nan_val, nan_val, nan_arr, nan_val, nan_val, nan_val, int(mask.sum()), 0), \
               (nan_val, nan_val, nan_val, nan_arr, nan_val, nan_val, nan_val, int(mask.sum()), 0), \
               d_fit

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

    # 指数関数フィッティング (Exponential: log N = rD + log k)
    X_exp = sm.add_constant(sizes[mask])
    model_exp = sm.OLS(log_N, X_exp)
    results_exp = model_exp.fit()

    log_k_exp, r_exp = results_exp.params
    k_exp = np.exp(log_k_exp)
    R2_exp = results_exp.rsquared
    r_exp_se = results_exp.bse[1]
    r_exp_t = results_exp.tvalues[1]
    r_exp_p = results_exp.pvalues[1]
    dof_exp = results_exp.df_resid
    n_exp = int(results_exp.nobs)

    D_fit = np.linspace(sizes.min(), sizes.max(), 200)
    N_pow_fit = k_pow * D_fit**r_pow
    N_exp_fit = k_exp * np.exp(r_exp * D_fit)

    return (k_pow, np.abs(r_pow), R2_pow, N_pow_fit, r_pow_t, r_pow_p, r_pow_se, n_pow, dof_pow), \
           (k_exp, np.abs(r_exp), R2_exp, N_exp_fit, r_exp_t, r_exp_p, r_exp_se, n_exp, dof_exp), \
           D_fit


def calc_fitting_area_normalized(sizes, counts, area):
    """面積規格化されたべき則フィッティングを実行"""
    counts_normalized = counts / area if len(counts) > 0 else np.array([np.nan])
    mask = sizes > 0
    if len(sizes) == 0 or mask.sum() < MIN_FIT_POINTS:
        nan_arr = np.full(200, np.nan)
        nan_val = np.nan
        d_fit = np.linspace(sizes.min(), sizes.max(), 200) if len(sizes) >= 2 else np.linspace(1.0, 10.0, 200)
        return (nan_val, nan_val, nan_val, nan_arr, nan_val, nan_val, nan_val, int(mask.sum()), 0), \
               d_fit, counts_normalized

    log_D = np.log(sizes[mask])
    log_N = np.log(counts_normalized[mask])

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

    D_fit = np.linspace(sizes.min(), sizes.max(), 200)
    N_pow_fit = k_pow * D_fit**r_pow

    return (k_pow, np.abs(r_pow), R2_pow, N_pow_fit, r_pow_t, r_pow_p, r_pow_se, n_pow, dof_pow), \
           D_fit, counts_normalized


def create_multi_range_comparison_plot(ranges_data_list, xlabel, ylabel, output_path,
                                       show_plot=False, dpi_png=300, dpi_pdf=600,
                                       xlim=None):
    """
    複数範囲のRSFDを1つのプロットに重ねて表示する関数（両対数グラフのみ）

    Parameters:
    -----------
    ranges_data_list : list of dict
        各範囲のデータリスト。各要素は以下のキーを持つ辞書:
        {
            'x_data': array, 'y_data': array,
            'label': str, 'color': str,
            'fit_x': array, 'fit_y': array, 'fit_params': dict
        }
    xlabel, ylabel : str
        軸ラベル
    output_path : str
        出力パス（拡張子なし）
    show_plot : bool
        プロット表示の有無
    dpi_png, dpi_pdf : int
        解像度
    xlim : tuple, optional
        x軸範囲 (xmin, xmax)
    """
    plt.figure(figsize=(10, 8))

    for range_data in ranges_data_list:
        x_arr = np.asarray(range_data['x_data'], dtype=float)
        y_arr = np.asarray(range_data['y_data'], dtype=float)
        nan_mask = np.isnan(y_arr)

        if nan_mask.any():
            x_nan = np.where(np.isnan(x_arr[nan_mask]), 1.0, x_arr[nan_mask])
            plt.plot(x_nan, np.full(nan_mask.sum(), NAN_Y_MARKER),
                     marker='v', linestyle='', color=range_data['color'],
                     markersize=8, clip_on=False, zorder=5,
                     label=f"{range_data['label']} (No data)")
        if (~nan_mask).any():
            plt.plot(x_arr[~nan_mask], y_arr[~nan_mask],
                     marker='o', linestyle='',
                     color=range_data['color'],
                     label=f"{range_data['label']} (Observed)")

        if not np.isnan(range_data['fit_params']['k']):
            k_str = format_k_value(range_data['fit_params']['k'])
            r_val = range_data['fit_params']['r']
            R2_val = range_data['fit_params']['R2']
            p_str = range_data['fit_params']['p_str']
            plt.plot(range_data['fit_x'], range_data['fit_y'],
                    linestyle='--', linewidth=1.5,
                    color=range_data['color'],
                    label=f"{range_data['label']} (Fit: N = {k_str} D⁻{r_val:.2f}, R²={R2_val:.3f}, {p_str})")

    plt.xscale('log')
    plt.yscale('log')

    if xlim:
        plt.xlim(max(xlim[0], 0.5), xlim[1])
    plt.ylim(3e-5, 5e-2)

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(labelsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True)
    plt.tight_layout()

    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f'プロット保存: {output_path}.png')


# ------------------------------------------------------------------
# 1. 入力JSONファイルの読み込み
# ------------------------------------------------------------------
print('=== RSFD 複数範囲比較ツール ===')
print('比較設定JSONファイルのパスを入力してください:')
config_json_path = input().strip()

if not (os.path.exists(config_json_path) and config_json_path.lower().endswith('.json')):
    raise FileNotFoundError('正しい .json ファイルを指定してください。')

with open(config_json_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# 必須フィールドの確認
required_keys = ['label_file', 'group1_size', 'output_dir', 'ranges']
for key in required_keys:
    if key not in config:
        raise ValueError(f'設定JSONに必須フィールドがありません: {key}')

data_path = config['label_file']
group1_size = float(config['group1_size'])
output_dir = config['output_dir']
ranges_list = config['ranges']

if not (os.path.exists(data_path) and data_path.lower().endswith('.json')):
    raise FileNotFoundError(f'ラベルデータファイルが見つかりません: {data_path}')

if len(ranges_list) < 1:
    raise ValueError('rangesには1つ以上の範囲を指定してください。')

print(f'ラベルファイル: {data_path}')
print(f'Group 1 サイズ: {group1_size} cm')
print(f'出力ディレクトリ: {output_dir}')
print(f'比較範囲数: {len(ranges_list)}')

# ------------------------------------------------------------------
# 2. 出力ディレクトリの作成
# ------------------------------------------------------------------
output_dir_group1_3 = os.path.join(output_dir, 'RSFD_group1-3_comparison')
output_dir_group2_3 = os.path.join(output_dir, 'RSFD_group2-3_comparison')
output_dir_comparison = os.path.join(output_dir, 'RSFD_comparison')
os.makedirs(output_dir_group1_3, exist_ok=True)
os.makedirs(output_dir_group2_3, exist_ok=True)
os.makedirs(output_dir_comparison, exist_ok=True)

# 入力JSONファイルを出力ディレクトリにコピー
config_copy_path = os.path.join(output_dir, os.path.basename(config_json_path))
shutil.copy2(config_json_path, config_copy_path)
print(f'設定ファイルをコピー: {config_copy_path}')

# ------------------------------------------------------------------
# 3. ラベルデータの読み込み
# ------------------------------------------------------------------
with open(data_path, 'r') as f:
    results = json.load(f).get('results', {})

x_all = np.array([v['x'] for v in results.values()])
y_all = np.array([v['y'] for v in results.values()])
lab_all = np.array([v['label'] for v in results.values()], dtype=int)
time_top_all = np.array([none_to_nan(v['time_top']) for v in results.values()], dtype=float)
time_bottom_all = np.array([none_to_nan(v['time_bottom']) for v in results.values()], dtype=float)
print(f'ラベルデータ読み込み完了: {len(lab_all)}個')

# ------------------------------------------------------------------
# 4. 各範囲の処理
# ------------------------------------------------------------------
print('\n=== 複数範囲比較処理を開始 ===')

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
          'olive', 'cyan', 'magenta', 'yellow', 'navy', 'teal', 'maroon']

all_ranges_data_area_normalized_est = []
all_ranges_data_area_normalized_2_3 = []
all_ranges_summary = []

num_ranges = len(ranges_list)

for range_idx, range_info in enumerate(ranges_list):
    print(f'\n--- 範囲 {range_idx + 1}/{num_ranges} の処理 ---')

    # データをコピー
    x = x_all.copy()
    y = y_all.copy()
    lab = lab_all.copy()
    time_top = time_top_all.copy()
    time_bottom = time_bottom_all.copy()

    # データ範囲フィルタリング
    original_count = len(lab)
    time_min = range_info.get('time_min')
    time_max = range_info.get('time_max')
    horizontal_min = range_info.get('horizontal_min')
    horizontal_max = range_info.get('horizontal_max')
    exclude_flag = range_info.get('exclude', False)

    if time_min is not None and time_max is not None:
        mask_group1 = (lab == 1) & (y >= time_min) & (y <= time_max)
        mask_others = (lab != 1) & (time_top >= time_min) & (time_top <= time_max)
        time_mask = mask_group1 | mask_others

        if exclude_flag:
            time_mask = ~time_mask

        x = x[time_mask]
        y = y[time_mask]
        lab = lab[time_mask]
        time_top = time_top[time_mask]
        time_bottom = time_bottom[time_mask]

        if exclude_flag:
            print(f'時間範囲除外後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
        else:
            print(f'時間範囲フィルタリング後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')

    if horizontal_min is not None and horizontal_max is not None:
        horizontal_mask = (x >= horizontal_min) & (x <= horizontal_max)

        if exclude_flag:
            horizontal_mask = ~horizontal_mask

        x = x[horizontal_mask]
        y = y[horizontal_mask]
        lab = lab[horizontal_mask]
        time_top = time_top[horizontal_mask]
        time_bottom = time_bottom[horizontal_mask]

        if exclude_flag:
            print(f'水平位置範囲除外後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
        else:
            print(f'水平位置範囲フィルタリング後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')

    print(f'フィルタリング完了: {len(lab)}個のデータを使用')

    # サイズ配列を作成
    counts = {k: int(np.sum(lab == k)) for k in range(1, 7)}
    size_label1 = np.full(counts[1], group1_size)
    size_label2 = np.full(counts[2], 6.0)
    mask2_valid = (lab == 2) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
    mask3_valid = (lab == 3) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
    er = 9.0
    c = 299_792_458
    sizes_group2 = (time_bottom[mask2_valid] - time_top[mask2_valid]) * 1e-9 * c / np.sqrt(er) * 0.5 * 100
    sizes_group3 = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(er) * 0.5 * 100

    sizes_group2 = np.round(sizes_group2, decimals=3)
    sizes_group3 = np.round(sizes_group3, decimals=3)

    # 範囲ラベル作成
    if 'label' in range_info and range_info['label']:
        range_label = range_info['label']
    else:
        range_label_parts = []
        if time_min is not None and time_max is not None:
            range_label_parts.append(f"t{time_min}-{time_max}")
        if horizontal_min is not None and horizontal_max is not None:
            range_label_parts.append(f"x{horizontal_min}-{horizontal_max}")

        if range_label_parts:
            base_label = f"Range {range_idx + 1}: {', '.join(range_label_parts)}"
        else:
            base_label = f"Range {range_idx + 1}: Full"

        if exclude_flag:
            range_label = f"{base_label} (exclude)"
        else:
            range_label = base_label

    color = colors[range_idx % len(colors)]

    # 1) Group2サイズ推定
    all_sizes_estimate_group2 = np.concatenate([size_label1, sizes_group2, sizes_group3])
    unique_sizes_estimate_group2 = np.sort(np.unique(all_sizes_estimate_group2))
    cum_counts_estimate_group2 = np.array([np.sum(all_sizes_estimate_group2 >= s) for s in unique_sizes_estimate_group2])

    # 2) Group2-3のみ
    all_sizes_group2_3 = np.concatenate([sizes_group2, sizes_group3])
    if len(all_sizes_group2_3) == 0:
        unique_sizes_group2_3 = np.array([1.0])
        cum_counts_group2_3 = np.array([np.nan])
    else:
        unique_sizes_group2_3 = np.sort(np.unique(all_sizes_group2_3))
        cum_counts_group2_3 = np.array([np.sum(all_sizes_group2_3 >= s) for s in unique_sizes_group2_3])

    # 3) 面積規格化 Group1-3
    area_range = range_info.get('area') or 16136
    (k_pow_area_est, r_pow_area_est, R2_pow_area_est, N_pow_fit_area_est,
     t_pow_area_est, p_pow_area_est, se_pow_area_est, n_pow_area_est, dof_pow_area_est), \
    D_fit_area_est, cum_counts_area_est = calc_fitting_area_normalized(
        unique_sizes_estimate_group2, cum_counts_estimate_group2, area_range)

    all_ranges_data_area_normalized_est.append({
        'x_data': unique_sizes_estimate_group2,
        'y_data': cum_counts_area_est,
        'fit_x': D_fit_area_est,
        'fit_y': N_pow_fit_area_est,
        'fit_params': {
            'k': k_pow_area_est,
            'r': r_pow_area_est,
            'R2': R2_pow_area_est,
            'p_str': format_p_value(p_pow_area_est),
            'area': area_range
        },
        'label': range_label,
        'color': color
    })

    # 4) 面積規格化 Group2-3
    (k_pow_area_2_3, r_pow_area_2_3, R2_pow_area_2_3, N_pow_fit_area_2_3,
     t_pow_area_2_3, p_pow_area_2_3, se_pow_area_2_3, n_pow_area_2_3, dof_pow_area_2_3), \
    D_fit_area_2_3, cum_counts_area_2_3 = calc_fitting_area_normalized(
        unique_sizes_group2_3, cum_counts_group2_3, area_range)

    all_ranges_data_area_normalized_2_3.append({
        'x_data': unique_sizes_group2_3,
        'y_data': cum_counts_area_2_3,
        'fit_x': D_fit_area_2_3,
        'fit_y': N_pow_fit_area_2_3,
        'fit_params': {
            'k': k_pow_area_2_3,
            'r': r_pow_area_2_3,
            'R2': R2_pow_area_2_3,
            'p_str': format_p_value(p_pow_area_2_3),
            'area': area_range
        },
        'label': range_label,
        'color': color
    })

    # 5) サマリー情報を収集
    all_ranges_summary.append({
        'range_label': range_label,
        'time_min': time_min,
        'time_max': time_max,
        'horizontal_min': horizontal_min,
        'horizontal_max': horizontal_max,
        'area': area_range,
        'exclude': exclude_flag,
        'total_rocks': len(lab),
        'label_1_count': counts[1],
        'label_2_count': counts[2],
        'label_3_count': counts[3],
        'group1_3_fit': {
            'k': k_pow_area_est,
            'r': r_pow_area_est,
            'R2': R2_pow_area_est,
            'p': p_pow_area_est
        },
        'group2_3_fit': {
            'k': k_pow_area_2_3,
            'r': r_pow_area_2_3,
            'R2': R2_pow_area_2_3,
            'p': p_pow_area_2_3
        }
    })

# ------------------------------------------------------------------
# 5. 比較プロット生成
# ------------------------------------------------------------------
print('\n=== 比較プロット生成中 ===')

# 面積規格化 Group1-3 のプロット
output_path = os.path.join(output_dir_group1_3, 'RSFD_group1-3')
create_multi_range_comparison_plot(
    all_ranges_data_area_normalized_est, 'Rock size [cm]', 'Cumulative number of rocks /m²',
    output_path, show_plot=False, xlim=(0, 50)
)

# 面積規格化 Group2-3 のプロット
output_path = os.path.join(output_dir_group2_3, 'RSFD_group2-3')
create_multi_range_comparison_plot(
    all_ranges_data_area_normalized_2_3, 'Rock size [cm]', 'Cumulative number of rocks /m²',
    output_path, show_plot=False, xlim=(0, 50)
)

print('複数範囲比較プロット完了')

# ------------------------------------------------------------------
# 6. サマリーファイルの保存
# ------------------------------------------------------------------
summary_path = os.path.join(output_dir, 'multi_range_comparison_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('# Multi-Range Comparison Summary\n')
    f.write(f'# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write(f'# Config file: {config_json_path}\n')
    f.write(f'# Number of ranges: {num_ranges}\n')
    f.write('#\n')

    for i, summary in enumerate(all_ranges_summary, 1):
        f.write(f'# ===== Range {i}: {summary["range_label"]} =====\n')
        if summary['time_min'] is not None and summary['time_max'] is not None:
            f.write(f'# Time range: {summary["time_min"]}-{summary["time_max"]} ns\n')
        else:
            f.write('# Time range: Not specified\n')
        if summary['horizontal_min'] is not None and summary['horizontal_max'] is not None:
            f.write(f'# Horizontal range: {summary["horizontal_min"]}-{summary["horizontal_max"]} m\n')
        else:
            f.write('# Horizontal range: Not specified\n')
        f.write(f'# Area: {summary["area"]} m²\n')
        if summary['exclude']:
            f.write('# Mode: Exclude specified range\n')
        else:
            f.write('# Mode: Use specified range only\n')
        f.write(f'# Total rocks (Label 1-3): {summary["total_rocks"]}\n')
        f.write(f'# Label 1: {summary["label_1_count"]}\n')
        f.write(f'# Label 2: {summary["label_2_count"]}\n')
        f.write(f'# Label 3: {summary["label_3_count"]}\n')
        f.write('#\n')
        f.write('# [Group1-3 Fitting (Area Normalized)]\n')
        f.write(f'# k: {summary["group1_3_fit"]["k"]:.4e}\n')
        f.write(f'# r: {summary["group1_3_fit"]["r"]:.4f}\n')
        f.write(f'# R²: {summary["group1_3_fit"]["R2"]:.4f}\n')
        f.write(f'# p: {summary["group1_3_fit"]["p"]:.4e}\n')
        f.write('#\n')
        f.write('# [Group2-3 Fitting (Area Normalized)]\n')
        f.write(f'# k: {summary["group2_3_fit"]["k"]:.4e}\n')
        f.write(f'# r: {summary["group2_3_fit"]["r"]:.4f}\n')
        f.write(f'# R²: {summary["group2_3_fit"]["R2"]:.4f}\n')
        f.write(f'# p: {summary["group2_3_fit"]["p"]:.4e}\n')
        f.write('#\n')

print(f'サマリーファイル保存: {summary_path}')
print('\nすべて完了しました！')
