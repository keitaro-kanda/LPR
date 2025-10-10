#!/usr/bin/env python3
# RSFD_generator.py
# ------------------------------------------------------------
# ラベル JSON から
#   1) ラベル1→1 cm, ラベル2→6 cm, ラベル3→式で計算
# の岩石サイズを取得し，
# 線形‑線形の累積サイズ‑頻度分布 (個数) を描画・保存し，
# べき則／指数関数フィッティングおよび比較プロットを滑らかに追加
# ------------------------------------------------------------

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. 入力ファイルチェック
# ------------------------------------------------------------------
print('検出された岩石のラベルデータファイル(.json)のパスを入力してください:')
data_path = input().strip()
if not (os.path.exists(data_path) and data_path.lower().endswith('.json')):
    raise FileNotFoundError('正しい .json ファイルを指定してください。')

# モード選択
print('\n=== データ範囲モード選択 ===')
print('1: 全範囲のデータを使用')
print('2: 特定の時間・距離範囲のデータのみを使用')
print('3: 特定の時間・距離範囲のデータのみを取り除いて使用')
mode = input('モードを選択してください (1/2/3): ').strip()

if mode not in ['1', '2', '3']:
    raise ValueError('モードは1, 2, 3のいずれかを選択してください。')

# データ範囲の入力
if mode == '1':
    # モード1: 全範囲使用
    time_range = ''
    horizontal_range = ''
    print('全範囲のデータを使用します。')
else:
    # モード2/3: 範囲入力
    print('\n=== データ範囲指定 ===')
    if mode == '2':
        print('指定した範囲のデータのみを使用します。')
    else:  # mode == '3'
        print('指定した範囲のデータを除外します。')
    time_range = input('時間範囲 [ns] を入力してください（例: 50-100, Enter: 指定なし）: ').strip()
    horizontal_range = input('水平位置範囲 [m] を入力してください（例: 0-100, Enter: 指定なし）: ').strip()

try:
    if time_range:
        time_min, time_max = map(float, time_range.split('-'))
    else:
        time_min, time_max = None, None

    if horizontal_range:
        horizontal_min, horizontal_max = map(float, horizontal_range.split('-'))
    else:
        horizontal_min, horizontal_max = None, None
except ValueError:
    raise ValueError('範囲の入力形式が正しくありません。例: 0-100')

# 出力フォルダ
base_dir = os.path.join(os.path.dirname(os.path.dirname(data_path)), 'RSFD')
file_name = os.path.splitext(os.path.basename(data_path))[0]

# 範囲指定に応じた出力ディレクトリ名
if mode == '1':
    # モード1: 全範囲
    output_dir = os.path.join(base_dir, f'{file_name}_full_range')
elif mode == '2':
    # モード2: 特定範囲のみ使用（既存の命名）
    if time_range and not horizontal_range:
        output_dir = os.path.join(base_dir, f'{file_name}_t{time_min}-{time_max}')
    elif horizontal_range and not time_range:
        output_dir = os.path.join(base_dir, f'{file_name}_x{horizontal_min}-{horizontal_max}')
    elif time_range and horizontal_range:
        output_dir = os.path.join(base_dir, f'{file_name}_t{time_min}-{time_max}_x{horizontal_min}-{horizontal_max}')
    else:
        output_dir = os.path.join(base_dir, f'{file_name}_full_range')
elif mode == '3':
    # モード3: 特定範囲を除外
    if time_range and not horizontal_range:
        output_dir = os.path.join(base_dir, f'{file_name}_remove_t{time_min}-{time_max}')
    elif horizontal_range and not time_range:
        output_dir = os.path.join(base_dir, f'{file_name}_remove_x{horizontal_min}-{horizontal_max}')
    elif time_range and horizontal_range:
        output_dir = os.path.join(base_dir, f'{file_name}_remove_t{time_min}-{time_max}_x{horizontal_min}-{horizontal_max}')
    else:
        output_dir = os.path.join(base_dir, f'{file_name}_full_range')

os.makedirs(output_dir, exist_ok=True)
# プロット用サブフォルダ（カテゴリ別）
output_dir_linear = os.path.join(output_dir, '1_non_fit')
output_dir_power = os.path.join(output_dir, '2_power_law_fit')
output_dir_exp = os.path.join(output_dir, '3_exponential_fit')
output_dir_comparison = os.path.join(output_dir, '4_fit_comparison')
os.makedirs(output_dir_linear, exist_ok=True)
os.makedirs(output_dir_power, exist_ok=True)
os.makedirs(output_dir_exp, exist_ok=True)
os.makedirs(output_dir_comparison, exist_ok=True)

# ------------------------------------------------------------------
# 2. JSON 読み込み
# ------------------------------------------------------------------
def none_to_nan(v):
    return np.nan if v is None else v

with open(data_path, 'r') as f:
    results = json.load(f).get('results', {})

x   = np.array([v['x']            for v in results.values()])
y   = np.array([v['y']            for v in results.values()])
lab = np.array([v['label']        for v in results.values()], dtype=int)
time_top    = np.array([none_to_nan(v['time_top'])    for v in results.values()], dtype=float)
time_bottom = np.array([none_to_nan(v['time_bottom']) for v in results.values()], dtype=float)
print('ラベルデータ読み込み完了:', len(lab), '個')

# データ範囲フィルタリング
original_count = len(lab)
if time_min is not None and time_max is not None:
    # Group1: y値で判定、Group2-6: time_topで判定
    mask_group1 = (lab == 1) & (y >= time_min) & (y <= time_max)
    mask_others = (lab != 1) & (time_top >= time_min) & (time_top <= time_max)
    time_mask = mask_group1 | mask_others

    # モード3の場合は論理反転（指定範囲を除外）
    if mode == '3':
        time_mask = ~time_mask

    x = x[time_mask]
    y = y[time_mask]
    lab = lab[time_mask]
    time_top = time_top[time_mask]
    time_bottom = time_bottom[time_mask]

    if mode == '2':
        print(f'時間範囲フィルタリング後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
    elif mode == '3':
        print(f'時間範囲除外後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')

if horizontal_min is not None and horizontal_max is not None:
    horizontal_mask = (x >= horizontal_min) & (x <= horizontal_max)

    # モード3の場合は論理反転（指定範囲を除外）
    if mode == '3':
        horizontal_mask = ~horizontal_mask

    x = x[horizontal_mask]
    y = y[horizontal_mask]
    lab = lab[horizontal_mask]
    time_top = time_top[horizontal_mask]
    time_bottom = time_bottom[horizontal_mask]

    if mode == '2':
        print(f'水平位置範囲フィルタリング後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
    elif mode == '3':
        print(f'水平位置範囲除外後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')

print(f'フィルタリング完了: {len(lab)}個のデータを使用')

# ------------------------------------------------------------------
# 3. ラベル別個数をテキスト出力
# ------------------------------------------------------------------
counts = {k: int(np.sum(lab == k)) for k in range(1, 7)}
with open(os.path.join(output_dir, 'RSFD_counts_by_label.txt'), 'w') as f:
    f.write('# RSFD Label Counts\n')
    f.write(f'# Original data count: {original_count}\n')

    # モードに応じたフィルタ情報の記録
    if mode == '1':
        f.write('# Mode: Full range (no filtering)\n')
    elif mode == '2':
        f.write('# Mode: Use only specified range\n')
        if time_min is not None and time_max is not None:
            f.write(f'# Time range filter: {time_min} - {time_max} ns\n')
        if horizontal_min is not None and horizontal_max is not None:
            f.write(f'# Horizontal range filter: {horizontal_min} - {horizontal_max} m\n')
    elif mode == '3':
        f.write('# Mode: Remove specified range\n')
        if time_min is not None and time_max is not None:
            f.write(f'# Removed time range: {time_min} - {time_max} ns\n')
        if horizontal_min is not None and horizontal_max is not None:
            f.write(f'# Removed horizontal range: {horizontal_min} - {horizontal_max} m\n')

    f.write(f'# Filtered data count: {len(lab)} ({len(lab)/original_count*100:.1f}%)\n')
    f.write('\n')
    for k, v in counts.items():
        f.write(f'Label {k}: {v}\n')

# ------------------------------------------------------------------
# 4. ラベル1・2・3 → サイズ配列を作成
# ------------------------------------------------------------------
size_label1 = np.full(counts[1], 1.0)      # ラベル1：1 cm
size_label2 = np.full(counts[2], 6.0)      # ラベル2：6 cm
mask2_valid = (lab == 2) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
mask3_valid = (lab == 3) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
er = 9.0
c  = 299_792_458  # m/s
sizes_group2 = (time_bottom[mask2_valid] - time_top[mask2_valid]) * 1e-9 * c / np.sqrt(er) * 0.5 * 100  # [cm]
sizes_group3 = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(er) * 0.5 * 100  # [cm]
sizes_group3_max = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(6) * 0.5 * 100  # [cm] # grpup3のエラー範囲
sizes_group3_min = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(15) * 0.5 * 100  # [cm] # group3のエラー範囲
all_sizes_cm_traditional = np.concatenate([size_label1, size_label2, sizes_group3])
all_sizes_cm_estimamte_group2 = np.concatenate([size_label1, sizes_group2, sizes_group3]) # Group2も式で計算した場合
#all_sizes_cm = sizes_group3 # Group3のみだとどうなる？の検証
if all_sizes_cm_traditional.size == 0:
    raise RuntimeError('有効なラベル1–3が見つかりませんでした。')

# ------------------------------------------------------------------
# 5. 累積サイズ‑頻度分布 (≥ size) を計算
# ------------------------------------------------------------------
unique_sizes_traditional = np.sort(np.unique(all_sizes_cm_traditional))
cum_counts_traditional   = np.array([(all_sizes_cm_traditional >= s).sum() for s in unique_sizes_traditional], dtype=int)

unique_sizes_estimate_group2 = np.sort(np.unique(all_sizes_cm_estimamte_group2))
cum_counts_estimate_group2   = np.array([(all_sizes_cm_estimamte_group2 >= s).sum() for s in unique_sizes_estimate_group2], dtype=int)

# ------------------------------------------------------------------
# 6. 汎用プロット関数の定義
# ------------------------------------------------------------------
def create_rsfd_plot(x_data, y_data, xlabel, ylabel, output_path,
                     scale_type='linear', fit_lines=None,
                     show_plot=False, dpi_png=300, dpi_pdf=600,
                     marker='o', linestyle='-', linewidth=1.5, color=None, label=None):
    """
    RSFDプロットを作成・保存する汎用関数

    Parameters:
    -----------
    x_data, y_data : array
        プロットするデータ
    xlabel, ylabel : str
        軸ラベル
    output_path : str
        出力パス（拡張子なし）
    scale_type : str
        'linear', 'semilog', 'loglog'
    fit_lines : list of dict, optional
        フィット曲線のリスト [{'x': x, 'y': y, 'label': label, 'color': color, 'linestyle': style}, ...]
    show_plot : bool
        プロット表示の有無
    dpi_png, dpi_pdf : int
        解像度
    marker, linestyle, linewidth, color, label :
        データプロットのスタイル設定
    """
    plt.figure(figsize=(8, 6))

    # データプロット
    if marker and linestyle:
        plot_kwargs = {'marker': marker, 'linestyle': linestyle, 'linewidth': linewidth}
        if color:
            plot_kwargs['color'] = color
        if label:
            plot_kwargs['label'] = label
        plt.plot(x_data, y_data, **plot_kwargs)
    elif marker:  # scatter plot
        scatter_kwargs = {'marker': marker}
        if color:
            scatter_kwargs['color'] = color
        if label:
            scatter_kwargs['label'] = label
        plt.scatter(x_data, y_data, **scatter_kwargs)

    # フィット曲線の追加
    if fit_lines:
        for fit_line in fit_lines:
            plt.plot(fit_line['x'], fit_line['y'],
                    linestyle=fit_line.get('linestyle', '--'),
                    linewidth=fit_line.get('linewidth', 1.5),
                    color=fit_line.get('color', 'red'),
                    label=fit_line.get('label', ''))

    # 軸スケール設定
    if scale_type == 'semilog':
        plt.yscale('log')
    elif scale_type == 'loglog':
        plt.xscale('log')
        plt.yscale('log')

    # 軸ラベルとグリッド
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(labelsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 凡例（ラベルがある場合のみ）
    if label or fit_lines:
        plt.legend(fontsize=14)

    plt.tight_layout()

    # 保存
    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)

    # 表示
    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f'プロット保存: {output_path}.png')

# ------------------------------------------------------------------
# 7. 線形プロット保存（3種類のスケール）
# ------------------------------------------------------------------
# 7.1 従来手法（Group2=6cm固定）
for scale in ['linear', 'semilog', 'loglog']:
    output_path = os.path.join(output_dir_linear, f'RSFD_linear_{scale}')
    create_rsfd_plot(
        unique_sizes_traditional, cum_counts_traditional,
        'Rock size [cm]', 'Cumulative number of rocks',
        output_path, scale_type=scale,
        show_plot=(scale == 'linear')  # linearのみ表示
    )

# TXT保存
with open(os.path.join(output_dir, 'RSFD_linear.txt'), 'w') as f:
    f.write('# size_cm\tcumulative_count\n')
    for s, n in zip(unique_sizes_traditional, cum_counts_traditional):
        f.write(f'{s:.3f}\t{n}\n')
print('累積データTXT保存: RSFD_linear.txt')

# Label‑3 詳細ダンプ
if mask3_valid.any():
    dump_path = os.path.join(output_dir, 'Label3_detail.txt')
    with open(dump_path, 'w') as f:
        f.write('#x\t y\t time_top\t time_bottom\n')
        for xi, yi, tp, bt in zip(x[mask3_valid], y[mask3_valid], time_top[mask3_valid], time_bottom[mask3_valid]):
            f.write(f'{xi:.6f}\t{yi:.6f}\t{tp:.3f}\t{bt:.3f}\n')
    print('Label‑3 詳細を保存:', dump_path)

# 7.2 Group2もサイズ推定した場合
for scale in ['linear', 'semilog', 'loglog']:
    output_path = os.path.join(output_dir_linear, f'RSFD_linear_estimate_group2_{scale}')
    create_rsfd_plot(
        unique_sizes_estimate_group2, cum_counts_estimate_group2,
        'Rock size [cm]', 'Cumulative number of rocks',
        output_path, scale_type=scale,
        show_plot=(scale == 'linear')  # linearのみ表示
    )

# TXT保存
with open(os.path.join(output_dir, 'RSFD_linear.txt'), 'w') as f:
    f.write('# size_cm\tcumulative_count\n')
    for s, n in zip(unique_sizes_estimate_group2, cum_counts_traditional):
        f.write(f'{s:.3f}\t{n}\n')
print('累積データTXT保存: RSFD_linear.txt')

# Label‑3 詳細ダンプ
if mask2_valid.any() or mask3_valid.any():
    dump_path = os.path.join(output_dir, 'Label2-3_detail.txt')
    with open(dump_path, 'w') as f:
        f.write('#label\t x\t y\t time_top\t time_bottom\t size_cm\n')

        # Label2とLabel3のデータを統合
        combined_mask = mask2_valid | mask3_valid
        x_combined = x[combined_mask]
        y_combined = y[combined_mask]
        time_top_combined = time_top[combined_mask]
        time_bottom_combined = time_bottom[combined_mask]
        lab_combined = lab[combined_mask]

        # サイズを計算
        size_cm_combined = (time_bottom_combined - time_top_combined) * 1e-9 * c / np.sqrt(er) * 0.5 * 100  # [cm]

        # サイズの昇順でソート
        sort_indices = np.argsort(size_cm_combined)

        # ソートされた順序で出力
        for i in sort_indices:
            f.write(f'{lab_combined[i]}\t{x_combined[i]:.6f}\t{y_combined[i]:.6f}\t{time_top_combined[i]:.3f}\t{time_bottom_combined[i]:.3f}\t{size_cm_combined[i]:.3f}\n')
    print('Label‑2-3 詳細を保存:', dump_path)

# ------------------------------------------------------------------
# 8. フィッティング: べき則と指数関数
# ------------------------------------------------------------------
def calc_fitting(sizes, counts):
    # 対数変換
    mask = sizes > 0
    log_D = np.log(sizes[mask])
    log_N = np.log(counts[mask])

    # 7.1 べき則フィッティング
    r_pow, log_k_pow = np.polyfit(log_D, log_N, 1)
    k_pow = np.exp(log_k_pow)
    # R^2 計算用予測値
    N_pred_pow = k_pow * sizes**r_pow
    ss_res_pow = np.sum((counts - N_pred_pow)**2)
    ss_tot     = np.sum((counts - np.mean(counts))**2)
    R2_pow     = 1 - ss_res_pow / ss_tot

    # 7.2 指数関数フィッティング
    slope_exp, log_k_exp = np.polyfit(sizes, log_N, 1)
    r_exp = slope_exp
    k_exp = np.exp(log_k_exp)
    # R^2 計算用予測値
    N_pred_exp = k_exp * np.exp(r_exp * sizes)
    ss_res_exp = np.sum((counts - N_pred_exp)**2)
    R2_exp     = 1 - ss_res_exp / ss_tot

    # フィット曲線用に滑らかなサンプル点を生成
    D_fit = np.linspace(sizes.min(), sizes.max(), 200)
    N_pow_fit = k_pow * D_fit**r_pow
    N_exp_fit = k_exp * np.exp(r_exp * D_fit)
    return (k_pow, r_pow, R2_pow, N_pow_fit), (k_exp, r_exp, R2_exp, N_exp_fit), D_fit

(k_pow_trad, r_pow_trad, R2_pow_trad, N_pow_fit_trad),\
    (k_exp_trad, r_exp_trad, R2_exp_trad, N_exp_fit_trad), D_fit_trad\
    = calc_fitting(unique_sizes_traditional, cum_counts_traditional)

(k_pow_est_grp2, r_pow_est_grp2, R2_pow_est_grp2, N_pow_fit_est_grp2),\
    (k_exp_est_grp2, r_exp_est_grp2, R2_exp_est_grp2, N_exp_fit_est_grp2), D_fit_est_grp2\
    = calc_fitting(unique_sizes_estimate_group2, cum_counts_estimate_group2)

# ------------------------------------------------------------------
# 9. プロット: 個別フィット（3種類のスケール）
# ------------------------------------------------------------------
# 9.1 べき則フィット（従来手法）
fit_lines_pow_trad = [{
    'x': D_fit_trad, 'y': N_pow_fit_trad,
    'label': f'Power-law: k={k_pow_trad:.2e}, r={r_pow_trad:.3f}, R²={R2_pow_trad:.4f}',
    'color': 'red', 'linestyle': '--'
}]

for scale in ['linear', 'semilog', 'loglog']:
    output_path = os.path.join(output_dir_power, f'RSFD_power_law_fit_{scale}')
    create_rsfd_plot(
        unique_sizes_traditional, cum_counts_traditional,
        'Rock size [cm]', 'Cumulative number of rocks',
        output_path, scale_type=scale,
        fit_lines=fit_lines_pow_trad,
        marker='o', linestyle='', label='Data',
        show_plot=False
    )

# 9.2 指数関数フィット（従来手法）
fit_lines_exp_trad = [{
    'x': D_fit_trad, 'y': N_exp_fit_trad,
    'label': f'Exponential: k={k_exp_trad:.2e}, r={r_exp_trad:.3f}, R²={R2_exp_trad:.4f}',
    'color': 'green', 'linestyle': '--'
}]

for scale in ['linear', 'semilog', 'loglog']:
    output_path = os.path.join(output_dir_exp, f'RSFD_exponential_fit_{scale}')
    create_rsfd_plot(
        unique_sizes_traditional, cum_counts_traditional,
        'Rock size [cm]', 'Cumulative number of rocks',
        output_path, scale_type=scale,
        fit_lines=fit_lines_exp_trad,
        marker='o', linestyle='', label='Data',
        show_plot=False
    )

# ------------------------------------------------------------------
# 10. プロット: フィッティング比較（3種類のスケール）
# ------------------------------------------------------------------
# 10.1 従来手法（Group2=6cm固定）
fit_lines_comparison_trad = [
    {
        'x': D_fit_trad, 'y': N_pow_fit_trad,
        'label': f'Power-law: k={k_pow_trad:.2e}, r={r_pow_trad:.3f}, R²={R2_pow_trad:.4f}',
        'color': 'red', 'linestyle': '--'
    },
    {
        'x': D_fit_trad, 'y': N_exp_fit_trad,
        'label': f'Exponential: k={k_exp_trad:.2e}, r={r_exp_trad:.3f}, R²={R2_exp_trad:.4f}',
        'color': 'green', 'linestyle': '--'
    }
]

for scale in ['linear', 'semilog', 'loglog']:
    output_path = os.path.join(output_dir_comparison, f'RSFD_fit_comparison_{scale}')
    create_rsfd_plot(
        unique_sizes_traditional, cum_counts_traditional,
        'Rock size [cm]', 'Cumulative number of rocks',
        output_path, scale_type=scale,
        fit_lines=fit_lines_comparison_trad,
        marker='o', linestyle='', label='Data',
        show_plot=(scale == 'linear')  # linearのみ表示
    )

# 10.2 Group2もサイズ推定した場合
fit_lines_comparison_est_grp2 = [
    {
        'x': D_fit_est_grp2, 'y': N_pow_fit_est_grp2,
        'label': f'Power-law: k={k_pow_est_grp2:.2e}, r={r_pow_est_grp2:.3f}, R²={R2_pow_est_grp2:.4f}',
        'color': 'red', 'linestyle': '--'
    },
    {
        'x': D_fit_est_grp2, 'y': N_exp_fit_est_grp2,
        'label': f'Exponential: k={k_exp_est_grp2:.2e}, r={r_exp_est_grp2:.3f}, R²={R2_exp_est_grp2:.4f}',
        'color': 'green', 'linestyle': '--'
    }
]

for scale in ['linear', 'semilog', 'loglog']:
    output_path = os.path.join(output_dir_comparison, f'RSFD_fit_comparison_estimate_group2_{scale}')
    create_rsfd_plot(
        unique_sizes_estimate_group2, cum_counts_estimate_group2,
        'Rock size [cm]', 'Cumulative number of rocks',
        output_path, scale_type=scale,
        fit_lines=fit_lines_comparison_est_grp2,
        marker='o', linestyle='', label='Data',
        show_plot=(scale == 'linear')  # linearのみ表示
    )

print('すべて完了しました！')