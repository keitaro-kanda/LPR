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

# 出力フォルダ
base_dir = os.path.join(os.path.dirname(os.path.dirname(data_path)), 'RSFD')
file_name = os.path.splitext(os.path.basename(data_path))[0]
output_dir = os.path.join(base_dir, file_name)
os.makedirs(output_dir, exist_ok=True)
# プロット用サブフォルダ
output_dir_plot = os.path.join(output_dir, 'plots')
os.makedirs(output_dir_plot, exist_ok=True)

# ------------------------------------------------------------------
# 2. JSON 読み込み
# ------------------------------------------------------------------
def none_to_nan(v):
    return np.nan if v is None else v

with open(data_path, 'r') as f:
    results = json.load(f).get('results', {})

x   = np.array([v['x']            for v in results.values()])
t   = np.array([v['y']            for v in results.values()])
lab = np.array([v['label']        for v in results.values()], dtype=int)
time_top    = np.array([none_to_nan(v['time_top'])    for v in results.values()], dtype=float)
time_bottom = np.array([none_to_nan(v['time_bottom']) for v in results.values()], dtype=float)
print('ラベルデータ読み込み完了:', len(lab), '個')

# ------------------------------------------------------------------
# 3. ラベル別個数をテキスト出力
# ------------------------------------------------------------------
counts = {k: int(np.sum(lab == k)) for k in range(1, 7)}
with open(os.path.join(output_dir, 'RSFD_counts_by_label.txt'), 'w') as f:
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
# 6. 線形‑線形プロット保存
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(unique_sizes_traditional, cum_counts_traditional, marker='o', linestyle='-', linewidth=1.5)
plt.xlabel('Rock size [cm]', fontsize=20)
plt.ylabel('Cumulative number of rocks', fontsize=20)
plt.tick_params(labelsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
linear_png = os.path.join(output_dir_plot, 'RSFD_linear.png')
linear_pdf = os.path.join(output_dir_plot, 'RSFD_linear.pdf')
plt.savefig(linear_png, dpi=300)
plt.savefig(linear_pdf, dpi=600)
plt.show()
print('線形‑線形プロット保存:', linear_png)

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
        f.write('#x\t t\t time_top\t time_bottom\n')
        for xi, ti, tp, bt in zip(x[mask3_valid], t[mask3_valid], time_top[mask3_valid], time_bottom[mask3_valid]):
            f.write(f'{xi:.6f}\t{ti:.6f}\t{tp:.3f}\t{bt:.3f}\n')
    print('Label‑3 詳細を保存:', dump_path)

# ------------------------------------------------------------------
# 6-2. 線形‑線形プロット保存： Group2もサイズ推定した場合
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(unique_sizes_estimate_group2, cum_counts_estimate_group2, marker='o', linestyle='-', linewidth=1.5)
plt.xlabel('Rock size [cm]', fontsize=20)
plt.ylabel('Cumulative number of rocks', fontsize=20)
plt.tick_params(labelsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
linear_png = os.path.join(output_dir_plot, 'RSFD_linear_estimate_group2.png')
linear_pdf = os.path.join(output_dir_plot, 'RSFD_linear_estimate_group2.pdf')
plt.savefig(linear_png, dpi=300)
plt.savefig(linear_pdf, dpi=600)
plt.show()
print('線形‑線形プロット保存:', linear_png)

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
        f.write('#x\t t\t time_top\t time_bottom\n')
        for xi, ti, tp, bt in zip(x[mask2_valid | mask3_valid], t[mask2_valid | mask3_valid], time_top[mask2_valid | mask3_valid], time_bottom[mask2_valid | mask3_valid]):
            f.write(f'{xi:.6f}\t{ti:.6f}\t{tp:.3f}\t{bt:.3f}\n')
    print('Label‑2-3 詳細を保存:', dump_path)

# ------------------------------------------------------------------
# 7. フィッティング: べき則と指数関数
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
    (k_exp_trad, r_exp_trad, R2_exp_trad, N_exp_fit_trad), D_fit\
    = calc_fitting(unique_sizes_traditional, cum_counts_traditional)

(k_pow_est_grp2, r_pow_est_grp2, R2_pow_est_grp2, N_pow_fit_est_grp2),\
    (k_exp_est_grp2, r_exp_est_grp2, R2_exp_est_grp2, N_exp_fit_est_grp2), D_fit\
    = calc_fitting(unique_sizes_estimate_group2, cum_counts_estimate_group2)

# ------------------------------------------------------------------
# 8. プロット: 個別フィット
# ------------------------------------------------------------------
# 8.1 べき則
plt.figure(figsize=(8, 6))
plt.scatter(unique_sizes_traditional, cum_counts_traditional, marker='o', label='Data')
# エラーバーつきのプロット↓
# plt.errorbar(unique_sizes[0], cum_counts_traditional[0], xerr=[[0], [5]], fmt='o', color='black', capsize=5) # Group1
# plt.errorbar(unique_sizes[1], cum_counts_traditional[1], xerr=[[0], [1]], fmt='o', color='black', capsize=5) # Group2
# for i in range(2, len(unique_sizes)):
#     plt.errorbar(unique_sizes[i], cum_counts_traditional[i], xerr=[[unique_sizes[i] -sizes_group3_min[i-2]], [sizes_group3_max[i-2] - unique_sizes[i]]], fmt='o', color='black', capsize=5) # Group3

plt.plot(D_fit, N_pow_fit_trad, linestyle='--', linewidth=1.5, color='red',
            label=f'Power-law: k={k_pow_trad:.2e}, r={r_pow_trad:.3f}, R²={R2_pow_trad:.4f}')
plt.xlabel('Rock size [cm]', fontsize=20)
plt.ylabel('Cumulative number of rocks', fontsize=20)
plt.tick_params(labelsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=14)
plt.tight_layout()
pow_png = os.path.join(output_dir_plot, 'RSFD_power_law_fit.png')
plt.savefig(pow_png, dpi=300)
plt.savefig(os.path.join(output_dir_plot, 'RSFD_power_law_fit.pdf'), dpi=600)
plt.show()
print('べき則フィッティングプロット保存:', pow_png)

# 8.2 指数関数
plt.figure(figsize=(8, 6))
plt.scatter(unique_sizes_traditional, cum_counts_traditional, marker='o', label='Data')
plt.plot(D_fit, N_exp_fit_trad, linestyle='--', linewidth=1.5, color='green',
            label=f'Exponential: k={k_exp_trad:.2e}, r={r_exp_trad:.3f}, R²={R2_exp_trad:.4f}')
plt.xlabel('Rock size [cm]', fontsize=20)
plt.ylabel('Cumulative number of rocks', fontsize=20)
plt.tick_params(labelsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=14)
plt.tight_layout()
exp_png = os.path.join(output_dir_plot, 'RSFD_exponential_fit.png')
plt.savefig(exp_png, dpi=300)
plt.savefig(os.path.join(output_dir_plot, 'RSFD_exponential_fit.pdf'), dpi=600)
plt.show()
print('指数関数フィッティングプロット保存:', exp_png)

# ------------------------------------------------------------------
# 9. プロット: フィッティング比較 (滑らか)
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(unique_sizes_traditional, cum_counts_traditional, marker='o', label='Data')
plt.plot(D_fit, N_pow_fit_trad, linestyle='--', linewidth=1.5, color='red',
            label=f'Power-law: k={k_pow_trad:.2e}, r={r_pow_trad:.3f}, R²={R2_pow_trad:.4f}')
plt.plot(D_fit, N_exp_fit_trad, linestyle='--', linewidth=1.5, color='green',
            label=f'Exponential: k={k_exp_trad:.2e}, r={r_exp_trad:.3f}, R²={R2_exp_trad:.4f}')
plt.xlabel('Rock size [cm]', fontsize=20)
plt.ylabel('Cumulative number of rocks', fontsize=20)
plt.xscale('log')
plt.yscale('log')
plt.tick_params(labelsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=14)
plt.tight_layout()
comp_png = os.path.join(output_dir_plot, 'RSFD_fit_comparison.png')
plt.savefig(comp_png, dpi=300)
plt.savefig(os.path.join(output_dir_plot, 'RSFD_fit_comparison.pdf'), dpi=600)
plt.show()
print('フィッティング比較プロット保存:', comp_png)

# ------------------------------------------------------------------
# 9-2. プロット: フィッティング比較 (滑らか)： Group2もサイズ推定した場合
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(unique_sizes_estimate_group2, cum_counts_estimate_group2, marker='o', label='Data')
plt.plot(D_fit, N_pow_fit_est_grp2, linestyle='--', linewidth=1.5, color='red',
            label=f'Power-law: k={k_pow_est_grp2:.2e}, r={r_pow_est_grp2:.3f}, R²={R2_pow_est_grp2:.4f}')
plt.plot(D_fit, N_exp_fit_est_grp2, linestyle='--', linewidth=1.5, color='green',
            label=f'Exponential: k={k_exp_est_grp2:.2e}, r={r_exp_est_grp2:.3f}, R²={R2_exp_est_grp2:.4f}')
plt.xlabel('Rock size [cm]', fontsize=20)
plt.ylabel('Cumulative number of rocks', fontsize=20)
plt.xscale('log')
plt.yscale('log')
plt.tick_params(labelsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=14)
plt.tight_layout()
comp_png = os.path.join(output_dir_plot, 'RSFD_fit_comparison_estimate_group2.png')
plt.savefig(comp_png, dpi=300)
plt.savefig(os.path.join(output_dir_plot, 'RSFD_fit_comparison_estimate_group2.pdf'), dpi=600)
plt.show()
print('フィッティング比較プロット保存:', comp_png)

print('すべて完了しました！')