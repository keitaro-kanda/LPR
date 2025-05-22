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
output_dir = os.path.join(os.path.dirname(data_path), 'RSFD')
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
mask3_valid = (lab == 3) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
er = 9.0
c  = 299_792_458  # m/s
sizes_group3 = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(er) * 0.5 * 100  # [cm]
sizes_group3_max = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(6) * 0.5 * 100  # [cm] # grpup3のエラー範囲
sizes_group3_min = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(15) * 0.5 * 100  # [cm] # group3のエラー範囲
all_sizes_cm = np.concatenate([size_label1, size_label2, sizes_group3])
#all_sizes_cm = sizes_group3 # Group3のみだとどうなる？の検証
if all_sizes_cm.size == 0:
    raise RuntimeError('有効なラベル1–3が見つかりませんでした。')

# ------------------------------------------------------------------
# 5. 累積サイズ‑頻度分布 (≥ size) を計算
# ------------------------------------------------------------------
unique_sizes = np.sort(np.unique(all_sizes_cm))
cum_counts   = np.array([(all_sizes_cm >= s).sum() for s in unique_sizes], dtype=int)

# ------------------------------------------------------------------
# 6. 線形‑線形プロット保存
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(unique_sizes, cum_counts, marker='o', linestyle='-', linewidth=1.5)
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
    for s, n in zip(unique_sizes, cum_counts):
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
# 7. フィッティング: べき則と指数関数
# ------------------------------------------------------------------
# 対数変換
mask = unique_sizes > 0
log_D = np.log(unique_sizes[mask])
log_N = np.log(cum_counts[mask])

# 7.1 べき則フィッティング
r_pow, log_k_pow = np.polyfit(log_D, log_N, 1)
k_pow = np.exp(log_k_pow)
# R^2 計算用予測値
N_pred_pow = k_pow * unique_sizes**r_pow
ss_res_pow = np.sum((cum_counts - N_pred_pow)**2)
ss_tot     = np.sum((cum_counts - np.mean(cum_counts))**2)
R2_pow     = 1 - ss_res_pow / ss_tot

# 7.2 指数関数フィッティング
slope_exp, log_k_exp = np.polyfit(unique_sizes, log_N, 1)
r_exp = slope_exp
k_exp = np.exp(log_k_exp)
# R^2 計算用予測値
N_pred_exp = k_exp * np.exp(r_exp * unique_sizes)
ss_res_exp = np.sum((cum_counts - N_pred_exp)**2)
R2_exp     = 1 - ss_res_exp / ss_tot

# フィット曲線用に滑らかなサンプル点を生成
D_fit = np.linspace(unique_sizes.min(), unique_sizes.max(), 200)
N_pow_fit = k_pow * D_fit**r_pow
N_exp_fit = k_exp * np.exp(r_exp * D_fit)

# ------------------------------------------------------------------
# 8. プロット: 個別フィット
# ------------------------------------------------------------------
# 8.1 べき則
plt.figure(figsize=(8, 6))
plt.scatter(unique_sizes, cum_counts, marker='o', label='Data')
# エラーバーつきのプロット↓
# plt.errorbar(unique_sizes[0], cum_counts[0], xerr=[[0], [5]], fmt='o', color='black', capsize=5) # Group1
# plt.errorbar(unique_sizes[1], cum_counts[1], xerr=[[0], [1]], fmt='o', color='black', capsize=5) # Group2
# for i in range(2, len(unique_sizes)):
#     plt.errorbar(unique_sizes[i], cum_counts[i], xerr=[[unique_sizes[i] -sizes_group3_min[i-2]], [sizes_group3_max[i-2] - unique_sizes[i]]], fmt='o', color='black', capsize=5) # Group3

plt.plot(D_fit, N_pow_fit, linestyle='--', linewidth=1.5, color='red',
            label=f'Power-law: k={k_pow:.2e}, r={r_pow:.3f}, R²={R2_pow:.4f}')
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
plt.scatter(unique_sizes, cum_counts, marker='o', label='Data')
plt.plot(D_fit, N_exp_fit, linestyle='--', linewidth=1.5, color='green',
            label=f'Exponential: k={k_exp:.2e}, r={r_exp:.3f}, R²={R2_exp:.4f}')
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
plt.scatter(unique_sizes, cum_counts, marker='o', label='Data')
plt.plot(D_fit, N_pow_fit, linestyle='--', linewidth=1.5, color='red',
            label=f'Power-law: k={k_pow:.2e}, r={r_pow:.3f}, R²={R2_pow:.4f}')
plt.plot(D_fit, N_exp_fit, linestyle='--', linewidth=1.5, color='green',
            label=f'Exponential: k={k_exp:.2e}, r={r_exp:.3f}, R²={R2_exp:.4f}')
plt.xlabel('Rock size [cm]', fontsize=20)
plt.ylabel('Cumulative number of rocks', fontsize=20)
plt.tick_params(labelsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=14)
plt.tight_layout()
comp_png = os.path.join(output_dir_plot, 'RSFD_fit_comparison.png')
plt.savefig(comp_png, dpi=300)
plt.savefig(os.path.join(output_dir_plot, 'RSFD_fit_comparison.pdf'), dpi=600)
plt.show()
print('フィッティング比較プロット保存:', comp_png)

print('すべて完了しました！')

