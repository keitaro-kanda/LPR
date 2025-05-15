#!/usr/bin/env python3
# RSFD_from_csv.py
# ------------------------------------------------------------
# CSVファイルから
#   'diameter', 'cumulative number' 列を読み込み，
# 線形‑線形の累積サイズ‑頻度分布 (個数) を描画・保存し，
# べき則／指数関数フィッティングおよび比較プロットを作成
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. 入力ファイルチェック
# ------------------------------------------------------------------
print('RSFDプロット用CSVファイルのパスを入力してください:')
data_path = input().strip()
if not (os.path.exists(data_path) and data_path.lower().endswith('.csv')):
    raise FileNotFoundError('正しい .csv ファイルを指定してください。')

# 出力フォルダ設定
base_dir = os.path.dirname(data_path)
output_dir = os.path.join(base_dir, 'RSFD_csv')
os.makedirs(output_dir, exist_ok=True)
plot_dir = os.path.join(output_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

# ------------------------------------------------------------------
# 2. CSV 読み込みと前処理
# ------------------------------------------------------------------
data = pd.read_csv(data_path)
# 必要な列があるか確認
required_cols = ['diameter', 'cumulative number']
for col in required_cols:
    if col not in data.columns:
        raise KeyError(f"CSVに必要な列 '{col}' がありません。")

# 数値型に変換
diam = pd.to_numeric(data['diameter'], errors='coerce').values
cum_n = pd.to_numeric(data['cumulative number'], errors='coerce').values

# ソート
order = np.argsort(diam)
diam = diam[order]
cum_n = cum_n[order]

# 有効データのフィルタ: diam > 0, cum_n > 0, 両方とも非NaN
valid = (~np.isnan(diam)) & (~np.isnan(cum_n)) & (diam > 0) & (cum_n > 0)
if not np.any(valid):
    raise RuntimeError('有効なデータポイントがありません。diameter>0、cumulative number>0 の行を確認してください。')

D = diam[valid]
N = cum_n[valid]

# ------------------------------------------------------------------
# 3. 線形‑線形プロット
# ------------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(D, N, marker='o', linestyle='-')
plt.xlabel('Diameter', fontsize=16)
plt.ylabel('Cumulative number', fontsize=16)
plt.tick_params(labelsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
linear_png = os.path.join(plot_dir, 'RSFD_csv_linear.png')
linear_pdf = os.path.join(plot_dir, 'RSFD_csv_linear.pdf')
plt.savefig(linear_png, dpi=300)
plt.savefig(linear_pdf, dpi=600)
plt.show()
print('線形‑線形プロット保存:', linear_png)

# 元データテーブルも保存（オプション）
csv_out = os.path.join(output_dir, 'RSFD_csv_filtered.csv')
pd.DataFrame({'diameter': D, 'cumulative number': N}).to_csv(csv_out, index=False)
print('フィルタ後データ保存:', csv_out)

# ------------------------------------------------------------------
# 4. フィッティング準備
# ------------------------------------------------------------------
log_D = np.log(D)
log_N = np.log(N)
ss_tot = np.sum((N - np.mean(N))**2)

# べき則: log-log 回帰
r_pow, log_k_pow = np.polyfit(log_D, log_N, 1)
k_pow = np.exp(log_k_pow)
# フィット曲線
D_fit = np.linspace(D.min(), D.max(), 1000)
N_pow_fit = k_pow * D_fit**r_pow
# 決定係数 R^2
ss_res_pow = np.sum((N - (k_pow * D**r_pow))**2)
R2_pow = 1 - ss_res_pow/ss_tot

# 指数関数: semilog-y 回帰
slope_exp, log_k_exp = np.polyfit(D, log_N, 1)
r_exp = slope_exp
k_exp = np.exp(log_k_exp)
N_exp_fit = k_exp * np.exp(r_exp * D_fit)
ss_res_exp = np.sum((N - (k_exp * np.exp(r_exp * D)))**2)
R2_exp = 1 - ss_res_exp/ss_tot

# ------------------------------------------------------------------
# 5. プロット: べき則フィッティング
# ------------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(D, N, marker='o', label='Data')
plt.plot(D_fit, N_pow_fit, linestyle='--', linewidth=1.5, color='red',
         label=f'Power-law: k={k_pow:.2e}, r={r_pow:.3f}, R²={R2_pow:.4f}')
plt.xlabel('Diameter', fontsize=16)
plt.ylabel('Cumulative number', fontsize=16)
plt.tick_params(labelsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
pow_png = os.path.join(plot_dir, 'RSFD_csv_pow_fit.png')
plt.savefig(pow_png, dpi=300)
plt.savefig(os.path.join(plot_dir, 'RSFD_csv_pow_fit.pdf'), dpi=600)
plt.show()
print('べき則フィットプロット保存:', pow_png)

# ------------------------------------------------------------------
# 6. プロット: 指数関数フィッティング
# ------------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(D, N, marker='o', label='Data')
plt.plot(D_fit, N_exp_fit, linestyle='--', linewidth=1.5, color='green',
         label=f'Exponential: k={k_exp:.2e}, r={r_exp:.3f}, R²={R2_exp:.4f}')
plt.xlabel('Diameter', fontsize=16)
plt.ylabel('Cumulative number', fontsize=16)
plt.tick_params(labelsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
exp_png = os.path.join(plot_dir, 'RSFD_csv_exp_fit.png')
plt.savefig(exp_png, dpi=300)
plt.savefig(os.path.join(plot_dir, 'RSFD_csv_exp_fit.pdf'), dpi=600)
plt.show()
print('指数関数フィットプロット保存:', exp_png)

# ------------------------------------------------------------------
# 7. プロット: フィッティング比較
# ------------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(D, N, marker='o', label='Data')
plt.plot(D_fit, N_pow_fit, linestyle='--', linewidth=1.5, color='red', label='Power-law fit')
plt.plot(D_fit, N_exp_fit, linestyle='--', linewidth=1.5, color='green', label='Exponential fit')
plt.xlabel('Diameter', fontsize=16)
plt.ylabel('Cumulative number', fontsize=16)
#plt.xscale('log')
#plt.yscale('log')
plt.tick_params(labelsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
comp_png = os.path.join(plot_dir, 'RSFD_csv_comparison.png')
plt.savefig(comp_png, dpi=300)
plt.savefig(os.path.join(plot_dir, 'RSFD_csv_comparison.pdf'), dpi=600)
plt.show()
print('比較プロット保存:', comp_png)

print('すべて完了しました！')
