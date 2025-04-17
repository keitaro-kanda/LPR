#!/usr/bin/env python3
# RSFD_generator.py
# ------------------------------------------------------------
# ラベル JSON から
#   1) ラベル1→1 cm, ラベル2→6 cm, ラベル3→式で計算
# の岩石サイズを取得し，
# 線形‑線形の累積サイズ‑頻度分布 (個数) を描画・保存する
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
# ラベル1：1 cm
size_label1 = np.full(counts[1], 1.0)                          # [cm]

# ラベル2：6 cm
size_label2 = np.full(counts[2], 6.0)                          # [cm]

# ラベル3：time 差から計算　　　　　　　　　　　　　　　   　　　# <<< ADDED
mask3_valid = (lab == 3) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
er = 9.0
c  = 299_792_458  # m/s
sizes_group3 = (time_bottom[mask3_valid] - time_top[mask3_valid]) \
               * 1e-9 * c / np.sqrt(er) * 0.5 * 100             # [cm]

# まだ time_top/bottom が NaN のラベル3はサイズ不明 → 無視

# まとめる
all_sizes_cm = np.concatenate([size_label1, size_label2, sizes_group3])
if all_sizes_cm.size == 0:
    raise RuntimeError('有効なラベル1–3が見つかりませんでした。')

# ------------------------------------------------------------------
# 5. 累積サイズ‑頻度分布 (≥ size) を計算　　　　　　　　# <<< ADDED
# ------------------------------------------------------------------
unique_sizes = np.sort(np.unique(all_sizes_cm))                 # 昇順
cum_counts   = np.array([(all_sizes_cm >= s).sum()
                         for s in unique_sizes], dtype=int)

# ------------------------------------------------------------------
# 6. プロット (線形‑線形)　　　　　　　　　　　　　　　# <<< ADDED
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(unique_sizes, cum_counts,
         marker='o', linestyle='-', linewidth=1.5)
plt.xlabel('Rock size [cm]', fontsize=20)
plt.ylabel('Cumulative number of rocks', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 画面表示

# 画像保存
plot_png = os.path.join(output_dir, 'RSFD_linear.png')
plot_pdf = os.path.join(output_dir, 'RSFD_linear.pdf')
plt.savefig(plot_png, format='png', dpi=300)
plt.savefig(plot_pdf, format='pdf', dpi=600)
plt.show()


print('プロット保存:', plot_png, '&', plot_pdf)

# ------------------------------------------------------------------
# 7. サイズと累積個数を TXT 出力　　　　　　　　　　　　# <<< ADDED
# ------------------------------------------------------------------
table_txt = os.path.join(output_dir, 'RSFD_linear.txt')
with open(table_txt, 'w') as f:
    f.write('# size_cm\tcumulative_count\n')
    for s, n in zip(unique_sizes, cum_counts):
        f.write(f'{s:.3f}\t{n}\n')
print('累積データ保存:', table_txt)

print('すべて完了しました！')
