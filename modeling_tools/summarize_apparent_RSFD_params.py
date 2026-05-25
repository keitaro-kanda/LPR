import json
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DIR = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD_relative'

# LPR観測値
LPR_obs = [
    {'r': 1.08, 'k': 7.21e-3}, # full range
    {'r': 1.92, 'k':19.3e-3},  # 0-3 m
    {'r': 1.00, 'k':5.68e-3}   # 0-3 m
]

# --- JSONファイルの自動検索とユーザー選択 ---
json_files = sorted(glob.glob(os.path.join(BASE_DIR, 'rsfd_summary*.json')))
if not json_files:
    print(f"エラー: {BASE_DIR} 内に rsfd_summary*.json が見つかりません。")
    exit(1)

print("使用するJSONファイルを選択してください:")
for i, path in enumerate(json_files):
    print(f"  {i}: {os.path.basename(path)}")

try:
    choice = int(input("番号を入力してください: ").strip())
    if choice < 0 or choice >= len(json_files):
        raise ValueError
except ValueError:
    print("無効な入力です。終了します。")
    exit(1)

json_file_path = json_files[choice]
basename = os.path.basename(json_file_path)          # e.g. rsfd_summary_0.0-3.0m.json
stem = os.path.splitext(basename)[0]                 # e.g. rsfd_summary_0.0-3.0m
suffix = stem[len('rsfd_summary'):]                  # e.g. _0.0-3.0m  or  ""

if suffix == '':
    plot_dir_name = 'Summary_Plots_all_range'
else:
    plot_dir_name = f"Summary_Plots{suffix}"         # e.g. Summary_Plots_0.0-3.0m

output_dir = os.path.join(BASE_DIR, plot_dir_name)
print(f"\n選択ファイル : {json_file_path}")
print(f"出力ディレクトリ: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 横軸となる r の値を取得し、数値として昇順にソートする
r_keys = sorted(data.keys(), key=float)

# 岩石数のリストを取得し、数値として昇順にソートする
# 全 r の N キーを横断して収集（深さ範囲JSONでは r によって存在する N が異なるため）
all_n_keys = set()
for r in r_keys:
    for n_key in data[r].keys():
        try:
            int(n_key)
            all_n_keys.add(n_key)
        except ValueError:
            pass
rock_counts = sorted(all_n_keys, key=int)

# Calculation stop の有無を確認
calculation_stops = []
for r in r_keys:
    for count in rock_counts:
        if count in data[r] and data[r][count] == "Calculation stop":
            calculation_stops.append((r, count))
print(f"Calculation stops found for: {calculation_stops}")

# プロット線用のリスト
line_colors  = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange'] # r = 1.0-4.0 に対応（0.5刻み）
line_markers = ['o', 's', 'D', '^'] # N = 100, 500, 1000, 5000 に対応
line_styles  = ['-', '--', '-.', ':']

# =========================================================
# 1. Input r vs Apparent r
# =========================================================
plt.figure(figsize=(8, 6))

for count in rock_counts:
    x_vals = []
    y_means = []
    y_stds = []

    for r in r_keys:
        if count in data[r] and (r, count) not in calculation_stops:
            x_vals.append(float(r))
            y_means.append(data[r][count]["r_apparent_mean"])
            y_stds.append(data[r][count]["r_apparent_std"])

    plt.errorbar(
        x_vals, y_means, yerr=y_stds,
        label=f'N = {count}',
        marker=line_markers[rock_counts.index(count) % len(line_markers)],
        capsize=5,
        linestyle=line_styles[rock_counts.index(count) % len(line_styles)],
        color=line_colors[rock_counts.index(count) % len(line_colors)]
    )

plt.plot([0, 4.5], [0, 4.5], 'k--', label='Input = Apparent')

plt.xlabel('Input r', fontsize=16)
plt.ylabel('Apparent r', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'r_apparent_plot.png'))
plt.savefig(os.path.join(output_dir, 'r_apparent_plot.pdf'))
plt.show()

# =========================================================
# 2. Input r vs Apparent r / r true
# =========================================================
plt.figure(figsize=(8, 6))

for count in rock_counts:
    x_vals = []
    y_means = []
    y_stds = []

    for r in r_keys:
        if count in data[r] and (r, count) not in calculation_stops:
            x_vals.append(float(r))
            r_true = data[r][count]["r_true_mean"]
            y_means.append(data[r][count]["r_apparent_mean"] / r_true)
            y_stds.append(data[r][count]["r_apparent_std"] / r_true)

    plt.errorbar(
        x_vals, y_means, yerr=y_stds,
        label=f'N = {count}',
        marker=line_markers[rock_counts.index(count) % len(line_markers)],
        capsize=5,
        linestyle=line_styles[rock_counts.index(count) % len(line_styles)],
        color=line_colors[rock_counts.index(count) % len(line_colors)]
    )

plt.axhline(1, color='k', linestyle='--', label='Apparent / True = 1')

plt.xlabel('Input r', fontsize=16)
plt.ylabel('Apparent r / r true', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'r_apparent_ratio_plot.png'))
plt.savefig(os.path.join(output_dir, 'r_apparent_ratio_plot.pdf'))
plt.show()

# =========================================================
# 3. Input r vs Apparent k / k true
# =========================================================
plt.figure(figsize=(8, 6))

for count in rock_counts:
    x_vals = []
    y_means = []
    y_stds = []

    for r in r_keys:
        if count in data[r] and (r, count) not in calculation_stops:
            x_vals.append(float(r))
            k_true = data[r][count]["k_true_mean"]
            y_means.append(data[r][count]["k_apparent_mean"] / k_true)
            y_stds.append(data[r][count]["k_apparent_std"] / k_true)

    plt.errorbar(
        x_vals, y_means, yerr=y_stds,
        label=f'N = {count}',
        marker=line_markers[rock_counts.index(count) % len(line_markers)],
        capsize=5,
        linestyle=line_styles[rock_counts.index(count) % len(line_styles)],
        color=line_colors[rock_counts.index(count) % len(line_colors)]
    )

plt.axhline(1, color='k', linestyle='--', label='Apparent / True = 1')

plt.xlabel('Input r', fontsize=16)
plt.ylabel('Apparent k / k true', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'k_apparent_ratio_plot.png'))
plt.savefig(os.path.join(output_dir, 'k_apparent_ratio_plot.pdf'))
plt.show()

# =========================================================
# 4. k true vs k apparent
# =========================================================
plt.figure(figsize=(8, 6))

for count in rock_counts:
    x_vals = []
    y_means = []
    y_stds = []

    for r in r_keys:
        if count in data[r] and (r, count) not in calculation_stops:
            x_vals.append(data[r][count]["k_true_mean"])
            y_means.append(data[r][count]["k_apparent_mean"])
            y_stds.append(data[r][count]["k_apparent_std"])

    plt.errorbar(
        x_vals, y_means, yerr=y_stds,
        label=f'N = {count}',
        marker=line_markers[rock_counts.index(count) % len(line_markers)],
        capsize=5,
        linestyle=line_styles[rock_counts.index(count) % len(line_styles)],
        color=line_colors[rock_counts.index(count) % len(line_colors)]
    )

plt.xlabel('Input k', fontsize=16)
plt.ylabel('Apparent k', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'k_apparent_plot.png'))
plt.savefig(os.path.join(output_dir, 'k_apparent_plot.pdf'))
plt.show()

# =========================================================
# 5. apparent r vs apparent k（LPR観測値ごとにプロット）
#    色: input_r ごと、マーカー: N ごと、線スタイル: 共通
# =========================================================
# 色・マーカーのマッピング
r_colors  = {r: line_colors[idx % len(line_colors)]   for idx, r     in enumerate(r_keys)}
n_markers = {c: line_markers[idx % len(line_markers)] for idx, c     in enumerate(rock_counts)}

num_obs = len(LPR_obs)

for i in range(num_obs):
    r_obs = LPR_obs[i]['r']
    k_obs = LPR_obs[i]['k']

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in r_keys:
        color = r_colors[r]

        # 同じ input_r の点を共通線スタイルで接続
        x_line = [data[r][c]["r_apparent_mean"] for c in rock_counts
                  if c in data[r] and (r, c) not in calculation_stops]
        y_line = [data[r][c]["k_apparent_mean"] for c in rock_counts
                  if c in data[r] and (r, c) not in calculation_stops]
        if x_line:
            ax.plot(x_line, y_line, color=color, linestyle='-', linewidth=1.5, zorder=1)

        # 各点をマーカー＋エラーバーでプロット
        for count in rock_counts:
            if count in data[r] and (r, count) not in calculation_stops:
                ax.errorbar(
                    data[r][count]["r_apparent_mean"],
                    data[r][count]["k_apparent_mean"],
                    xerr=data[r][count]["r_apparent_std"],
                    yerr=data[r][count]["k_apparent_std"],
                    marker=n_markers[count],
                    markersize=8,
                    color=color,
                    capsize=5,
                    linestyle='none',
                    zorder=2
                )

    ax.axhline(k_obs, color='k', linestyle='--', linewidth=2.5)
    ax.axvline(r_obs, color='k', linestyle='--', linewidth=2.5)

    # カスタム凡例: 色→input_r、マーカー→N
    legend_r   = [Line2D([0], [0], color=r_colors[r], linestyle='-', linewidth=1.5,
                          label=f'Input r = {float(r):.1f}') for r in r_keys]
    legend_n   = [Line2D([0], [0], color='gray', marker=n_markers[c], linestyle='none',
                          markersize=8, label=f'N = {c}') for c in rock_counts]
    legend_obs = [Line2D([0], [0], color='k', linestyle='--', linewidth=2.5, label='Observed')]
    ax.legend(handles=legend_r + legend_n + legend_obs, fontsize=14,
              bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

    ax.set_xlabel('Apparent r', fontsize=16)
    ax.set_ylabel('Apparent k', fontsize=16)
    ax.set_xlim(0.2, 4.0)
    ax.set_ylim(3e-4, 0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'apparent_r_vs_k_{i}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'apparent_r_vs_k_{i}.pdf'), bbox_inches='tight')
    plt.show()
