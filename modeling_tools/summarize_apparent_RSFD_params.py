import json
import matplotlib.pyplot as plt
import os

# JSONファイルの読み込み
# 実際のファイル名に合わせて変更してください
json_file_path = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD_relative/rsfd_summary.json'

base_dir = os.path.dirname(json_file_path)
output_dir = os.path.join(base_dir, 'Summary_Plots')
print(f"Output directory: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# LPR観測値
r_obs = 1.08
k_obs = 7.21e-3

with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 横軸となる r の値を取得し、数値として昇順にソートする
r_keys = sorted(data.keys(), key=float)

# 岩石数のリストを取得し、数値として昇順にソートする
# (最初の r のデータから岩石数のキーを取得)
rock_counts = sorted(data[r_keys[0]].keys(), key=int)

# Calculation stop の有無を確認
# Calculation stopがTrueの場合、プロットには含めないようにするためのフラグを設定
calculation_stops = []
for r in r_keys:
    for count in rock_counts:
        if count in data[r] and "calculation_stop" in data[r][count] and data[r][count]["calculation_stop"]:
            calculation_stops.append((r, count))

# プロット線用のリスト
line_colors = ['r', 'g', 'b', 'magenta']
line_markers = ['o', 's', 'D', '^']
line_styles = ['-', '--', '-.', ':']

# =========================================================
# 1. 既存のプロット: Input r vs Apparent r
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
        x_vals, 
        y_means, 
        yerr=y_stds, 
        label=f'N = {count}', 
        marker=line_markers[rock_counts.index(count) % len(line_markers)],
        capsize=5,
        linestyle=line_styles[rock_counts.index(count) % len(line_styles)],
        color=line_colors[rock_counts.index(count) % len(line_colors)]
    )

# input = apparent の線を追加
plt.plot([0, 4.5], [0, 4.5], 'k--', label='Input = Apparent')  # 黒の破線でプロット

# グラフの装飾
plt.xlabel('Input r', fontsize=16)
plt.ylabel('Apparent r', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# グラフの表示と保存
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'r_apparent_plot.png'))
plt.savefig(os.path.join(output_dir, 'r_apparent_plot.pdf'))
plt.show()

# =========================================================
# 2. 追加プロット: Input r vs Apparent r / r true
# =========================================================
plt.figure(figsize=(8, 6))

for count in rock_counts:
    x_vals = []
    y_means = []
    y_stds = []
    
    for r in r_keys:
        if count in data[r] and (r, count) not in calculation_stops:
            x_vals.append(float(r))
            # 値と誤差を r_true_mean で割る
            r_true = data[r][count]["r_true_mean"]
            y_means.append(data[r][count]["r_apparent_mean"] / r_true)
            y_stds.append(data[r][count]["r_apparent_std"] / r_true)
            
    plt.errorbar(
        x_vals, 
        y_means, 
        yerr=y_stds, 
        label=f'N = {count}', 
        marker=line_markers[rock_counts.index(count) % len(line_markers)],
        capsize=5,
        linestyle=line_styles[rock_counts.index(count) % len(line_styles)],
        color=line_colors[rock_counts.index(count) % len(line_colors)]
    )

plt.axhline(1, color='k', linestyle='--', label='Apparent / True = 1')  # y=1の水平線を追加

# グラフの装飾
plt.xlabel('Input r', fontsize=16)
plt.ylabel('Apparent r / r true', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# グラフの表示と保存
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'r_apparent_ratio_plot.png'))
plt.savefig(os.path.join(output_dir, 'r_apparent_ratio_plot.pdf'))
plt.show()

# =========================================================
# 3. 追加プロット: Input r vs Apparent k / k true
# =========================================================
plt.figure(figsize=(8, 6))

for count in rock_counts:
    x_vals = []
    y_means = []
    y_stds = []
    
    for r in r_keys:
        if count in data[r] and (r, count) not in calculation_stops:
            x_vals.append(float(r))
            # 値と誤差を k_true_mean で割る
            k_true = data[r][count]["k_true_mean"]
            y_means.append(data[r][count]["k_apparent_mean"] / k_true)
            y_stds.append(data[r][count]["k_apparent_std"] / k_true)
            
    plt.errorbar(
        x_vals, 
        y_means, 
        yerr=y_stds, 
        label=f'N = {count}', 
        marker=line_markers[rock_counts.index(count) % len(line_markers)],
        capsize=5,
        linestyle=line_styles[rock_counts.index(count) % len(line_styles)],
        color=line_colors[rock_counts.index(count) % len(line_colors)]
    )

plt.axhline(1, color='k', linestyle='--', label='Apparent / True = 1')  # y=1の水平線を追加

# グラフの装飾
plt.xlabel('Input r', fontsize=16)
plt.ylabel('Apparent k / k true', fontsize=16)
# plt.ylim(0, 15)
# plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# グラフの表示と保存
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'k_apparent_ratio_plot.png'))
plt.savefig(os.path.join(output_dir, 'k_apparent_ratio_plot.pdf'))
plt.show()

# =========================================================
# 4. 追加プロット: k true vs k apparent
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
        x_vals, 
        y_means, 
        yerr=y_stds, 
        label=f'N = {count}', 
        marker=line_markers[rock_counts.index(count) % len(line_markers)],
        capsize=5,
        linestyle=line_styles[rock_counts.index(count) % len(line_styles)],
        color=line_colors[rock_counts.index(count) % len(line_colors)]
    )

# input = apparent の線を追加
#plt.plot([0, 4.5], [0, 4.5], 'k--', label='Input = Apparent')  # 黒の破線でプロット

# グラフの装飾
plt.xlabel('Input k', fontsize=16)
plt.ylabel('Apparent k', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# グラフの表示と保存
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'k_apparent_plot.png'))
plt.savefig(os.path.join(output_dir, 'k_apparent_plot.pdf'))
plt.show()


# =========================================================
# 5. 追加プロット: apparent r vs apparent k
# =========================================================
plt.figure(figsize=(8, 6))

for count in rock_counts:
    x_means = []
    x_stds = []
    y_means = []
    y_stds = []
    
    for r in r_keys:
        if count in data[r] and (r, count) not in calculation_stops:
            x_means.append(data[r][count]["r_apparent_mean"])
            x_stds.append(data[r][count]["r_apparent_std"])
            y_means.append(data[r][count]["k_apparent_mean"])
            y_stds.append(data[r][count]["k_apparent_std"])
            
    plt.errorbar(
        x_means, 
        y_means, 
        xerr=x_stds,
        yerr=y_stds, 
        label=f'N = {count}', 
        marker=line_markers[rock_counts.index(count) % len(line_markers)],
        capsize=5,
        linestyle=line_styles[rock_counts.index(count) % len(line_styles)],
        color=line_colors[rock_counts.index(count) % len(line_colors)]
    )

plt.axhline(k_obs, color='k', linestyle='--', linewidth=2.5, label='Observed')  # 観測値の水平線を追加
plt.axvline(r_obs, color='k', linestyle='--', linewidth=2.5)  # 観測値の垂直線を追加

# グラフの装飾
plt.xlabel('Apparent r', fontsize=16)
plt.ylabel('Apparent k', fontsize=16)
plt.xlim(0.2, 3.5)
plt.ylim(3e-4, 0.3)
plt.tick_params(axis='both', which='major', labelsize=14)
#plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# グラフの表示と保存
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'apparent_r_vs_k.png'))
plt.savefig(os.path.join(output_dir, 'apparent_r_vs_k.pdf'))
plt.show()