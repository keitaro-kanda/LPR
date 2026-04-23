import json
import matplotlib.pyplot as plt
import os

# JSONファイルの読み込み
# 実際のファイル名に合わせて変更してください
json_file_path = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD_relative/rsfd_summary.json'

output_dir = os.path.dirname(json_file_path)
print(f"Output directory: {output_dir}")

with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 横軸となる r の値を取得し、数値として昇順にソートする
r_keys = sorted(data.keys(), key=float)

# 岩石数のリストを取得し、数値として昇順にソートする
# (最初の r のデータから岩石数のキーを取得)
rock_counts = sorted(data[r_keys[0]].keys(), key=int)

# グラフの描画設定
plt.figure(figsize=(10, 6))

# 岩石数ごとにデータをまとめてプロット
for count in rock_counts:
    x_vals = []
    y_means = []
    y_stds = []
    
    for r in r_keys:
        # 特定の r において、その岩石数のデータが存在するか確認
        if count in data[r]:
            x_vals.append(float(r))
            y_means.append(data[r][count]["r_apparent_mean"])
            y_stds.append(data[r][count]["r_apparent_std"])
            
    # エラーバー付きの折れ線グラフをプロット
    # marker='o' でデータ点に丸印を表示、capsize=5 でエラーバーの端の横線の長さを指定
    plt.errorbar(
        x_vals, 
        y_means, 
        yerr=y_stds, 
        label=f'N = {count}', 
        marker='o', 
        capsize=5,
        linestyle='-'
    )

# グラフの装飾
plt.xlabel('Input r', fontsize=16)
plt.ylabel('Apparent r', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlim(0, 4.5)
plt.ylim(0, 4.5)
plt.legend(title='Rock Count', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# グラフの表示と保存
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'r_apparent_plot.png'))
plt.savefig(os.path.join(output_dir, 'r_apparent_plot.pdf'))
plt.show()