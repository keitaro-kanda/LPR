"""
RSFDのパラメータ比較用モジュール
jsonファイルのパスを入力し、深さごとのRSFDパラメータをプロットする。
    jsonファイルテンプレート：
    {
        "0-2 m": {
        "label": "0-2",
        "Depth": 2.0,
        "Number of rocks": 13,
        "r": 1.04,
        "k": 4.5e-3,
        "p": 0.020
        },
        "2-4 m": {
            "label": "2-4",
            "Depth": 4.0,
            "Number of rocks": 37,
            "r": 1.09,
            "k": 12.3e-3,
            "p": 0.020
        },
        ...
    }
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json


# Input
json_path = input("Enter the path to the RSFD parameters JSON file: ").strip()
if not os.path.isfile(json_path):
    raise FileNotFoundError(f"File not found: {json_path}")

# Output directory
base_dir = os.path.dirname(json_path)
output_dir = os.path.join(base_dir, 'params_comparison_plots')
os.makedirs(output_dir, exist_ok=True)


# Load JSON data
with open(json_path, 'r') as f:
    rsfd_data = json.load(f)


# Extract data for plotting
labels = []
depths = []
rock_nums = []
r_values = []
k_values = []
p_values = []
for key, value in rsfd_data.items():
    labels.append(value['label'])
    depths.append(value['Depth'])
    rock_nums.append(value['Number of rocks'])
    r_values.append(value['r'])
    k_values.append(value['k'])
    p_values.append(value['p'])
# Convert to numpy arrays for easier handling
depths = np.array(depths)
rock_nums = np.array(rock_nums)
r_values = np.array(r_values)
k_values = np.array(k_values)
p_values = np.array(p_values)

# convert str in p_values array
# "<0.001" -> 0.0005
# "nan" - > 1.0
for i in range(len(p_values)):
    if isinstance(p_values[i], str):
        if p_values[i].startswith('<'):
            p_values[i] = float(p_values[i][1:]) / 2
        elif p_values[i].lower() == 'nan':
            p_values[i] = 1.0
p_values = p_values.astype(float)

# Plotting
# depth VS number of rocks plot
plt.figure(figsize=(8, 6))
plt.plot(depths, rock_nums, marker='o', linestyle='-', color='black')

plt.xlabel('Depth [m]', fontsize=18)
plt.xticks(depths, labels)
plt.ylabel('Number of detected rocks', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, ls='--', alpha=0.5, axis='both')

plt.savefig(os.path.join(output_dir, 'depth_vs_rock_num.png'), dpi=120)
plt.savefig(os.path.join(output_dir, 'depth_vs_rock_num.pdf'), dpi=600)
plt.show()

# depth VS r plot
plt.figure(figsize=(8, 6))
plt.plot(depths, r_values, marker='o', linestyle='-', color='blue')

plt.xlabel('Depth [m]', fontsize=18)
plt.xticks(depths, labels)
plt.ylabel('RSFD Slope r', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.ylim(0, max(r_values) + 0.5)
plt.grid(True, ls='--', alpha=0.5, axis='both')

plt.savefig(os.path.join(output_dir, 'depth_vs_r.png'), dpi=120)
plt.savefig(os.path.join(output_dir, 'depth_vs_r.pdf'), dpi=600)
plt.show()

# depth VS k plot
plt.figure(figsize=(8, 6))
plt.plot(depths, k_values, marker='o', linestyle='-', color='red')

plt.xlabel('Depth [m]', fontsize=18)
plt.xticks(depths, labels)
plt.ylabel('RSFD Coefficient k', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.yscale('log')
plt.ylim(1e-3, max(k_values)*3)
plt.grid(True, ls='--', alpha=0.5, axis='both')

plt.savefig(os.path.join(output_dir, 'depth_vs_k.png'), dpi=120)
plt.savefig(os.path.join(output_dir, 'depth_vs_k.pdf'), dpi=600)
plt.show()

# depth VS p plot
plt.figure(figsize=(8, 6))
plt.plot(depths, p_values, marker='o', linestyle='-', color='green')
plt.axhline(y=0.05, color='gray', linestyle='--')

plt.xlabel('Depth [m]', fontsize=18)
plt.xticks(depths, labels)
plt.ylabel('p-value', fontsize=18)
plt.yscale('log')
plt.yticks([0.0005, 0.001, 0.01, 0.1, 1], ['<0.001', '0.001', '0.01', '0.1', 'nan'])
plt.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, ls='--', alpha=0.5, axis='both')

plt.savefig(os.path.join(output_dir, 'depth_vs_p.png'), dpi=120)
plt.savefig(os.path.join(output_dir, 'depth_vs_p.pdf'), dpi=600)
plt.show()