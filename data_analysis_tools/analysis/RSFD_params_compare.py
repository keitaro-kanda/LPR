"""
RSFDのパラメータ比較用モジュール
make_RSFD_plot.pyの結果をまとめて比較プロットを作成する
どの程度需要があるか分からないので、とりあえずハードコーディングで実装
"""

import numpy as np
import matplotlib.pyplot as plt
import os


depths = [2, 4, 6, 8, 10, 12]
r_values = [1.04, 1.16, 1.15, 1.13, 1.07, 1.08]
k_values = [4.48e-3, 9.25e-3, 9.48e-3, 7.81e-3, 6.60e-3, 5.85e-3]

output_dir = '/Volumes/SSD_Kanda_SAMSUNG/CE4_LPR/LPR_2B/Processed_Data/order_0_1_3_4_5/4_Gain_function/RSFD/Evalate_depth_range_to_appanrent_r/RSFD_group1-3_comparison'

# depth VS r plot
plt.figure(figsize=(8, 6))
plt.plot(depths, r_values, marker='o', linestyle='-', color='blue')
plt.xlabel('Maximum depth to analyze [m]', fontsize=18)
plt.ylabel('RSFD Slope r', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.ylim(0, max(r_values) + 0.5)
plt.grid(True, ls='--', alpha=0.5, axis='both')

plt.savefig(os.path.join(output_dir, 'depth_vs_r.png'), dpi=120)
plt.savefig(os.path.join(output_dir, 'depth_vs_r.pdf'), dpi=600)
plt.show()

# depth VS k plot
plt.figure(figsize=(8, 6))
plt.plot(depths, k_values, marker='o', linestyle='-', color='green')
plt.xlabel('Maximum depth to analyze [m]', fontsize=18)
plt.ylabel('RSFD Coefficient k', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.yscale('log')
plt.grid(True, ls='--', alpha=0.5, axis='both')

plt.savefig(os.path.join(output_dir, 'depth_vs_k.png'), dpi=120)
plt.savefig(os.path.join(output_dir, 'depth_vs_k.pdf'), dpi=600)
plt.show()
