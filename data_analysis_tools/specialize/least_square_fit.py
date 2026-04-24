import numpy as np
import matplotlib.pyplot as plt


# モデルデータの読み取り
model_data_csv_path = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD_relative/r_2.0/1000/aggregated_stats_range.csv'
model_data = np.genfromtxt(model_data_csv_path, delimiter=',', skip_header=1)
r_model_mean = model_data[:, 1]
r_model_std = model_data[:, 2]
k_model_mean = model_data[:, 3]
k_model_std = model_data[:, 4]

# kをe-3で規格化
k_model_mean /= 1e-3
k_model_std /= 1e-3


# 出力ディレクトリの設定
output_dir = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD_relative/Compare_observation'

# モデルデータの保存
model_data_to_save = np.column_stack((r_model_mean, r_model_std, k_model_mean, k_model_std))
np.savetxt(f'{output_dir}/model_data.csv', model_data_to_save, delimiter=',', header='r_mean,r_std,k_mean,k_std', comments='')


def least_square_fit(x, y):
    """
    最小二乗法を用いて線形フィットを行う関数。
    
    Parameters:
    x (array-like): 独立変数のデータポイント。
    y (array-like): 従属変数のデータポイント。
    
    Returns:
    slope (float): フィットした直線の傾き。
    intercept (float): フィットした直線の切片。
    """
    # データポイントの数
    n = len(x)
    
    # xとyの平均を計算
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # 傾きと切片を計算
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    
    return slope, intercept


max_depth = np.arange(1, 13)

# LPR観測値 (Group 1-3)
r_obs_list = [-0.2, 1.04, 1.41, 1.16, 1.23, 1.15, 1.10, 1.12, 1.15, 1.07, 1.08, 1.08]
k_obs_list = [-5, 4.48, 7.05, 9.25, 10.29, 9.48, 8.33, 7.77, 7.58, 6.60, 6.39, 6.85] # e-3で規格化
r_obs_4fit = r_obs_list[2:]
k_obs_4fit = k_obs_list[4:]


# 最小二乗法でフィット
r_obs_fit, r_obs_intercept = least_square_fit(max_depth[2:], r_obs_4fit)
k_obs_fit, k_obs_intercept = least_square_fit(max_depth[4:], k_obs_4fit)

r_model_fit, r_model_intercept = least_square_fit(max_depth, r_model_mean)
k_model_fit, k_model_intercept = least_square_fit(max_depth, k_model_mean)


# 目盛りカスタマイズ用の関数を追加 (-0.2をNaNにし、正の目盛りは維持)
def apply_custom_yticks_r():
    ax = plt.gca()
    yticks = ax.get_yticks()
    new_ticks = [-0.2] + [t for t in yticks if t >= 0]
    new_labels = ['NaN'] + [f"{t:g}" for t in new_ticks[1:]]
    ax.set_yticks(new_ticks)
    ax.set_yticklabels(new_labels, fontsize=14)

def apply_custom_yticks_k():
    ax = plt.gca()
    yticks = ax.get_yticks()
    new_ticks = [-5] + [t for t in yticks if t >= 0]
    new_labels = ['NaN'] + [f"{t:g}" for t in new_ticks[1:]]
    ax.set_yticks(new_ticks)
    ax.set_yticklabels(new_labels, fontsize=14)


# プロット
depth_range_label = ['0-1', '0-2', '0-3', '0-4', '0-5', '0-6', '0-7', '0-8', '0-9', '0-10', '0-11', '0-12']

# ===========================================================
# r_obsとk_model（フィッティング）
# ===========================================================

plt.figure(figsize=(8, 6))

# obs
plt.plot(max_depth, r_obs_list, 'o-', label='Observed r', color='blue')
plt.plot(max_depth[2:], r_obs_fit * max_depth[2:] + r_obs_intercept, color='k', linestyle='--', label=f'Fit for Observed r, slope={r_obs_fit:.3f}')

# model
plt.plot(max_depth, r_model_mean, 's--', label='Model r', color='skyblue')
plt.plot(max_depth, r_model_fit * max_depth + r_model_intercept, color='gray', linestyle='-.', label=f'Fit for Model r, slope={r_model_fit:.3f}')

plt.xlabel('Depth range [m]', fontsize=16)
plt.ylabel('r', fontsize=16)
plt.xticks(max_depth, depth_range_label, rotation=45, fontsize=14)
apply_custom_yticks_r()
plt.legend(fontsize=14, loc = 'lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{output_dir}/r_fit.png', dpi=300)
plt.savefig(f'{output_dir}/r_fit.pdf', dpi=300)
plt.close()


# ===========================================================
# k_obsとk_model（フィッティング）
# ===========================================================

plt.figure(figsize=(8, 6))

# obs
plt.plot(max_depth, k_obs_list, 'o-', label='Observed k', color='red')
plt.plot(max_depth[4:], k_obs_fit * max_depth[4:] + k_obs_intercept, color='k', linestyle='--', label=f'Fit for Observed k, slope={k_obs_fit:.3f}')

# model
plt.plot(max_depth, k_model_mean, 's--', label='Model k', color='salmon')
plt.plot(max_depth, k_model_fit * max_depth + k_model_intercept, color='gray', linestyle='-.', label=f'Fit for Model k, slope={k_model_fit:.3f}')

plt.xlabel('Depth range [m]', fontsize=16)
plt.ylabel(r'k [$\times 10^{-3}$]', fontsize=16)
plt.xticks(max_depth, depth_range_label, rotation=45, fontsize=14)
apply_custom_yticks_k()
plt.legend(fontsize=14, loc = 'upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{output_dir}/k_fit.png', dpi=300)
plt.savefig(f'{output_dir}/k_fit.pdf', dpi=300)
plt.close()


# ===========================================================
# r_obsとr_model（モデルのエラーバー付き）
# ===========================================================
plt.figure(figsize=(8, 6))

# obs
plt.plot(max_depth, r_obs_list, 'o-', label='Observed r', color='blue')
# model
plt.errorbar(max_depth, r_model_mean, yerr=r_model_std, fmt='s--', label='Model r', color='skyblue', capsize=5)

plt.xlabel('Depth range [m]', fontsize=16)
plt.ylabel(r'r', fontsize=16)
plt.xticks(max_depth, depth_range_label, rotation=45, fontsize=14)
apply_custom_yticks_r()
plt.legend(fontsize=14, loc = 'upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{output_dir}/r_error.png', dpi=300)
plt.savefig(f'{output_dir}/r_error.pdf', dpi=300)
plt.close()


# ===========================================================
# k_obsとk_model（モデルのエラーバー付き）
# ===========================================================
plt.figure(figsize=(8, 6))

# obs
plt.plot(max_depth, k_obs_list, 'o-', label='Observed k', color='red')
# model
plt.errorbar(max_depth, k_model_mean, yerr=k_model_std, fmt='s--', label='Model k', color='salmon', capsize=5)

plt.xlabel('Depth range [m]', fontsize=16)
plt.ylabel(r'k [$\times 10^{-3}$]', fontsize=16)
plt.xticks(max_depth, depth_range_label, rotation=45, fontsize=14)
apply_custom_yticks_k()
plt.legend(fontsize=14, loc = 'upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{output_dir}/k_error.png', dpi=300)
plt.savefig(f'{output_dir}/k_error.pdf', dpi=300)
plt.close()