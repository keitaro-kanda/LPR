import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# --- パス設定 ---
base_dir = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD_relative'
json_path = os.path.join(base_dir, 'rsfd_summary.json')
output_base = os.path.join(base_dir, 'Summary_depth_variation')
os.makedirs(output_base, exist_ok=True)

# --- カラー・マーカー設定 (summarize_apparent_RSFD_params.py と統一) ---
line_colors  = ['r', 'g', 'b', 'magenta']
line_markers = ['o', 's', 'D', '^']
line_styles  = ['-', '--', '-.', ':']

# --- JSON 読み込み ---
print(f"JSON 読み込み: {json_path}")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# r の値を昇順にソート
r_keys = sorted(data.keys(), key=float)

# 全 r にわたって有効な N を収集・昇順ソート
all_rock_counts = set()
for r in r_keys:
    for count in data[r]:
        if data[r][count] != "Calculation stop":
            all_rock_counts.add(count)
rock_counts = sorted(all_rock_counts, key=int)

print(f"有効な r の値: {[float(r) for r in r_keys]}")
print(f"有効な N の値: {[int(c) for c in rock_counts]}")


def get_style(idx):
    return (
        line_colors[idx % len(line_colors)],
        line_markers[idx % len(line_markers)],
        line_styles[idx % len(line_styles)],
    )


# --- r ごとにプロットを作成 ---
for r_str in r_keys:
    r_true = float(r_str)
    print(f"\n=== r = {r_true} のプロット作成中 ===")

    output_dir = os.path.join(output_base, f'r_{r_true}')
    os.makedirs(output_dir, exist_ok=True)

    # この r で Calculation stop でない N のみ対象
    valid_counts = [
        c for c in rock_counts
        if c in data[r_str] and data[r_str][c] != "Calculation stop"
    ]

    if not valid_counts:
        print(f"  -> 有効なデータなし。スキップします。")
        continue

    # CSV 読み込み
    range_data  = {}
    moving_data = {}

    for count in valid_counts:
        csv_range  = os.path.join(base_dir, f'r_{r_true}', count, 'aggregated_stats_range.csv')
        csv_moving = os.path.join(base_dir, f'r_{r_true}', count, 'aggregated_stats_moving.csv')

        if os.path.exists(csv_range):
            range_data[count] = pd.read_csv(csv_range)
        else:
            print(f"  [警告] ファイルが見つかりません: {csv_range}")

        if os.path.exists(csv_moving):
            moving_data[count] = pd.read_csv(csv_moving)
        else:
            print(f"  [警告] ファイルが見つかりません: {csv_moving}")

    # x 軸ラベル (cumulative range 共通)
    range_tick_vals = np.arange(2, 13, 2)
    range_tick_labels = [f"0-{int(v)}" for v in range_tick_vals]

    # =========================================================
    # (a) 累積深さ範囲 vs apparent r
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, count in enumerate(valid_counts):
        if count not in range_data:
            continue
        df = range_data[count]
        color, marker, ls = get_style(idx)
        ax.errorbar(
            df['depth_range'], df['r_apparent_mean'],
            yerr=df['r_apparent_std'],
            label=f'N = {count}',
            color=color, marker=marker, linestyle=ls,
            capsize=5, linewidth=1.5, markersize=6,
        )
    ax.axhline(y=r_true, color='k', linestyle='--', linewidth=1.5, label=f'True r = {r_true}')
    ax.set_xticks(range_tick_vals)
    ax.set_xticklabels(range_tick_labels, fontsize=16)
    ax.set_xlabel('Depth Range [m]', fontsize=18)
    ax.set_ylabel('Apparent Slope r', fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    prefix = os.path.join(output_dir, 'range_r_apparent')
    fig.savefig(f'{prefix}.png')
    fig.savefig(f'{prefix}.pdf')
    plt.close(fig)
    print(f"  -> 保存: {prefix}.png/.pdf")

    # =========================================================
    # (b) 累積深さ範囲 vs apparent k
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, count in enumerate(valid_counts):
        if count not in range_data:
            continue
        df = range_data[count]
        color, marker, ls = get_style(idx)
        ax.errorbar(
            df['depth_range'], df['k_apparent_mean'],
            yerr=df['k_apparent_std'],
            label=f'N = {count}',
            color=color, marker=marker, linestyle=ls,
            capsize=5, linewidth=1.5, markersize=6,
        )
    ax.set_xticks(range_tick_vals)
    ax.set_xticklabels(range_tick_labels, fontsize=16)
    ax.set_xlabel('Depth Range [m]', fontsize=18)
    ax.set_ylabel('Apparent Density Coefficient k [1/m²]', fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    prefix = os.path.join(output_dir, 'range_k_apparent')
    fig.savefig(f'{prefix}.png')
    fig.savefig(f'{prefix}.pdf')
    plt.close(fig)
    print(f"  -> 保存: {prefix}.png/.pdf")

    # =========================================================
    # (c) moving window 中心深さ vs apparent r
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, count in enumerate(valid_counts):
        if count not in moving_data:
            continue
        df = moving_data[count]
        color, marker, ls = get_style(idx)
        ax.errorbar(
            df['depth_center'], df['r_apparent_mean'],
            yerr=df['r_apparent_std'],
            label=f'N = {count}',
            color=color, marker=marker, linestyle=ls,
            capsize=5, linewidth=1.5, markersize=6,
        )
    ax.axhline(y=r_true, color='k', linestyle='--', linewidth=1.5, label=f'True r = {r_true}')
    ax.set_xlabel('Depth Center [m]', fontsize=18)
    ax.set_ylabel('Apparent Slope r', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    prefix = os.path.join(output_dir, 'moving_r_apparent')
    fig.savefig(f'{prefix}.png')
    fig.savefig(f'{prefix}.pdf')
    plt.close(fig)
    print(f"  -> 保存: {prefix}.png/.pdf")

    # =========================================================
    # (d) moving window 中心深さ vs apparent k
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, count in enumerate(valid_counts):
        if count not in moving_data:
            continue
        df = moving_data[count]
        color, marker, ls = get_style(idx)
        ax.errorbar(
            df['depth_center'], df['k_apparent_mean'],
            yerr=df['k_apparent_std'],
            label=f'N = {count}',
            color=color, marker=marker, linestyle=ls,
            capsize=5, linewidth=1.5, markersize=6,
        )
    ax.set_xlabel('Depth Center [m]', fontsize=18)
    ax.set_ylabel('Apparent Density Coefficient k [1/m²]', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    prefix = os.path.join(output_dir, 'moving_k_apparent')
    fig.savefig(f'{prefix}.png')
    fig.savefig(f'{prefix}.pdf')
    plt.close(fig)
    print(f"  -> 保存: {prefix}.png/.pdf")

# =========================================================
# 観測データの読み込み
# =========================================================

_C0  = 3e8
_EPS = 3.0
_v   = _C0 / np.sqrt(_EPS)  # 媒質中の光速 [m/s]

obs_moving_path = (
    '/Volumes/SSD_Kanda_SAMSUNG/CE4_LPR/LPR_2B/Processed_Data/'
    'order_0_1_3_4_5/4_Gain_function/RSFD_moving_window_comparison/'
    'vertical_window23.1ns_group1_1cm/vertical_moving_window_rsfd_statistics.txt'
)
obs_range_path = (
    '/Volumes/SSD_Kanda_SAMSUNG/CE4_LPR/LPR_2B/Processed_Data/'
    'order_0_1_3_4_5/4_Gain_function/RSFD_compare_multiple_region/'
    'compare_depth_range/multi_range_comparison_summary.txt'
)

# --- Moving window 観測データ ---
obs_moving_df = pd.DataFrame()
if os.path.exists(obs_moving_path):
    rows = []
    with open(obs_moving_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            # Format: Window Center_ns start - end k r R2 p num_rocks area
            if len(parts) >= 11 and parts[0].isdigit():
                center_ns = float(parts[1])
                depth_m   = center_ns * 1e-9 * _v / 2
                rows.append({
                    'depth_center': depth_m,
                    'k': float(parts[5]),
                    'r': float(parts[6]),
                })
    obs_moving_df = pd.DataFrame(rows).dropna()
    print(f"Moving 観測データ読み込み完了: {len(obs_moving_df)} 行")
else:
    print(f"[警告] Moving 観測ファイルが見つかりません: {obs_moving_path}")

# --- Range 観測データ ---
obs_range_df = pd.DataFrame()
if os.path.exists(obs_range_path):
    rows = []
    current_depth = None
    k_val = r_val = None
    in_group13 = False

    with open(obs_range_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m = re.match(r'# =+ Range \d+: 0-(\d+(?:\.\d+)?) m =+', line)
            if m:
                current_depth = float(m.group(1))
                k_val = r_val = None
                in_group13 = False
            elif '# [Group1-3 Fitting' in line:
                in_group13 = True
            elif '# [Group2-3 Fitting' in line:
                in_group13 = False
            elif in_group13 and line.startswith('# k:'):
                val = line.split(':', 1)[1].strip()
                k_val = np.nan if val == 'nan' else float(val)
            elif in_group13 and line.startswith('# r:'):
                val = line.split(':', 1)[1].strip()
                r_val = np.nan if val == 'nan' else float(val)
                if current_depth is not None:
                    rows.append({'depth_range': current_depth, 'k': k_val, 'r': r_val})
                    in_group13 = False

    obs_range_df = pd.DataFrame(rows).dropna()
    print(f"Range 観測データ読み込み完了: {len(obs_range_df)} 行")
else:
    print(f"[警告] Range 観測ファイルが見つかりません: {obs_range_path}")

# =========================================================
# 追加機能: N ごとに異なる入力 r の規格化深さ変化プロット
# =========================================================

# r 値ごとの色（7種類）
r_line_colors = ['red', 'green', 'blue', 'magenta', 'orange', 'cyan', 'brown']

comp_range_dir  = os.path.join(output_base, 'range_comparison')
comp_moving_dir = os.path.join(output_base, 'moving_comparison')
os.makedirs(comp_range_dir,  exist_ok=True)
os.makedirs(comp_moving_dir, exist_ok=True)

print("\n=== N ごとの規格化深さ変化プロット作成中 ===")

for count in rock_counts:
    print(f"\n--- N = {count} ---")

    # この N で有効な r のみ収集（r の昇順）
    valid_r_keys = [
        r for r in r_keys
        if count in data[r] and data[r][count] != "Calculation stop"
    ]
    if not valid_r_keys:
        print(f"  -> 有効な r なし。スキップします。")
        continue

    # 各 r の CSV を読み込み
    range_dfs  = {}  # r_str -> DataFrame
    moving_dfs = {}  # r_str -> DataFrame

    for r_str in valid_r_keys:
        r_val = float(r_str)
        csv_range  = os.path.join(base_dir, f'r_{r_val}', count, 'aggregated_stats_range.csv')
        csv_moving = os.path.join(base_dir, f'r_{r_val}', count, 'aggregated_stats_moving.csv')

        if os.path.exists(csv_range):
            range_dfs[r_str] = pd.read_csv(csv_range)
        else:
            print(f"  [警告] 見つかりません: {csv_range}")

        if os.path.exists(csv_moving):
            moving_dfs[r_str] = pd.read_csv(csv_moving)
        else:
            print(f"  [警告] 見つかりません: {csv_moving}")

    def _normalize_with_std(mean_series, std_series):
        """平均値の最大値で mean と std を同時に規格化する"""
        max_val = mean_series.max()
        if max_val == 0 or np.isnan(max_val):
            return mean_series, std_series
        return mean_series / max_val, std_series / max_val

    # --- (range/moving) × (r only / k only / rk combined) の 6 パターン ---
    for analysis_type, dfs_dict, x_col, xlabel, out_dir, obs_df in [
        ('range',  range_dfs,  'depth_range',  'Depth Range [m]',  comp_range_dir,  obs_range_df),
        ('moving', moving_dfs, 'depth_center', 'Depth Center [m]', comp_moving_dir, obs_moving_df),
    ]:
        use_range_ticks = (analysis_type == 'range')
        if use_range_ticks:
            tick_vals = np.arange(2, 13, 2)

        # ---- プロット 1: 規格化 apparent r ----
        fig, ax = plt.subplots(figsize=(10, 8))
        for r_idx, r_str in enumerate(valid_r_keys):
            if r_str not in dfs_dict:
                continue
            df  = dfs_dict[r_str]
            col = r_line_colors[r_idx % len(r_line_colors)]
            r_norm, r_norm_std = _normalize_with_std(df['r_apparent_mean'], df['r_apparent_std'])
            ax.plot(df[x_col], r_norm,
                    label=f'r = {float(r_str)}',
                    color=col, linestyle='-', linewidth=1.5, marker='o', markersize=5)
            ax.fill_between(df[x_col],
                            r_norm - r_norm_std, r_norm + r_norm_std,
                            color=col, alpha=0.2)

        # 観測データを観測値自身の最大値で規格化
        obs_r_norm = obs_k_norm = None
        if len(obs_df) > 0:
            obs_r_max = obs_df['r'].max()
            obs_k_max = obs_df['k'].max()
            if obs_r_max > 0:
                obs_r_norm = obs_df['r'] / obs_r_max
            if obs_k_max > 0:
                obs_k_norm = obs_df['k'] / obs_k_max

        if use_range_ticks:
            ax.set_xticks(tick_vals)
            ax.set_xticklabels([f"0-{int(v)}" for v in tick_vals], fontsize=16)
        else:
            ax.tick_params(axis='x', labelsize=16)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel('Normalized Apparent r', fontsize=18)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylim(bottom=0)

        # 観測 r を黒線で重ね描き
        if obs_r_norm is not None:
            ax.plot(obs_df[x_col], obs_r_norm,
                    label='Observed', color='k', linestyle='-',
                    linewidth=2.5, marker='*', markersize=8, zorder=10)

        ax.legend(fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        prefix = os.path.join(out_dir, f'N_{count}_r')
        fig.savefig(f'{prefix}.png')
        fig.savefig(f'{prefix}.pdf')
        plt.close(fig)
        print(f"  -> 保存: {prefix}.png/.pdf")

        # ---- プロット 2: 規格化 apparent k ----
        fig, ax = plt.subplots(figsize=(10, 8))
        for r_idx, r_str in enumerate(valid_r_keys):
            if r_str not in dfs_dict:
                continue
            df  = dfs_dict[r_str]
            col = r_line_colors[r_idx % len(r_line_colors)]
            k_norm, k_norm_std = _normalize_with_std(df['k_apparent_mean'], df['k_apparent_std'])
            ax.plot(df[x_col], k_norm,
                    label=f'r = {float(r_str)}',
                    color=col, linestyle='-', linewidth=1.5, marker='s', markersize=5)
            ax.fill_between(df[x_col],
                            k_norm - k_norm_std, k_norm + k_norm_std,
                            color=col, alpha=0.2)

        if use_range_ticks:
            ax.set_xticks(tick_vals)
            ax.set_xticklabels([f"0-{int(v)}" for v in tick_vals], fontsize=16)
        else:
            ax.tick_params(axis='x', labelsize=16)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel('Normalized Apparent k', fontsize=18)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylim(bottom=0)

        # 観測 k を黒線で重ね描き
        if obs_k_norm is not None:
            ax.plot(obs_df[x_col], obs_k_norm,
                    label='Observed', color='k', linestyle='-',
                    linewidth=2.5, marker='*', markersize=8, zorder=10)

        ax.legend(fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        prefix = os.path.join(out_dir, f'N_{count}_k')
        fig.savefig(f'{prefix}.png')
        fig.savefig(f'{prefix}.pdf')
        plt.close(fig)
        print(f"  -> 保存: {prefix}.png/.pdf")

        # ---- プロット 3: 規格化 r（実線）＋規格化 k（破線）重ね合わせ ----
        fig, ax = plt.subplots(figsize=(10, 8))
        for r_idx, r_str in enumerate(valid_r_keys):
            if r_str not in dfs_dict:
                continue
            df      = dfs_dict[r_str]
            col     = r_line_colors[r_idx % len(r_line_colors)]
            r_label = f'r = {float(r_str)}'
            r_norm, r_norm_std = _normalize_with_std(df['r_apparent_mean'], df['r_apparent_std'])
            k_norm, k_norm_std = _normalize_with_std(df['k_apparent_mean'], df['k_apparent_std'])
            ax.plot(df[x_col], r_norm,
                    label=f'{r_label} (r)',
                    color=col, linestyle='-', linewidth=1.5, marker='o', markersize=5)
            ax.fill_between(df[x_col],
                            r_norm - r_norm_std, r_norm + r_norm_std,
                            color=col, alpha=0.15)
            ax.plot(df[x_col], k_norm,
                    label=f'{r_label} (k)',
                    color=col, linestyle='--', linewidth=1.5, marker='s', markersize=5)
            ax.fill_between(df[x_col],
                            k_norm - k_norm_std, k_norm + k_norm_std,
                            color=col, alpha=0.15)

        # 観測 r（実線）と k（破線）を黒で重ね描き
        if obs_r_norm is not None:
            ax.plot(obs_df[x_col], obs_r_norm,
                    label='Observed (r)', color='k', linestyle='-',
                    linewidth=2.5, marker='*', markersize=8, zorder=10)
        if obs_k_norm is not None:
            ax.plot(obs_df[x_col], obs_k_norm,
                    label='Observed (k)', color='k', linestyle='--',
                    linewidth=2.5, marker='*', markersize=8, zorder=10)

        if use_range_ticks:
            ax.set_xticks(tick_vals)
            ax.set_xticklabels([f"0-{int(v)}" for v in tick_vals], fontsize=16)
        else:
            ax.tick_params(axis='x', labelsize=16)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel('Normalized Apparent Value', fontsize=18)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=12, ncol=2)
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        prefix = os.path.join(out_dir, f'N_{count}_rk')
        fig.savefig(f'{prefix}.png')
        fig.savefig(f'{prefix}.pdf')
        plt.close(fig)
        print(f"  -> 保存: {prefix}.png/.pdf")

print("\n全プロット完了。")
