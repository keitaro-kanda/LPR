#!/usr/bin/env python3
# RSFD_generator.py
# ------------------------------------------------------------
# ラベル JSON から
#   1) ラベル1→1 cm, ラベル2→6 cm, ラベル3→式で計算
# の岩石サイズを取得し，
# 線形‑線形の累積サイズ‑頻度分布 (個数) を描画・保存し，
# べき則／指数関数フィッティングおよび比較プロットを滑らかに追加
# 誘電率 er = 6, 9, 12, 15 の4種類で計算・比較
# ------------------------------------------------------------

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from matplotlib.ticker import FixedLocator, FixedFormatter

NAN_Y_MARKER = 1e-5
MIN_FIT_POINTS = 3
ER_VALUES = [6, 9, 12, 15]
ER_COLORS = {6: 'blue', 9: 'green', 12: 'orange', 15: 'red'}
C_LIGHT = 299_792_458  # m/s


# ------------------------------------------------------------------
# 補助関数
# ------------------------------------------------------------------
def none_to_nan(v):
    return np.nan if v is None else v


def format_p_value(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def format_k_value(k):
    coefficient = k / 1e-3
    if coefficient >= 10:
        return f"{coefficient:.1f}×10⁻³"
    return f"{coefficient:.2f}×10⁻³"


def create_rsfd_plot(x_data, y_data, xlabel, ylabel, output_path,
                     fit_x=None, fit_y=None, fit_params=None,
                     show_plot=False, dpi_png=300, dpi_pdf=600,
                     data_color='black', fit_color='red',
                     xlim=None):
    plt.figure(figsize=(8, 6))

    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)
    nan_mask = np.isnan(y_arr)
    if nan_mask.any():
        x_nan = np.where(np.isnan(x_arr[nan_mask]), 1.0, x_arr[nan_mask])
        plt.plot(x_nan, np.full(nan_mask.sum(), NAN_Y_MARKER),
                 marker='v', linestyle='', color=data_color, markersize=8,
                 clip_on=False, zorder=5, label='No data (NaN)')
    if (~nan_mask).any():
        plt.plot(x_arr[~nan_mask], y_arr[~nan_mask],
                 marker='o', linestyle='', color=data_color, label='Observed data')

    if fit_x is not None and fit_y is not None and fit_params is not None:
        if not np.isnan(fit_params['k']):
            k_str = format_k_value(fit_params['k'])
            fit_label = f"Fit: r={fit_params['r']:.2f}, k={k_str}, R²={fit_params['R2']:.3f}"
            plt.plot(fit_x, fit_y, linestyle='--', linewidth=1.5, color=fit_color, label=fit_label)

    plt.xscale('log')
    plt.yscale('log')

    ax = plt.gca()
    ax.xaxis.set_major_locator(FixedLocator([1, 5, 10, 50]))
    ax.xaxis.set_major_formatter(FixedFormatter(['1 cm', '5 cm', '10 cm', '50 cm']))

    plt.xlim(0.7, 70)
    plt.ylim(3e-5, 5e-2)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(labelsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=16, frameon=True, fancybox=True)
    plt.tight_layout()

    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f'プロット保存: {output_path}.png')


def calc_fitting(sizes, counts):
    mask = sizes > 0
    if len(sizes) == 0 or mask.sum() < MIN_FIT_POINTS:
        nan_arr = np.full(200, np.nan)
        nan_val = np.nan
        d_fit = np.linspace(sizes.min(), sizes.max(), 200) if len(sizes) >= 2 else np.linspace(1.0, 10.0, 200)
        return (nan_val, nan_val, nan_val, nan_arr, nan_val, nan_val, nan_val, int(mask.sum()), 0), \
               (nan_val, nan_val, nan_val, nan_arr, nan_val, nan_val, nan_val, int(mask.sum()), 0), \
               d_fit

    log_D = np.log(sizes[mask])
    log_N = np.log(counts[mask])

    res_pow = sm.OLS(log_N, sm.add_constant(log_D)).fit()
    log_k_pow, r_pow = res_pow.params
    k_pow = np.exp(log_k_pow)

    res_exp = sm.OLS(log_N, sm.add_constant(sizes[mask])).fit()
    log_k_exp, r_exp = res_exp.params
    k_exp = np.exp(log_k_exp)

    D_fit = np.linspace(sizes.min(), sizes.max(), 200)

    return (k_pow, np.abs(r_pow), res_pow.rsquared,
            k_pow * D_fit**r_pow,
            res_pow.tvalues[1], res_pow.pvalues[1], res_pow.bse[1],
            int(res_pow.nobs), res_pow.df_resid), \
           (k_exp, np.abs(r_exp), res_exp.rsquared,
            k_exp * np.exp(r_exp * D_fit),
            res_exp.tvalues[1], res_exp.pvalues[1], res_exp.bse[1],
            int(res_exp.nobs), res_exp.df_resid), \
           D_fit


def calc_fitting_area_normalized(sizes, counts, area):
    counts_normalized = counts / area if len(counts) > 0 else np.array([np.nan])
    mask = sizes > 0
    if len(sizes) == 0 or mask.sum() < MIN_FIT_POINTS:
        nan_arr = np.full(200, np.nan)
        nan_val = np.nan
        d_fit = np.linspace(sizes.min(), sizes.max(), 200) if len(sizes) >= 2 else np.linspace(1.0, 10.0, 200)
        return (nan_val, nan_val, nan_val, nan_arr, nan_val, nan_val, nan_val, int(mask.sum()), 0), \
               d_fit, counts_normalized

    log_D = np.log(sizes[mask])
    log_N = np.log(counts_normalized[mask])

    res_pow = sm.OLS(log_N, sm.add_constant(log_D)).fit()
    log_k_pow, r_pow = res_pow.params
    k_pow = np.exp(log_k_pow)

    D_fit = np.linspace(sizes.min(), sizes.max(), 200)

    return (k_pow, np.abs(r_pow), res_pow.rsquared,
            k_pow * D_fit**r_pow,
            res_pow.tvalues[1], res_pow.pvalues[1], res_pow.bse[1],
            int(res_pow.nobs), res_pow.df_resid), \
           D_fit, counts_normalized


def save_processing_config(config_path, data_path, mode, time_range, horizontal_range,
                            time_min, time_max, horizontal_min, horizontal_max, output_dir,
                            area=None, group1_size=1.0):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {"label_file": data_path, "processing_history": []}

    config["processing_history"].append({
        "id": len(config["processing_history"]) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "time_range": time_range,
        "horizontal_range": horizontal_range,
        "time_min": time_min,
        "time_max": time_max,
        "horizontal_min": horizontal_min,
        "horizontal_max": horizontal_max,
        "output_dir": output_dir,
        "area": area,
        "group1_size": group1_size
    })

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f'処理設定を保存: {config_path}')


def load_all_configs():
    print('\n検出された岩石のラベルデータファイル(.json)のパスを入力してください:')
    data_path = input().strip()
    if not (os.path.exists(data_path) and data_path.lower().endswith('.json')):
        raise FileNotFoundError('正しい .json ファイルを指定してください。')

    base_dir = os.path.join(os.path.dirname(os.path.dirname(data_path)), 'RSFD')
    config_path = os.path.join(base_dir, 'processing_config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'設定ファイルが見つかりません: {config_path}\n先に新規データ処理を実行してください。')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if not config["processing_history"]:
        raise ValueError('処理履歴が空です。')

    print('\n=== 処理履歴 ===')
    for record in config["processing_history"]:
        print(f'ID: {record["id"]}')
        print(f'  日時: {record["timestamp"]}')
        print(f'  モード: {record["mode"]}')
        if record["mode"] != '4':
            print(f'  時間範囲: {record["time_range"] if record["time_range"] else "指定なし"}')
            print(f'  水平位置範囲: {record["horizontal_range"] if record["horizontal_range"] else "指定なし"}')
        print(f'  出力ディレクトリ: {record["output_dir"]}')
        print()

    return data_path, config["processing_history"]


def _make_combined_er_plot(er_results, xlabel, ylabel, output_path,
                            show_plot=False, dpi_png=300, dpi_pdf=600):
    """全誘電率の結果を1枚にまとめたプロット"""
    _, ax = plt.subplots(figsize=(10, 8))

    for er in ER_VALUES:
        if er not in er_results:
            continue
        d = er_results[er]
        color = ER_COLORS[er]
        ax.scatter(d['unique_sizes'], d['cum_counts'],
                   color=color, marker='o', s=40, label=f'ε_r = {er}', zorder=3)
        if not np.isnan(d['k']):
            k_str = format_k_value(d['k'])
            ax.plot(d['D_fit'], d['N_pow_fit'],
                    linestyle='--', linewidth=1.5, color=color,
                    label=f"Fit (ε_r={er}): r={d['r']:.2f}, k={k_str}, R²={d['R2']:.3f}")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(FixedLocator([1, 5, 10, 50]))
    ax.xaxis.set_major_formatter(FixedFormatter(['1 cm', '5 cm', '10 cm', '50 cm']))
    ax.set_xlim(0.7, 70)
    ax.set_ylim(3e-5, 5e-2)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True)
    plt.tight_layout()

    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f'全er比較プロット保存: {output_path}.png')


def process_core(data_path, mode, time_min, time_max, horizontal_min, horizontal_max,
                 output_dir, area, group1_size, show_plot_flag):
    """JSON読み込みから全er処理・プロット・TXT保存まで実行"""

    # サブディレクトリ作成
    output_dir_group1_3   = os.path.join(output_dir, 'RSFD_group1-3')
    output_dir_group2_3   = os.path.join(output_dir, 'RSFD_group2-3')
    output_dir_comparison = os.path.join(output_dir, 'RSFD_comparison')
    output_dir_txt_counts = os.path.join(output_dir, 'txt_counts_by_label')
    output_dir_txt_detail = os.path.join(output_dir, 'txt_label2-3_detail')
    output_dir_txt_summary = os.path.join(output_dir, 'txt_fitting_summary')
    for d in [output_dir_group1_3, output_dir_group2_3, output_dir_comparison,
              output_dir_txt_counts, output_dir_txt_detail, output_dir_txt_summary]:
        os.makedirs(d, exist_ok=True)

    # JSON読み込み
    with open(data_path, 'r') as f:
        results = json.load(f).get('results', {})

    x           = np.array([v['x']     for v in results.values()])
    y           = np.array([v['y']     for v in results.values()])
    lab         = np.array([v['label'] for v in results.values()], dtype=int)
    time_top    = np.array([none_to_nan(v['time_top'])    for v in results.values()], dtype=float)
    time_bottom = np.array([none_to_nan(v['time_bottom']) for v in results.values()], dtype=float)
    print('ラベルデータ読み込み完了:', len(lab), '個')

    # データ範囲フィルタリング
    original_count = len(lab)
    if time_min is not None and time_max is not None:
        mask_group1 = (lab == 1) & (y >= time_min) & (y <= time_max)
        mask_others = (lab != 1) & (time_top >= time_min) & (time_top <= time_max)
        time_mask = mask_group1 | mask_others
        if mode == '3':
            time_mask = ~time_mask
        x = x[time_mask]; y = y[time_mask]; lab = lab[time_mask]
        time_top = time_top[time_mask]; time_bottom = time_bottom[time_mask]
        verb = '除外' if mode == '3' else 'フィルタリング'
        print(f'時間範囲{verb}後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')

    if horizontal_min is not None and horizontal_max is not None:
        horizontal_mask = (x >= horizontal_min) & (x <= horizontal_max)
        if mode == '3':
            horizontal_mask = ~horizontal_mask
        x = x[horizontal_mask]; y = y[horizontal_mask]; lab = lab[horizontal_mask]
        time_top = time_top[horizontal_mask]; time_bottom = time_bottom[horizontal_mask]
        verb = '除外' if mode == '3' else 'フィルタリング'
        print(f'水平位置範囲{verb}後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')

    print(f'フィルタリング完了: {len(lab)}個のデータを使用')

    # er非依存の計算
    counts      = {k: int(np.sum(lab == k)) for k in range(1, 7)}
    size_label1 = np.full(counts[1], group1_size)
    size_label2 = np.full(counts[2], 6.0)
    mask2_valid = (lab == 2) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
    mask3_valid = (lab == 3) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))

    # counts_by_label TXTを一度だけ保存（er非依存）
    counts_txt_path = os.path.join(output_dir_txt_counts, 'RSFD_counts_by_label.txt')
    with open(counts_txt_path, 'w') as f:
        f.write('# RSFD Label Counts\n')
        f.write(f'# Original data count: {original_count}\n')
        if mode == '1':
            f.write('# Mode: Full range (no filtering)\n')
        elif mode == '2':
            f.write('# Mode: Use only specified range\n')
            if time_min is not None:
                f.write(f'# Time range filter: {time_min} - {time_max} ns\n')
            if horizontal_min is not None:
                f.write(f'# Horizontal range filter: {horizontal_min} - {horizontal_max} m\n')
        elif mode == '3':
            f.write('# Mode: Remove specified range\n')
            if time_min is not None:
                f.write(f'# Removed time range: {time_min} - {time_max} ns\n')
            if horizontal_min is not None:
                f.write(f'# Removed horizontal range: {horizontal_min} - {horizontal_max} m\n')
        f.write(f'# Filtered data count: {len(lab)} ({len(lab)/original_count*100:.1f}%)\n\n')
        for k, v in counts.items():
            f.write(f'Label {k}: {v}\n')
    print(f'ラベル別個数TXT保存: {counts_txt_path}')

    # まとめプロット用データ収集辞書
    er_results_grp1_3 = {}
    er_results_grp2_3 = {}

    # ------------------------------------------------------------------
    # 誘電率ループ
    # ------------------------------------------------------------------
    for er in ER_VALUES:
        print(f'\n--- er = {er} ---')

        # サイズ計算
        sizes_group2 = np.round(
            (time_bottom[mask2_valid] - time_top[mask2_valid]) * 1e-9 * C_LIGHT / np.sqrt(er) * 0.5 * 100,
            decimals=3)
        sizes_group3 = np.round(
            (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * C_LIGHT / np.sqrt(er) * 0.5 * 100,
            decimals=3)

        all_sizes_traditional = np.concatenate([size_label1, size_label2, sizes_group3])
        all_sizes_est_grp2    = np.concatenate([size_label1, sizes_group2, sizes_group3])
        all_sizes_grp2_3      = np.concatenate([sizes_group2, sizes_group3])

        if all_sizes_traditional.size == 0:
            print(f'er={er}: 有効なラベル1–3が見つかりません。スキップ。')
            continue

        # 累積サイズ-頻度分布
        uniq_trad = np.sort(np.unique(all_sizes_traditional))
        cum_trad  = np.array([(all_sizes_traditional >= s).sum() for s in uniq_trad], dtype=int)
        uniq_est  = np.sort(np.unique(all_sizes_est_grp2))
        cum_est   = np.array([(all_sizes_est_grp2 >= s).sum() for s in uniq_est], dtype=int)
        if len(all_sizes_grp2_3) == 0:
            uniq_g23 = np.array([1.0]); cum_g23 = np.array([np.nan])
        else:
            uniq_g23 = np.sort(np.unique(all_sizes_grp2_3))
            cum_g23  = np.array([(all_sizes_grp2_3 >= s).sum() for s in uniq_g23], dtype=int)

        # フィッティング（非規格化）
        (k_pow_trad, r_pow_trad, R2_pow_trad, _, t_pow_trad, p_pow_trad, se_pow_trad, n_pow_trad, dof_pow_trad), \
        (k_exp_trad, r_exp_trad, R2_exp_trad, _, t_exp_trad, p_exp_trad, se_exp_trad, n_exp_trad, dof_exp_trad), _ \
            = calc_fitting(uniq_trad, cum_trad)

        (k_pow_est, r_pow_est, R2_pow_est, _, t_pow_est, p_pow_est, se_pow_est, n_pow_est, dof_pow_est), \
        (k_exp_est, r_exp_est, R2_exp_est, _, t_exp_est, p_exp_est, se_exp_est, n_exp_est, dof_exp_est), _ \
            = calc_fitting(uniq_est, cum_est)

        (k_pow_g23, r_pow_g23, R2_pow_g23, _, t_pow_g23, p_pow_g23, se_pow_g23, n_pow_g23, dof_pow_g23), \
        (k_exp_g23, r_exp_g23, R2_exp_g23, _, t_exp_g23, p_exp_g23, se_exp_g23, n_exp_g23, dof_exp_g23), _ \
            = calc_fitting(uniq_g23, cum_g23)

        # 面積規格化フィッティング (Group1-3)
        (k_area_est, r_area_est, R2_area_est, N_fit_area_est,
         t_area_est, p_area_est, se_area_est, n_area_est, dof_area_est), \
        D_fit_est, cum_area_est = calc_fitting_area_normalized(uniq_est, cum_est, area)

        # 面積規格化フィッティング (Group2-3)
        (k_area_g23, r_area_g23, R2_area_g23, N_fit_area_g23,
         t_area_g23, p_area_g23, se_area_g23, n_area_g23, dof_area_g23), \
        D_fit_g23, cum_area_g23 = calc_fitting_area_normalized(uniq_g23, cum_g23, area)

        # プロット: Group1-3
        create_rsfd_plot(
            uniq_est, cum_area_est,
            'Rock size [cm]', 'Cumulative number of rocks /m²',
            os.path.join(output_dir_group1_3, f'RSFD_group1-3_er{er}'),
            fit_x=D_fit_est, fit_y=N_fit_area_est,
            fit_params={'k': k_area_est, 'r': r_area_est, 'R2': R2_area_est,
                        'p_str': format_p_value(p_area_est)},
            show_plot=show_plot_flag
        )

        # プロット: Group2-3
        create_rsfd_plot(
            uniq_g23, cum_area_g23,
            'Rock size [cm]', 'Cumulative number of rocks /m²',
            os.path.join(output_dir_group2_3, f'RSFD_group2-3_er{er}'),
            fit_x=D_fit_g23, fit_y=N_fit_area_g23,
            fit_params={'k': k_area_g23, 'r': r_area_g23, 'R2': R2_area_g23,
                        'p_str': format_p_value(p_area_g23)},
            show_plot=False
        )

        # プロット: Group1-3 vs Group2-3 比較
        comp_path = os.path.join(output_dir_comparison, f'RSFD_comparison_er{er}')
        _, ax = plt.subplots(figsize=(10, 8))
        ax.plot(uniq_est, cum_area_est,
                marker='o', linestyle='', color='black', label='Observed data')
        if not np.isnan(k_area_est):
            k_str_est = format_k_value(k_area_est)
            ax.plot(D_fit_est, N_fit_area_est,
                    linestyle='--', linewidth=1.5, color='blue',
                    label=f'Contain size-unmeasurable Fit: r={r_area_est:.2f}, k={k_str_est}, R²={R2_area_est:.3f}')
        if not np.isnan(k_area_g23):
            k_str_g23 = format_k_value(k_area_g23)
            ax.plot(D_fit_g23, N_fit_area_g23,
                    linestyle='--', linewidth=1.5, color='red',
                    label=f'Only size-measurable Fit: r={r_area_g23:.2f}, k={k_str_g23}, R²={R2_area_g23:.3f}')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.xaxis.set_major_locator(FixedLocator([1, 5, 10, 50]))
        ax.xaxis.set_major_formatter(FixedFormatter(['1 cm', '5 cm', '10 cm', '50 cm']))
        ax.set_xlim(0.5, 50)
        ax.set_xlabel('Rock size [cm]', fontsize=20)
        ax.set_ylabel('Cumulative number of rocks /m²', fontsize=20)
        ax.tick_params(labelsize=16)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True)
        plt.tight_layout()
        plt.savefig(f'{comp_path}.png', dpi=300)
        plt.savefig(f'{comp_path}.pdf', dpi=600)
        plt.close()
        print(f'Group比較プロット保存: {comp_path}.png')

        # TXT: Label2-3_detail（er別）
        if mask2_valid.any() or mask3_valid.any():
            combined_mask = mask2_valid | mask3_valid
            x_c   = x[combined_mask]; y_c = y[combined_mask]
            tt_c  = time_top[combined_mask]; tb_c = time_bottom[combined_mask]
            lab_c = lab[combined_mask]
            size_c = (tb_c - tt_c) * 1e-9 * C_LIGHT / np.sqrt(er) * 0.5 * 100
            sort_idx = np.argsort(size_c)
            detail_path = os.path.join(output_dir_txt_detail, f'Label2-3_detail_er{er}.txt')
            with open(detail_path, 'w') as f:
                f.write(f'# er = {er}\n')
                f.write('#label\t x\t y\t time_top\t time_bottom\t size_cm\n')
                for i in sort_idx:
                    f.write(f'{lab_c[i]}\t{x_c[i]:.6f}\t{y_c[i]:.6f}\t'
                            f'{tt_c[i]:.3f}\t{tb_c[i]:.3f}\t{size_c[i]:.8f}\n')
            print(f'Label2-3詳細TXT保存: {detail_path}')

        # TXT: Fitting summary（er別）
        summary_path = os.path.join(output_dir_txt_summary, f'RSFD_fitting_summary_er{er}.txt')
        with open(summary_path, 'w') as f:
            f.write(f'# er = {er}\n')
            f.write('DataSet\tModel\tk\tr\tR_squared\tr_StdErr\tr_t_value\tr_p_value\tN_points\tDOF\n')
            f.write(f'Traditional\tPower\t{k_pow_trad:.4e}\t{r_pow_trad:.4f}\t{R2_pow_trad:.4f}\t'
                    f'{se_pow_trad:.4f}\t{t_pow_trad:.3f}\t{p_pow_trad:.4e}\t{n_pow_trad}\t{dof_pow_trad}\n')
            f.write(f'Traditional\tExponential\t{k_exp_trad:.4e}\t{r_exp_trad:.4f}\t{R2_exp_trad:.4f}\t'
                    f'{se_exp_trad:.4f}\t{t_exp_trad:.3f}\t{p_exp_trad:.4e}\t{n_exp_trad}\t{dof_exp_trad}\n')
            f.write(f'Estimate_Group2\tPower\t{k_pow_est:.4e}\t{r_pow_est:.4f}\t{R2_pow_est:.4f}\t'
                    f'{se_pow_est:.4f}\t{t_pow_est:.3f}\t{p_pow_est:.4e}\t{n_pow_est}\t{dof_pow_est}\n')
            f.write(f'Estimate_Group2\tExponential\t{k_exp_est:.4e}\t{r_exp_est:.4f}\t{R2_exp_est:.4f}\t'
                    f'{se_exp_est:.4f}\t{t_exp_est:.3f}\t{p_exp_est:.4e}\t{n_exp_est}\t{dof_exp_est}\n')
            f.write(f'Group2-3_Only\tPower\t{k_pow_g23:.4e}\t{r_pow_g23:.4f}\t{R2_pow_g23:.4f}\t'
                    f'{se_pow_g23:.4f}\t{t_pow_g23:.3f}\t{p_pow_g23:.4e}\t{n_pow_g23}\t{dof_pow_g23}\n')
            f.write(f'Group2-3_Only\tExponential\t{k_exp_g23:.4e}\t{r_exp_g23:.4f}\t{R2_exp_g23:.4f}\t'
                    f'{se_exp_g23:.4f}\t{t_exp_g23:.3f}\t{p_exp_g23:.4e}\t{n_exp_g23}\t{dof_exp_g23}\n')
            f.write(f'Area_Normalized_Est_Grp2\tPower\t{k_area_est:.4e}\t{r_area_est:.4f}\t{R2_area_est:.4f}\t'
                    f'{se_area_est:.4f}\t{t_area_est:.3f}\t{p_area_est:.4e}\t{n_area_est}\t{dof_area_est}\n')
            f.write(f'Area_Normalized_Grp2-3\tPower\t{k_area_g23:.4e}\t{r_area_g23:.4f}\t{R2_area_g23:.4f}\t'
                    f'{se_area_g23:.4f}\t{t_area_g23:.3f}\t{p_area_g23:.4e}\t{n_area_g23}\t{dof_area_g23}\n')
            f.write(f'\n# Area: {area} m²\n')
        print(f'フィッティングサマリー保存: {summary_path}')

        # まとめプロット用にデータを収集
        er_results_grp1_3[er] = {
            'unique_sizes': uniq_est, 'cum_counts': cum_area_est,
            'D_fit': D_fit_est, 'N_pow_fit': N_fit_area_est,
            'r': r_area_est, 'k': k_area_est, 'R2': R2_area_est
        }
        er_results_grp2_3[er] = {
            'unique_sizes': uniq_g23, 'cum_counts': cum_area_g23,
            'D_fit': D_fit_g23, 'N_pow_fit': N_fit_area_g23,
            'r': r_area_g23, 'k': k_area_g23, 'R2': R2_area_g23
        }

    # ------------------------------------------------------------------
    # 全er比較まとめプロット
    # ------------------------------------------------------------------
    _make_combined_er_plot(
        er_results_grp1_3,
        'Rock size [cm]', 'Cumulative number of rocks /m²',
        os.path.join(output_dir_group1_3, 'RSFD_group1-3_all_er'),
        show_plot=show_plot_flag
    )
    _make_combined_er_plot(
        er_results_grp2_3,
        'Rock size [cm]', 'Cumulative number of rocks /m²',
        os.path.join(output_dir_group2_3, 'RSFD_group2-3_all_er'),
        show_plot=False
    )


# ------------------------------------------------------------------
# 1. 起動モード選択
# ------------------------------------------------------------------
print('=== RSFD Plot Generator ===')
print('1: 新規データ処理')
print('2: 記録ファイルから既存プロットを再作成')
startup_mode = input('モードを選択してください (1/2): ').strip()

if startup_mode not in ['1', '2']:
    raise ValueError('モードは1または2を選択してください。')

show_plot_flag = (startup_mode == '1')

# ------------------------------------------------------------------
# 2. モード別処理
# ------------------------------------------------------------------
if startup_mode == '1':
    print('\n検出された岩石のラベルデータファイル(.json)のパスを入力してください:')
    data_path = input().strip()
    if not (os.path.exists(data_path) and data_path.lower().endswith('.json')):
        raise FileNotFoundError('正しい .json ファイルを指定してください。')

    print('\n=== データ範囲モード選択 ===')
    print('1: 全範囲のデータを使用')
    print('2: 特定の時間・距離範囲のデータのみを使用')
    print('3: 特定の時間・距離範囲のデータのみを取り除いて使用')
    mode = input('モードを選択してください (1/2/3): ').strip()

    if mode not in ['1', '2', '3']:
        raise ValueError('モードは1, 2, 3のいずれかを選択してください。')

    if mode == '1':
        time_range = ''
        horizontal_range = ''
        area = 16136
        print('全範囲のデータを使用します。')
        print(f'面積: {area} m²（固定値）')
    else:
        print('\n=== データ範囲指定 ===')
        print('指定した範囲のデータのみを使用します。' if mode == '2' else '指定した範囲のデータを除外します。')
        time_range       = input('時間範囲 [ns] を入力してください（例: 50-100, Enter: 指定なし）: ').strip()
        horizontal_range = input('水平位置範囲 [m] を入力してください（例: 0-100, Enter: 指定なし）: ').strip()
        area_input       = input('面積 [m²] を入力してください: ').strip()
        try:
            area = float(area_input)
        except ValueError:
            raise ValueError('面積の入力形式が正しくありません。数値を入力してください。')

    try:
        if time_range:
            time_min, time_max = map(float, time_range.split('-'))
        else:
            time_min, time_max = None, None
        if horizontal_range:
            horizontal_min, horizontal_max = map(float, horizontal_range.split('-'))
        else:
            horizontal_min, horizontal_max = None, None
    except ValueError:
        raise ValueError('範囲の入力形式が正しくありません。例: 0-100')

    print('\n=== Group 1 サイズ設定 ===')
    print('1: 1 cm')
    print('2: 2 cm')
    group1_size_choice = input('Group 1のサイズを選択してください (1/2): ').strip()
    if group1_size_choice not in ['1', '2']:
        raise ValueError('Group 1のサイズは1または2を選択してください。')
    group1_size     = 1.0 if group1_size_choice == '1' else 2.0
    group1_size_str = 'group1_1cm' if group1_size_choice == '1' else 'group1_2cm'

    base_dir  = os.path.join(os.path.dirname(os.path.dirname(data_path)), 'RSFD')
    file_name = os.path.splitext(os.path.basename(data_path))[0]

    if mode == '1':
        output_dir = os.path.join(base_dir, f'{file_name}_full_range_{group1_size_str}')
    elif mode == '2':
        if time_range and not horizontal_range:
            output_dir = os.path.join(base_dir, f'{file_name}_t{time_min}-{time_max}_{group1_size_str}')
        elif horizontal_range and not time_range:
            output_dir = os.path.join(base_dir, f'{file_name}_x{horizontal_min}-{horizontal_max}_{group1_size_str}')
        elif time_range and horizontal_range:
            output_dir = os.path.join(base_dir, f'{file_name}_t{time_min}-{time_max}_x{horizontal_min}-{horizontal_max}_{group1_size_str}')
        else:
            output_dir = os.path.join(base_dir, f'{file_name}_full_range_{group1_size_str}')
    else:  # mode == '3'
        if time_range and not horizontal_range:
            output_dir = os.path.join(base_dir, f'{file_name}_remove_t{time_min}-{time_max}_{group1_size_str}')
        elif horizontal_range and not time_range:
            output_dir = os.path.join(base_dir, f'{file_name}_remove_x{horizontal_min}-{horizontal_max}_{group1_size_str}')
        elif time_range and horizontal_range:
            output_dir = os.path.join(base_dir, f'{file_name}_remove_t{time_min}-{time_max}_x{horizontal_min}-{horizontal_max}_{group1_size_str}')
        else:
            output_dir = os.path.join(base_dir, f'{file_name}_full_range_{group1_size_str}')
    os.makedirs(output_dir, exist_ok=True)

    process_core(data_path, mode, time_min, time_max, horizontal_min, horizontal_max,
                 output_dir, area, group1_size, show_plot_flag)

    config_path = os.path.join(base_dir, 'processing_config.json')
    save_processing_config(
        config_path, data_path, mode, time_range, horizontal_range,
        time_min, time_max, horizontal_min, horizontal_max, output_dir,
        area=area, group1_size=group1_size
    )

else:  # startup_mode == '2'
    data_path, all_records = load_all_configs()
    print(f'\n全ての処理履歴（{len(all_records)}件）を順番に再作成します。\n')

    for record_idx, rec in enumerate(all_records, 1):
        print(f'=== 処理 {record_idx}/{len(all_records)}: ID {rec["id"]} ===')
        try:
            mode             = rec['mode']
            time_range       = rec['time_range']
            horizontal_range = rec['horizontal_range']
            time_min         = rec['time_min']
            time_max         = rec['time_max']
            horizontal_min   = rec['horizontal_min']
            horizontal_max   = rec['horizontal_max']
            output_dir       = rec['output_dir']
            area             = rec.get('area') or 16136
            group1_size      = rec.get('group1_size', 1.0)

            print(f'データファイル: {data_path}')
            print(f'モード: {mode}')
            print(f'時間範囲: {time_range if time_range else "指定なし"}')
            print(f'水平位置範囲: {horizontal_range if horizontal_range else "指定なし"}')
            print(f'出力ディレクトリ: {output_dir}')

            if mode == '4':
                print('モード4 (複数範囲比較) の履歴はスキップします。make_RSFD_compare_multiple_region.py を使用してください。')
                print(f'✓ 処理 {record_idx} スキップ\n')
                continue

            os.makedirs(output_dir, exist_ok=True)
            process_core(data_path, mode, time_min, time_max, horizontal_min, horizontal_max,
                         output_dir, area, group1_size, show_plot_flag=False)

            print(f'✓ 処理 {record_idx} 完了\n')

        except Exception as e:
            print(f'✗ 処理 {record_idx} でエラーが発生しました: {e}')
            print(f'次の処理に進みます...\n')
            continue

    print('\n全ての再作成処理が完了しました！')
    exit(0)

print('\nすべて完了しました！')
