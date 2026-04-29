import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import glob


MAX_DEPTH = 12.0
AREA_SIZE_M2 = 18300.0
ROCK_SIZE_MIN = 0.06
BASE_DIR = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD_relative'


def calculate_slope(diameters, area=1.0):
    if len(diameters) < 2:
        return np.nan, np.nan, np.array([]), np.array([])

    sorted_d = np.sort(diameters)
    y_cumulative = np.arange(len(sorted_d), 0, -1) / area

    x_log = np.log10(sorted_d)
    y_log = np.log10(y_cumulative)

    n = len(x_log)
    sx = x_log.sum(); sy = y_log.sum()
    sxx = np.dot(x_log, x_log); sxy = np.dot(x_log, y_log)
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    return slope, intercept, x_log, y_log


def analyze_one_combination(total_rocks, csv_files, d_start, d_end, area, out_dir):
    d_min_cm = ROCK_SIZE_MIN * 100.0

    r_true_list = []
    k_true_list = []
    r_apparent_list = []
    k_apparent_list = []

    slope_true_full_list      = []
    intercept_true_full_list  = []
    slope_true_full_cnt_list  = []
    intercept_true_full_cnt_list = []
    slope_det_list            = []
    intercept_det_list        = []
    slope_det_cnt_list        = []
    intercept_det_cnt_list    = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # 全範囲 True fit（プロット用）
        diameters_all_cm = df['diameter'].values * 100.0
        if len(diameters_all_cm) > 5:
            s_tf, i_tf, _, _ = calculate_slope(diameters_all_cm, AREA_SIZE_M2)
            s_tfc, i_tfc, _, _ = calculate_slope(diameters_all_cm, 1.0)
        else:
            s_tf = i_tf = s_tfc = i_tfc = np.nan
        slope_true_full_list.append(s_tf)
        intercept_true_full_list.append(i_tf)
        slope_true_full_cnt_list.append(s_tfc)
        intercept_true_full_cnt_list.append(i_tfc)

        # 深さ範囲フィルタ
        depth_mask = (df['depth'] >= d_start) & (df['depth'] < d_end)
        df_range = df[depth_mask]

        # 範囲内 True fit（density）
        diameters_true_cm = df_range['diameter'].values * 100.0
        if len(diameters_true_cm) > 5:
            slope_t, intercept_t, _, _ = calculate_slope(diameters_true_cm, area)
            r_true_list.append(-slope_t if not np.isnan(slope_t) else np.nan)
            k_true_list.append(
                (10**intercept_t) * (d_min_cm**slope_t) if not np.isnan(slope_t) else np.nan
            )
        else:
            r_true_list.append(np.nan)
            k_true_list.append(np.nan)

        # 範囲内 Apparent fit（density / count）
        diameters_det_cm = df_range.loc[df_range['is_detected'] == True, 'diameter'].values * 100.0
        if len(diameters_det_cm) > 5:
            slope_d, intercept_d, _, _ = calculate_slope(diameters_det_cm, area)
            slope_dc, intercept_dc, _, _ = calculate_slope(diameters_det_cm, 1.0)
            r_apparent_list.append(-slope_d if not np.isnan(slope_d) else np.nan)
            k_apparent_list.append(
                (10**intercept_d) * (d_min_cm**slope_d) if not np.isnan(slope_d) else np.nan
            )
        else:
            slope_d = intercept_d = slope_dc = intercept_dc = np.nan
            r_apparent_list.append(np.nan)
            k_apparent_list.append(np.nan)
        slope_det_list.append(slope_d)
        intercept_det_list.append(intercept_d)
        slope_det_cnt_list.append(slope_dc)
        intercept_det_cnt_list.append(intercept_dc)

    os.makedirs(out_dir, exist_ok=True)

    iter_df = pd.DataFrame({
        'iteration':              range(1, len(csv_files) + 1),
        'r_true':                 r_true_list,
        'k_true':                 k_true_list,
        'r_apparent':             r_apparent_list,
        'k_apparent':             k_apparent_list,
        'slope_true_full':        slope_true_full_list,
        'intercept_true_full':    intercept_true_full_list,
        'slope_true_full_cnt':    slope_true_full_cnt_list,
        'intercept_true_full_cnt': intercept_true_full_cnt_list,
        'slope_det':              slope_det_list,
        'intercept_det':          intercept_det_list,
        'slope_det_cnt':          slope_det_cnt_list,
        'intercept_det_cnt':      intercept_det_cnt_list,
    })
    iter_df.to_csv(os.path.join(out_dir, 'iteration_results.csv'), index=False)

    r_t_arr = np.array(r_true_list, dtype=float)
    k_t_arr = np.array(k_true_list, dtype=float)
    r_d_arr = np.array(r_apparent_list, dtype=float)
    k_d_arr = np.array(k_apparent_list, dtype=float)

    r_true_mean = float(np.nanmean(r_t_arr))
    r_true_std  = float(np.nanstd(r_t_arr))
    k_true_mean = float(np.nanmean(k_t_arr))
    k_true_std  = float(np.nanstd(k_t_arr))
    r_app_mean  = float(np.nanmean(r_d_arr))
    r_app_std   = float(np.nanstd(r_d_arr))
    k_app_mean  = float(np.nanmean(k_d_arr))
    k_app_std   = float(np.nanstd(k_d_arr))

    with open(os.path.join(out_dir, 'stats.txt'), 'w') as f:
        f.write(f"Depth range    : {d_start} - {d_end} m\n")
        f.write(f"Area           : {area:.2f} m2\n")
        f.write(f"r_true         = {r_true_mean:.4f} +/- {r_true_std:.4f}\n")
        f.write(f"k_true         = {k_true_mean:.4e} +/- {k_true_std:.4e}\n")
        f.write(f"r_apparent     = {r_app_mean:.4f} +/- {r_app_std:.4f}\n")
        f.write(f"k_apparent     = {k_app_mean:.4e} +/- {k_app_std:.4e}\n")

    summary = {
        "r_true_mean":     r_true_mean,
        "r_true_std":      r_true_std,
        "k_true_mean":     k_true_mean,
        "k_true_std":      k_true_std,
        "r_apparent_mean": r_app_mean,
        "r_apparent_std":  r_app_std,
        "k_apparent_mean": k_app_mean,
        "k_apparent_std":  k_app_std,
    }
    return summary, iter_df


def _build_common_xaxis(r_true, total_rocks):
    d_min_cm = ROCK_SIZE_MIN * 100.0
    D_max_auto_cm = d_min_cm * (total_rocks ** (1.0 / r_true))
    x_log_common = np.linspace(np.log10(d_min_cm), np.log10(D_max_auto_cm), 100)
    return 10**x_log_common, x_log_common, d_min_cm, D_max_auto_cm


def _fit_band(iter_df, slope_col, intercept_col, x_log_common, d_min_cm):
    """各イテレーションの直線を積み上げ、平均・標準偏差・r・k の統計を返す。"""
    lines = []
    for _, row in iter_df.iterrows():
        if not np.isnan(row[slope_col]):
            lines.append(row[slope_col] * x_log_common + row[intercept_col])

    if not lines:
        return None

    lines = np.array(lines)
    y_mean = np.mean(lines, axis=0)
    y_std  = np.std(lines,  axis=0)

    slopes = -iter_df[slope_col].dropna()
    intercepts = iter_df[intercept_col].dropna()
    k_vals = (10**intercepts) * (d_min_cm**(-slopes))

    return {
        'y_mean': y_mean,
        'y_std':  y_std,
        'mean_r': float(np.mean(slopes)),
        'std_r':  float(np.std(slopes)),
        'mean_k': float(np.mean(k_vals)),
        'std_k':  float(np.std(k_vals)),
    }


def plot_csfd_comparison_stats(iter_df, out_prefix, r_true, total_rocks, d_start, d_end):
    """
    密度正規化 RSFD 比較プロット
    - True fit  : 全深さ範囲 (0-12 m) の全岩石、area = AREA_SIZE_M2
    - Apparent  : 深さ範囲内の検出岩石、area = range_area
    evaluate_apparent_RSFD_relative.py の plot_csfd_stats と同様式
    """
    x_common, x_log_common, d_min_cm, D_max_auto_cm = _build_common_xaxis(r_true, total_rocks)

    true_band = _fit_band(iter_df, 'slope_true_full', 'intercept_true_full', x_log_common, d_min_cm)
    det_band  = _fit_band(iter_df, 'slope_det',       'intercept_det',       x_log_common, d_min_cm)

    plt.figure(figsize=(10, 8))

    if true_band:
        plt.plot(x_common, 10**true_band['y_mean'], 'k--', linewidth=2.0,
                 label=(f"True fit (All range): "
                        f"r={true_band['mean_r']:.2f} $\\pm$ {true_band['std_r']:.2f}, "
                        f"k={true_band['mean_k']:.2e} $\\pm$ {true_band['std_k']:.2e}"))
        plt.fill_between(x_common,
                         10**(true_band['y_mean'] - true_band['y_std']),
                         10**(true_band['y_mean'] + true_band['y_std']),
                         color='gray', alpha=0.3)

    if det_band:
        plt.plot(x_common, 10**det_band['y_mean'], 'r-', linewidth=2.5,
                 label=(f"Apparent fit ({d_start}-{d_end} m): "
                        f"r={det_band['mean_r']:.2f} $\\pm$ {det_band['std_r']:.2f}, "
                        f"k={det_band['mean_k']:.2e} $\\pm$ {det_band['std_k']:.2e}"))
        plt.fill_between(x_common,
                         10**(det_band['y_mean'] - det_band['y_std']),
                         10**(det_band['y_mean'] + det_band['y_std']),
                         color='red', alpha=0.3)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Diameter [cm]', fontsize=18)
    plt.xticks([6, 10, 20, 50, 100, 200, 500], [6, 10, 20, 50, 100, 200, 500], fontsize=16)
    plt.xlim(d_min_cm * 0.8, D_max_auto_cm * 1.2)
    plt.ylabel('Cumulative Rock Density N(>D) [1/m²]', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, which='both', ls='-', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png")
    plt.savefig(f"{out_prefix}.pdf")
    plt.close()


def plot_csfd_count_stats(iter_df, out_prefix, r_true, total_rocks, d_start, d_end):
    """
    個数正規化 RSFD 比較プロット (area = 1.0)
    - True fit  : 全深さ範囲 (0-12 m) の全岩石
    - Apparent  : 深さ範囲内の検出岩石
    evaluate_apparent_RSFD_relative.py の plot_csfd_count_stats と同様式
    """
    x_common, x_log_common, d_min_cm, D_max_auto_cm = _build_common_xaxis(r_true, total_rocks)

    true_band = _fit_band(iter_df, 'slope_true_full_cnt', 'intercept_true_full_cnt', x_log_common, d_min_cm)
    det_band  = _fit_band(iter_df, 'slope_det_cnt',       'intercept_det_cnt',       x_log_common, d_min_cm)

    plt.figure(figsize=(10, 8))

    if true_band:
        plt.plot(x_common, 10**true_band['y_mean'], 'k--', linewidth=2.0,
                 label=(f"True fit (All range): "
                        f"r={true_band['mean_r']:.2f} $\\pm$ {true_band['std_r']:.2f}, "
                        f"k={true_band['mean_k']:.2e} $\\pm$ {true_band['std_k']:.2e}"))
        plt.fill_between(x_common,
                         10**(true_band['y_mean'] - true_band['y_std']),
                         10**(true_band['y_mean'] + true_band['y_std']),
                         color='gray', alpha=0.3)

    if det_band:
        plt.plot(x_common, 10**det_band['y_mean'], 'r-', linewidth=2.5,
                 label=(f"Apparent fit ({d_start}-{d_end} m): "
                        f"r={det_band['mean_r']:.2f} $\\pm$ {det_band['std_r']:.2f}, "
                        f"k={det_band['mean_k']:.2e} $\\pm$ {det_band['std_k']:.2e}"))
        plt.fill_between(x_common,
                         10**(det_band['y_mean'] - det_band['y_std']),
                         10**(det_band['y_mean'] + det_band['y_std']),
                         color='red', alpha=0.3)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Diameter [cm]', fontsize=18)
    plt.xticks([6, 10, 20, 50, 100, 200, 500], [6, 10, 20, 50, 100, 200, 500], fontsize=16)
    plt.xlim(d_min_cm * 0.8, D_max_auto_cm * 1.2)
    plt.ylabel('Cumulative Rock Count N(>D)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, which='both', ls='-', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png")
    plt.savefig(f"{out_prefix}.pdf")
    plt.close()


def main():
    print("--- 深さ範囲限定 RSFD 再解析ツール ---")

    try:
        d_start = float(input("解析する深さ範囲の開始値 [m] を入力してください (例: 0.0): "))
        d_end   = float(input("解析する深さ範囲の終了値 [m] を入力してください (例: 3.0): "))
    except ValueError:
        print("数値の入力ではありません。終了します。")
        return

    if d_start >= d_end:
        print("エラー: 開始値が終了値以上です。終了します。")
        return
    if d_end > MAX_DEPTH:
        print(f"エラー: 終了値がMAX_DEPTH ({MAX_DEPTH} m) を超えています。終了します。")
        return

    area = (d_end - d_start) / MAX_DEPTH * AREA_SIZE_M2
    range_label = f"{d_start}-{d_end}m"
    analysis_dir = os.path.join(BASE_DIR, f"analysis_{range_label}")
    json_path = os.path.join(BASE_DIR, f"rsfd_summary_{range_label}.json")

    print(f"\n深さ範囲 : {d_start} - {d_end} m")
    print(f"解析面積 : {area:.2f} m2")
    print(f"出力先   : {analysis_dir}")
    print(f"JSON     : {json_path}\n")

    r_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "r_*")))
    if not r_dirs:
        print(f"エラー: {BASE_DIR} 内に r_* ディレクトリが見つかりません。")
        return

    summary_data = {}

    for r_dir in r_dirs:
        r_str = os.path.basename(r_dir).replace("r_", "")
        try:
            r_true = float(r_str)
        except ValueError:
            continue

        r_key = str(r_true)
        summary_data[r_key] = {}

        n_dirs = sorted(glob.glob(os.path.join(r_dir, "*")))
        for n_dir in n_dirs:
            n_basename = os.path.basename(n_dir)
            try:
                total_rocks = int(n_basename)
            except ValueError:
                continue

            n_key = str(total_rocks)

            csv_files = sorted(
                glob.glob(os.path.join(n_dir, "each_iteration", "iter_*", "simulated_detection.csv"))
            )

            if not csv_files:
                print(f"  [スキップ] r={r_true}, N={total_rocks}: CSVが見つかりません")
                summary_data[r_key][n_key] = "Calculation stop"
                continue

            print(f"  処理中: r={r_true}, N={total_rocks} ({len(csv_files)} イテレーション)")

            out_dir = os.path.join(analysis_dir, f"r_{r_true}", str(total_rocks))

            result, iter_df = analyze_one_combination(
                total_rocks, csv_files, d_start, d_end, area, out_dir
            )
            summary_data[r_key][n_key] = result

            plot_csfd_comparison_stats(
                iter_df, os.path.join(out_dir, 'RSFD_comparison_stats'),
                r_true, total_rocks, d_start, d_end
            )
            plot_csfd_count_stats(
                iter_df, os.path.join(out_dir, 'RSFD_count_stats'),
                r_true, total_rocks, d_start, d_end
            )

            print(f"    r_apparent = {result['r_apparent_mean']:.4f} +/- {result['r_apparent_std']:.4f}")
            print(f"    k_apparent = {result['k_apparent_mean']:.4e} +/- {result['k_apparent_std']:.4e}")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)

    print(f"\n全処理完了。JSONを保存しました: {json_path}")


if __name__ == "__main__":
    main()
