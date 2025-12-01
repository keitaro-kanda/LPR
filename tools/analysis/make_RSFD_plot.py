#!/usr/bin/env python3
# RSFD_generator.py
# ------------------------------------------------------------
# ラベル JSON から
#   1) ラベル1→1 cm, ラベル2→6 cm, ラベル3→式で計算
# の岩石サイズを取得し，
# 線形‑線形の累積サイズ‑頻度分布 (個数) を描画・保存し，
# べき則／指数関数フィッティングおよび比較プロットを滑らかに追加
# ------------------------------------------------------------

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from matplotlib.ticker import MultipleLocator

# ------------------------------------------------------------------
# 補助関数定義
# ------------------------------------------------------------------
def none_to_nan(v):
    """None値をnp.nanに変換"""
    return np.nan if v is None else v

def format_p_value(p):
    """p値のフォーマットを補助する"""
    if p < 0.001:
        return "p < 0.001"
    else:
        return f"p={p:.3f}"

def save_legend_info_to_txt(output_path, legend_entries):
    """
    Legend情報をTXTファイルに保存

    Parameters:
    -----------
    output_path : str
        出力パス（拡張子なし）
    legend_entries : list of dict
        Legendエントリのリスト [{'label': str, 'color': str, 'linestyle': str}, ...]
    """
    with open(f'{output_path}_legend.txt', 'w', encoding='utf-8') as f:
        f.write('# Legend Information\n')
        f.write('# -------------------\n')
        for entry in legend_entries:
            f.write(f"Label: {entry['label']}\n")
            if 'color' in entry:
                f.write(f"  Color: {entry['color']}\n")
            if 'linestyle' in entry:
                f.write(f"  Linestyle: {entry['linestyle']}\n")
            if 'marker' in entry:
                f.write(f"  Marker: {entry['marker']}\n")
            f.write('\n')
    print(f'Legend情報保存: {output_path}_legend.txt')

def save_legend_only_pdf(output_path, legend_entries):
    """
    Legend専用のPDFファイルを作成

    Parameters:
    -----------
    output_path : str
        出力パス（拡張子なし）
    legend_entries : list of dict
        Legendエントリのリスト [{'label': str, 'color': str, 'linestyle': str, 'marker': str}, ...]
    """
    fig = plt.figure(figsize=(8, len(legend_entries) * 0.5 + 1))
    ax = fig.add_subplot(111)

    # 空のプロットを作成し、legend用のハンドルを生成
    handles = []
    labels = []
    for entry in legend_entries:
        label = entry['label']
        color = entry.get('color', 'black')
        linestyle = entry.get('linestyle', '-')
        marker = entry.get('marker', '')
        linewidth = entry.get('linewidth', 1.5)

        # ダミーのプロット（表示されない）
        if marker and linestyle and linestyle != '':
            line, = ax.plot([], [], color=color, linestyle=linestyle,
                           marker=marker, linewidth=linewidth, label=label)
        elif marker:
            line, = ax.plot([], [], color=color, marker=marker,
                           linestyle='', label=label)
        else:
            line, = ax.plot([], [], color=color, linestyle=linestyle,
                           linewidth=linewidth, label=label)
        handles.append(line)
        labels.append(label)

    # 軸を非表示にする
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Legendのみを表示
    legend = ax.legend(handles, labels, loc='center', fontsize=14,
                      frameon=True, fancybox=True, shadow=False)

    plt.tight_layout()
    plt.savefig(f'{output_path}_legend.pdf', dpi=600, bbox_inches='tight')
    plt.close()

    print(f'Legend専用PDF保存: {output_path}_legend.pdf')

def create_rsfd_plot(x_data, y_data, xlabel, ylabel, output_path,
                     scale_type='linear', fit_lines=None,
                     show_plot=False, dpi_png=300, dpi_pdf=600,
                     marker='o', linestyle='-', linewidth=1.5, color=None, label=None,
                     xlim=None):
    """
    RSFDプロットを作成・保存する汎用関数

    Parameters:
    -----------
    x_data, y_data : array
        プロットするデータ
    xlabel, ylabel : str
        軸ラベル
    output_path : str
        出力パス（拡張子なし）
    scale_type : str
        'linear', 'semilog', 'loglog'
    fit_lines : list of dict, optional
        フィット曲線のリスト [{'x': x, 'y': y, 'label': label, 'color': color, 'linestyle': style}, ...]
    show_plot : bool
        プロット表示の有無
    dpi_png, dpi_pdf : int
        解像度
    marker, linestyle, linewidth, color, label :
        データプロットのスタイル設定
    xlim : tuple, optional
        x軸範囲 (xmin, xmax)
    """
    plt.figure(figsize=(10, 8))

    # データプロット
    if marker and linestyle:
        plot_kwargs = {'marker': marker, 'linestyle': linestyle, 'linewidth': linewidth}
        if color:
            plot_kwargs['color'] = color
        if label:
            plot_kwargs['label'] = label
        plt.plot(x_data, y_data, **plot_kwargs)
    elif marker:  # scatter plot
        scatter_kwargs = {'marker': marker}
        if color:
            scatter_kwargs['color'] = color
        if label:
            scatter_kwargs['label'] = label
        plt.scatter(x_data, y_data, **scatter_kwargs)

    # フィット曲線の追加
    if fit_lines:
        for fit_line in fit_lines:
            plt.plot(fit_line['x'], fit_line['y'],
                    linestyle=fit_line.get('linestyle', '--'),
                    linewidth=fit_line.get('linewidth', 1.5),
                    color=fit_line.get('color', 'red'),
                    label=fit_line.get('label', ''))

    # 軸スケール設定
    if scale_type == 'semilog':
        plt.yscale('log')
    elif scale_type == 'loglog':
        plt.xscale('log')
        plt.yscale('log')

    # x軸範囲の設定（軸スケール設定の直後に実行）
    if xlim and scale_type == 'loglog':
        plt.xlim(max(xlim[0], 0.5), xlim[1])  # logスケールで負またはゼロを避ける
    elif xlim:
        plt.xlim(xlim)

    # 軸ラベルとグリッド
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(labelsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)

    # y軸のtick設定（最大値1-20の場合は2刻みに固定）
    ax = plt.gca()
    ylim = ax.get_ylim()
    if 1 <= ylim[1] <= 20:
        ax.yaxis.set_major_locator(MultipleLocator(2))

    plt.tight_layout()

    # 保存
    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)

    # 表示
    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f'プロット保存: {output_path}.png')

    # Legend情報の保存（ラベルがある場合のみ）
    if label or fit_lines:
        legend_entries = []

        # データラベル
        if label:
            legend_entries.append({
                'label': label,
                'color': color if color else 'black',
                'linestyle': linestyle if linestyle else '',
                'marker': marker if marker else ''
            })

        # フィット曲線ラベル
        if fit_lines:
            for fit_line in fit_lines:
                legend_entries.append({
                    'label': fit_line.get('label', ''),
                    'color': fit_line.get('color', 'red'),
                    'linestyle': fit_line.get('linestyle', '--'),
                    'linewidth': fit_line.get('linewidth', 1.5)
                })

        # TXT形式で保存
        save_legend_info_to_txt(output_path, legend_entries)

        # Legend専用PDF出力
        save_legend_only_pdf(output_path, legend_entries)

def calc_fitting(sizes, counts):
    """べき則と指数関数のフィッティングを実行"""
    # 対数変換
    mask = sizes > 0
    log_D = np.log(sizes[mask])
    log_N = np.log(counts[mask])

    # べき則フィッティング (Power-law: log N = r log D + log k)
    X_pow = sm.add_constant(log_D)
    model_pow = sm.OLS(log_N, X_pow)
    results_pow = model_pow.fit()

    log_k_pow, r_pow = results_pow.params
    k_pow = np.exp(log_k_pow)
    R2_pow = results_pow.rsquared
    r_pow_se = results_pow.bse[1]
    r_pow_t = results_pow.tvalues[1]
    r_pow_p = results_pow.pvalues[1]
    dof_pow = results_pow.df_resid
    n_pow = int(results_pow.nobs)

    # 指数関数フィッティング (Exponential: log N = rD + log k)
    X_exp = sm.add_constant(sizes[mask])
    model_exp = sm.OLS(log_N, X_exp)
    results_exp = model_exp.fit()

    log_k_exp, r_exp = results_exp.params
    k_exp = np.exp(log_k_exp)
    R2_exp = results_exp.rsquared
    r_exp_se = results_exp.bse[1]
    r_exp_t = results_exp.tvalues[1]
    r_exp_p = results_exp.pvalues[1]
    dof_exp = results_exp.df_resid
    n_exp = int(results_exp.nobs)

    # フィット曲線用に滑らかなサンプル点を生成
    D_fit = np.linspace(sizes.min(), sizes.max(), 200)
    N_pow_fit = k_pow * D_fit**r_pow
    N_exp_fit = k_exp * np.exp(r_exp * D_fit)

    # べき則の結果, 指数関数の結果, D_fit
    return (k_pow, np.abs(r_pow), R2_pow, N_pow_fit, r_pow_t, r_pow_p, r_pow_se, n_pow, dof_pow), \
           (k_exp, np.abs(r_exp), R2_exp, N_exp_fit, r_exp_t, r_exp_p, r_exp_se, n_exp, dof_exp), \
           D_fit

def calc_fitting_area_normalized(sizes, counts, area):
    """面積規格化されたべき則フィッティングを実行"""
    # 岩石数を面積で規格化
    counts_normalized = counts / area

    # 対数変換
    mask = sizes > 0
    log_D = np.log(sizes[mask])
    log_N = np.log(counts_normalized[mask])

    # べき則フィッティング (Power-law: log N = r log D + log k)
    X_pow = sm.add_constant(log_D)
    model_pow = sm.OLS(log_N, X_pow)
    results_pow = model_pow.fit()

    log_k_pow, r_pow = results_pow.params
    k_pow = np.exp(log_k_pow)
    R2_pow = results_pow.rsquared
    r_pow_se = results_pow.bse[1]
    r_pow_t = results_pow.tvalues[1]
    r_pow_p = results_pow.pvalues[1]
    dof_pow = results_pow.df_resid
    n_pow = int(results_pow.nobs)

    # フィット曲線用に滑らかなサンプル点を生成
    D_fit = np.linspace(sizes.min(), sizes.max(), 200)
    N_pow_fit = k_pow * D_fit**r_pow

    # べき則の結果, D_fit, 規格化されたカウント
    return (k_pow, np.abs(r_pow), R2_pow, N_pow_fit, r_pow_t, r_pow_p, r_pow_se, n_pow, dof_pow), \
           D_fit, counts_normalized

def save_processing_config(config_path, data_path, mode, time_range, horizontal_range,
                           time_min, time_max, horizontal_min, horizontal_max, output_dir,
                           num_ranges=None, ranges_list=None, area=None):
    """処理設定をJSONファイルに保存"""
    # 既存の設定ファイルを読み込み
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {
            "label_file": data_path,
            "processing_history": []
        }

    # 新しい処理記録を追加
    new_record = {
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
        "area": area
    }

    # モード4の場合は範囲リストも保存
    if mode == '4':
        new_record["num_ranges"] = num_ranges
        new_record["ranges_list"] = ranges_list

    config["processing_history"].append(new_record)

    # JSONファイルに保存
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f'処理設定を保存: {config_path}')

def load_all_configs():
    """設定ファイルを読み込み、全ての処理履歴を返す"""
    print('\n検出された岩石のラベルデータファイル(.json)のパスを入力してください:')
    data_path = input().strip()
    if not (os.path.exists(data_path) and data_path.lower().endswith('.json')):
        raise FileNotFoundError('正しい .json ファイルを指定してください。')

    # 設定ファイルのパス
    base_dir = os.path.join(os.path.dirname(os.path.dirname(data_path)), 'RSFD')
    config_path = os.path.join(base_dir, 'processing_config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'設定ファイルが見つかりません: {config_path}\n先に新規データ処理を実行してください。')

    # 設定ファイルを読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if not config["processing_history"]:
        raise ValueError('処理履歴が空です。')

    # 処理履歴を表示
    print('\n=== 処理履歴 ===')
    for record in config["processing_history"]:
        print(f'ID: {record["id"]}')
        print(f'  日時: {record["timestamp"]}')
        print(f'  モード: {record["mode"]}')
        if record["mode"] == '4':
            print(f'  範囲数: {record.get("num_ranges", "不明")}')
            if "ranges_list" in record:
                for i, r in enumerate(record["ranges_list"], 1):
                    exclude_str = " (除外)" if r.get("exclude", False) else ""
                    print(f'  範囲{i}: 時間={r.get("time_range", "指定なし")}, 水平位置={r.get("horizontal_range", "指定なし")}{exclude_str}')
        else:
            print(f'  時間範囲: {record["time_range"] if record["time_range"] else "指定なし"}')
            print(f'  水平位置範囲: {record["horizontal_range"] if record["horizontal_range"] else "指定なし"}')
        print(f'  出力ディレクトリ: {record["output_dir"]}')
        print()

    # 全ての処理履歴を返す
    return data_path, config["processing_history"]

def create_multi_range_comparison_plot(ranges_data_list, xlabel, ylabel, output_path,
                                       scale_type='linear', fit_type='power',
                                       show_plot=False, dpi_png=300, dpi_pdf=600,
                                       xlim=None):
    """
    複数範囲のRSFDを1つのプロットに重ねて表示する関数

    Parameters:
    -----------
    ranges_data_list : list of dict
        各範囲のデータリスト。各要素は以下のキーを持つ辞書:
        {
            'x_data': array, 'y_data': array,
            'label': str, 'color': str,
            'fit_x': array, 'fit_y': array, 'fit_params': dict
        }
    xlabel, ylabel : str
        軸ラベル
    output_path : str
        出力パス（拡張子なし）
    scale_type : str
        'linear', 'semilog', 'loglog'
    fit_type : str
        'power' or 'exponential'
    show_plot : bool
        プロット表示の有無
    dpi_png, dpi_pdf : int
        解像度
    xlim : tuple, optional
        x軸範囲 (xmin, xmax)
    """
    plt.figure(figsize=(10, 8))

    # 各範囲のデータとフィット曲線をプロット
    for range_data in ranges_data_list:
        # データプロット
        plt.plot(range_data['x_data'], range_data['y_data'],
                marker='o', linestyle='',
                color=range_data['color'],
                label=f"{range_data['label']} (Data)")

        # フィット曲線プロット
        plt.plot(range_data['fit_x'], range_data['fit_y'],
                linestyle='--', linewidth=1.5,
                color=range_data['color'],
                label=f"{range_data['label']} ({fit_type}: k={range_data['fit_params']['k']:.2e}, "
                      f"r={range_data['fit_params']['r']:.3f}, R²={range_data['fit_params']['R2']:.4f}, "
                      f"{range_data['fit_params']['p_str']})")

    # 軸スケール設定
    if scale_type == 'semilog':
        plt.yscale('log')
    elif scale_type == 'loglog':
        plt.xscale('log')
        plt.yscale('log')

    # x軸範囲の設定
    if xlim and scale_type == 'loglog':
        plt.xlim(max(xlim[0], 0.5), xlim[1])
    elif xlim:
        plt.xlim(xlim)

    # 軸ラベルとグリッド
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(labelsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)

    # y軸のtick設定（最大値1-20の場合は2刻みに固定）
    ax = plt.gca()
    ylim = ax.get_ylim()
    if 1 <= ylim[1] <= 20:
        ax.yaxis.set_major_locator(MultipleLocator(2))

    plt.tight_layout()

    # 保存
    plt.savefig(f'{output_path}.png', dpi=dpi_png)
    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)

    # 表示
    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f'プロット保存: {output_path}.png')

    # Legend情報の保存
    legend_entries = []
    for range_data in ranges_data_list:
        # データプロットのエントリ
        legend_entries.append({
            'label': f"{range_data['label']} (Data)",
            'color': range_data['color'],
            'marker': 'o',
            'linestyle': ''
        })
        # フィット曲線のエントリ
        legend_entries.append({
            'label': f"{range_data['label']} ({fit_type}: k={range_data['fit_params']['k']:.2e}, "
                     f"r={range_data['fit_params']['r']:.3f}, R²={range_data['fit_params']['R2']:.4f}, "
                     f"{range_data['fit_params']['p_str']})",
            'color': range_data['color'],
            'linestyle': '--',
            'linewidth': 1.5
        })

    # TXT形式で保存
    save_legend_info_to_txt(output_path, legend_entries)

    # Legend専用PDF出力
    save_legend_only_pdf(output_path, legend_entries)

# ------------------------------------------------------------------
# 1. 起動モード選択
# ------------------------------------------------------------------
print('=== RSFD Plot Generator ===')
print('1: 新規データ処理')
print('2: 記録ファイルから既存プロットを再作成')
startup_mode = input('モードを選択してください (1/2): ').strip()

if startup_mode not in ['1', '2']:
    raise ValueError('モードは1または2を選択してください。')

# ------------------------------------------------------------------
# 2. 入力ファイルチェック（モード1の場合）
# ------------------------------------------------------------------
if startup_mode == '1':
    print('\n検出された岩石のラベルデータファイル(.json)のパスを入力してください:')
    data_path = input().strip()
    if not (os.path.exists(data_path) and data_path.lower().endswith('.json')):
        raise FileNotFoundError('正しい .json ファイルを指定してください。')

    # モード選択
    print('\n=== データ範囲モード選択 ===')
    print('1: 全範囲のデータを使用')
    print('2: 特定の時間・距離範囲のデータのみを使用')
    print('3: 特定の時間・距離範囲のデータのみを取り除いて使用')
    print('4: 複数の範囲を比較（比較プロットのみ出力）')
    mode = input('モードを選択してください (1/2/3/4): ').strip()

    if mode not in ['1', '2', '3', '4']:
        raise ValueError('モードは1, 2, 3, 4のいずれかを選択してください。')

    # データ範囲の入力
    if mode == '1':
        # モード1: 全範囲使用
        time_range = ''
        horizontal_range = ''
        area = 16136  # 固定値
        print('全範囲のデータを使用します。')
        print(f'面積: {area} m²（固定値）')
    elif mode == '4':
        # モード4: 複数範囲比較
        print('\n=== 複数範囲比較モード ===')
        num_ranges = int(input('比較する範囲の数を入力してください: ').strip())
        if num_ranges < 1:
            raise ValueError('範囲の数は1以上を指定してください。')

        ranges_list = []
        for i in range(num_ranges):
            print(f'\n--- 範囲 {i+1} の設定 ---')
            print('この範囲をどう使用しますか？')
            print('  1: この範囲のデータのみを使用')
            print('  2: この範囲のデータを除外')
            range_mode = input('選択 (1/2): ').strip()

            if range_mode not in ['1', '2']:
                raise ValueError('選択は1または2を入力してください。')

            exclude_flag = (range_mode == '2')

            time_range_input = input('時間範囲 [ns] を入力してください（例: 50-100, Enter: 指定なし）: ').strip()
            horizontal_range_input = input('水平位置範囲 [m] を入力してください（例: 0-100, Enter: 指定なし）: ').strip()

            try:
                if time_range_input:
                    t_min, t_max = map(float, time_range_input.split('-'))
                else:
                    t_min, t_max = None, None

                if horizontal_range_input:
                    h_min, h_max = map(float, horizontal_range_input.split('-'))
                else:
                    h_min, h_max = None, None
            except ValueError:
                raise ValueError(f'範囲{i+1}の入力形式が正しくありません。例: 0-100')

            # 面積入力
            area_input = input('面積 [m²] を入力してください: ').strip()
            try:
                area_value = float(area_input)
            except ValueError:
                raise ValueError(f'範囲{i+1}の面積の入力形式が正しくありません。数値を入力してください。')

            ranges_list.append({
                'time_range': time_range_input,
                'horizontal_range': horizontal_range_input,
                'time_min': t_min,
                'time_max': t_max,
                'horizontal_min': h_min,
                'horizontal_max': h_max,
                'exclude': exclude_flag,
                'area': area_value
            })

        # モード4用の変数（後で使用）
        time_range = ''
        horizontal_range = ''
        time_min, time_max = None, None
        horizontal_min, horizontal_max = None, None
    else:
        # モード2/3: 範囲入力
        print('\n=== データ範囲指定 ===')
        if mode == '2':
            print('指定した範囲のデータのみを使用します。')
        else:  # mode == '3'
            print('指定した範囲のデータを除外します。')
        time_range = input('時間範囲 [ns] を入力してください（例: 50-100, Enter: 指定なし）: ').strip()
        horizontal_range = input('水平位置範囲 [m] を入力してください（例: 0-100, Enter: 指定なし）: ').strip()

        # 面積入力
        area_input = input('面積 [m²] を入力してください: ').strip()
        try:
            area = float(area_input)
        except ValueError:
            raise ValueError('面積の入力形式が正しくありません。数値を入力してください。')

    if mode != '4':
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

    # 出力フォルダ
    base_dir = os.path.join(os.path.dirname(os.path.dirname(data_path)), 'RSFD')
    file_name = os.path.splitext(os.path.basename(data_path))[0]

    # 範囲指定に応じた出力ディレクトリ名
    if mode == '1':
        # モード1: 全範囲
        output_dir = os.path.join(base_dir, f'{file_name}_full_range')
    elif mode == '2':
        # モード2: 特定範囲のみ使用（既存の命名）
        if time_range and not horizontal_range:
            output_dir = os.path.join(base_dir, f'{file_name}_t{time_min}-{time_max}')
        elif horizontal_range and not time_range:
            output_dir = os.path.join(base_dir, f'{file_name}_x{horizontal_min}-{horizontal_max}')
        elif time_range and horizontal_range:
            output_dir = os.path.join(base_dir, f'{file_name}_t{time_min}-{time_max}_x{horizontal_min}-{horizontal_max}')
        else:
            output_dir = os.path.join(base_dir, f'{file_name}_full_range')
    elif mode == '3':
        # モード3: 特定範囲を除外
        if time_range and not horizontal_range:
            output_dir = os.path.join(base_dir, f'{file_name}_remove_t{time_min}-{time_max}')
        elif horizontal_range and not time_range:
            output_dir = os.path.join(base_dir, f'{file_name}_remove_x{horizontal_min}-{horizontal_max}')
        elif time_range and horizontal_range:
            output_dir = os.path.join(base_dir, f'{file_name}_remove_t{time_min}-{time_max}_x{horizontal_min}-{horizontal_max}')
        else:
            output_dir = os.path.join(base_dir, f'{file_name}_full_range')
    elif mode == '4':
        # モード4: 複数範囲比較（ユーザー指定のディレクトリ名）
        custom_dir_name = input('出力ディレクトリ名を入力してください: ').strip()
        if not custom_dir_name:
            raise ValueError('出力ディレクトリ名を指定してください。')
        output_dir = os.path.join(base_dir, custom_dir_name)

    os.makedirs(output_dir, exist_ok=True)

    if mode != '4':
        # モード1-3: Group2-3専用のディレクトリのみ作成
        # プロット用サブフォルダ（Group2-3専用）
        output_dir_power_2_3 = os.path.join(output_dir, '1_power_law_fit_2-3')
        os.makedirs(output_dir_power_2_3, exist_ok=True)

        # 面積規格化用サブフォルダ
        output_dir_area_normalized_est = os.path.join(output_dir, '4_area_normalized_fit_estimate_group2')
        output_dir_area_normalized_2_3 = os.path.join(output_dir, '5_area_normalized_fit_group2-3')
        os.makedirs(output_dir_area_normalized_est, exist_ok=True)
        os.makedirs(output_dir_area_normalized_2_3, exist_ok=True)

        # Group比較用サブフォルダ
        output_dir_group_comparison = os.path.join(output_dir, '6_group_comparison')
        output_dir_group_comparison_area = os.path.join(output_dir, '7_group_comparison_area_normalized')
        os.makedirs(output_dir_group_comparison, exist_ok=True)
        os.makedirs(output_dir_group_comparison_area, exist_ok=True)
    else:
        # モード4: Group2-3専用の比較プロット用ディレクトリのみ作成
        # Group2-3専用ディレクトリ
        output_dir_power_2_3 = os.path.join(output_dir, '1_power_law_comparison_2-3')
        os.makedirs(output_dir_power_2_3, exist_ok=True)

        # 面積規格化用ディレクトリ
        output_dir_area_normalized_est = os.path.join(output_dir, '4_area_normalized_comparison_estimate_group2')
        output_dir_area_normalized_2_3 = os.path.join(output_dir, '5_area_normalized_comparison_group2-3')
        os.makedirs(output_dir_area_normalized_est, exist_ok=True)
        os.makedirs(output_dir_area_normalized_2_3, exist_ok=True)

        # Group比較用ディレクトリ
        output_dir_group_comparison = os.path.join(output_dir, '6_group_comparison')
        output_dir_group_comparison_area = os.path.join(output_dir, '7_group_comparison_area_normalized')
        os.makedirs(output_dir_group_comparison, exist_ok=True)
        os.makedirs(output_dir_group_comparison_area, exist_ok=True)

else:  # startup_mode == '2'
    # 設定ファイルから全ての処理履歴を読み込み
    data_path, all_records = load_all_configs()

    print(f'\n全ての処理履歴（{len(all_records)}件）を順番に再作成します。\n')

    # 各処理履歴をループ処理
    for record_idx, selected_record in enumerate(all_records, 1):
        print(f'=== 処理 {record_idx}/{len(all_records)}: ID {selected_record["id"]} ===')

        try:
            # 変数を復元
            mode = selected_record['mode']
            time_range = selected_record['time_range']
            horizontal_range = selected_record['horizontal_range']
            time_min = selected_record['time_min']
            time_max = selected_record['time_max']
            horizontal_min = selected_record['horizontal_min']
            horizontal_max = selected_record['horizontal_max']
            output_dir = selected_record['output_dir']
            area = selected_record.get('area', 16136)  # デフォルト値を設定

            # base_dirとfile_nameを復元
            base_dir = os.path.dirname(output_dir)
            file_name = os.path.splitext(os.path.basename(data_path))[0]

            print(f'データファイル: {data_path}')
            print(f'モード: {mode}')
            print(f'時間範囲: {time_range if time_range else "指定なし"}')
            print(f'水平位置範囲: {horizontal_range if horizontal_range else "指定なし"}')
            print(f'出力ディレクトリ: {output_dir}')

            # 出力ディレクトリを再作成
            os.makedirs(output_dir, exist_ok=True)

            if mode != '4':
                # モード1-3: Group2-3専用のディレクトリのみ作成
                # プロット用サブフォルダ（Group2-3専用）
                output_dir_power_2_3 = os.path.join(output_dir, '1_power_law_fit_2-3')
                os.makedirs(output_dir_power_2_3, exist_ok=True)

                # 面積規格化用サブフォルダ
                output_dir_area_normalized_est = os.path.join(output_dir, '4_area_normalized_fit_estimate_group2')
                output_dir_area_normalized_2_3 = os.path.join(output_dir, '5_area_normalized_fit_group2-3')
                os.makedirs(output_dir_area_normalized_est, exist_ok=True)
                os.makedirs(output_dir_area_normalized_2_3, exist_ok=True)

                # Group比較用サブフォルダ
                output_dir_group_comparison = os.path.join(output_dir, '6_group_comparison')
                output_dir_group_comparison_area = os.path.join(output_dir, '7_group_comparison_area_normalized')
                os.makedirs(output_dir_group_comparison, exist_ok=True)
                os.makedirs(output_dir_group_comparison_area, exist_ok=True)
            else:
                # モード4: Group2-3専用の比較プロット用ディレクトリのみ作成
                # Group2-3専用ディレクトリ
                output_dir_power_2_3 = os.path.join(output_dir, '1_power_law_comparison_2-3')
                os.makedirs(output_dir_power_2_3, exist_ok=True)

                # 面積規格化用ディレクトリ
                output_dir_area_normalized_est = os.path.join(output_dir, '4_area_normalized_comparison_estimate_group2')
                output_dir_area_normalized_2_3 = os.path.join(output_dir, '5_area_normalized_comparison_group2-3')
                os.makedirs(output_dir_area_normalized_est, exist_ok=True)
                os.makedirs(output_dir_area_normalized_2_3, exist_ok=True)

                # Group比較用ディレクトリ
                output_dir_group_comparison = os.path.join(output_dir, '6_group_comparison')
                output_dir_group_comparison_area = os.path.join(output_dir, '7_group_comparison_area_normalized')
                os.makedirs(output_dir_group_comparison, exist_ok=True)
                os.makedirs(output_dir_group_comparison_area, exist_ok=True)

                # モード4用の変数を復元
                num_ranges = selected_record.get('num_ranges', 1)
                ranges_list = selected_record.get('ranges_list', [])

            # ------------------------------------------------------------------
            # 共通処理: JSON 読み込み
            # ------------------------------------------------------------------
            if mode == '4':
                # モード4の場合は、startup_mode == '1'のモード4処理と同じ処理を実行
                print('\n=== 複数範囲比較処理を開始 ===')

                # 色のリスト（各範囲に割り当て）
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
                          'olive', 'cyan', 'magenta', 'yellow', 'navy', 'teal', 'maroon']

                # 各範囲のデータを格納するリスト
                all_ranges_data_traditional = []
                all_ranges_data_estimate_grp2 = []
                all_ranges_data_grp2_3 = []
                all_ranges_data_area_normalized_est = []  # 面積規格化Group2推定
                all_ranges_data_area_normalized_2_3 = []  # 面積規格化Group2-3

                # JSONデータ読み込み（一度だけ）
                with open(data_path, 'r') as f:
                    results = json.load(f).get('results', {})

                x_all = np.array([v['x'] for v in results.values()])
                y_all = np.array([v['y'] for v in results.values()])
                lab_all = np.array([v['label'] for v in results.values()], dtype=int)
                time_top_all = np.array([none_to_nan(v['time_top']) for v in results.values()], dtype=float)
                time_bottom_all = np.array([none_to_nan(v['time_bottom']) for v in results.values()], dtype=float)
                print(f'ラベルデータ読み込み完了: {len(lab_all)}個')

                # 各範囲について処理
                for range_idx, range_info in enumerate(ranges_list):
                    print(f'\n--- 範囲 {range_idx + 1}/{num_ranges} の処理 ---')

                    # データをコピー
                    x = x_all.copy()
                    y = y_all.copy()
                    lab = lab_all.copy()
                    time_top = time_top_all.copy()
                    time_bottom = time_bottom_all.copy()

                    # データ範囲フィルタリング
                    original_count = len(lab)
                    time_min_range = range_info['time_min']
                    time_max_range = range_info['time_max']
                    horizontal_min_range = range_info['horizontal_min']
                    horizontal_max_range = range_info['horizontal_max']
                    exclude_flag = range_info.get('exclude', False)

                    if time_min_range is not None and time_max_range is not None:
                        mask_group1 = (lab == 1) & (y >= time_min_range) & (y <= time_max_range)
                        mask_others = (lab != 1) & (time_top >= time_min_range) & (time_top <= time_max_range)
                        time_mask = mask_group1 | mask_others

                        # 除外フラグが立っている場合は論理反転
                        if exclude_flag:
                            time_mask = ~time_mask

                        x = x[time_mask]
                        y = y[time_mask]
                        lab = lab[time_mask]
                        time_top = time_top[time_mask]
                        time_bottom = time_bottom[time_mask]

                        if exclude_flag:
                            print(f'時間範囲除外後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
                        else:
                            print(f'時間範囲フィルタリング後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')

                    if horizontal_min_range is not None and horizontal_max_range is not None:
                        horizontal_mask = (x >= horizontal_min_range) & (x <= horizontal_max_range)

                        # 除外フラグが立っている場合は論理反転
                        if exclude_flag:
                            horizontal_mask = ~horizontal_mask

                        x = x[horizontal_mask]
                        y = y[horizontal_mask]
                        lab = lab[horizontal_mask]
                        time_top = time_top[horizontal_mask]
                        time_bottom = time_bottom[horizontal_mask]

                        if exclude_flag:
                            print(f'水平位置範囲除外後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
                        else:
                            print(f'水平位置範囲フィルタリング後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')

                    print(f'フィルタリング完了: {len(lab)}個のデータを使用')

                    # サイズ配列を作成
                    counts = {k: int(np.sum(lab == k)) for k in range(1, 7)}
                    size_label1 = np.full(counts[1], 1.0)
                    size_label2 = np.full(counts[2], 6.0)
                    mask2_valid = (lab == 2) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
                    mask3_valid = (lab == 3) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
                    er = 9.0
                    c = 299_792_458
                    sizes_group2 = (time_bottom[mask2_valid] - time_top[mask2_valid]) * 1e-9 * c / np.sqrt(er) * 0.5 * 100
                    sizes_group3 = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(er) * 0.5 * 100

                    # サイズ値を小数点以下3桁に丸めて、浮動小数点の微小誤差を排除
                    sizes_group2 = np.round(sizes_group2, decimals=3)
                    sizes_group3 = np.round(sizes_group3, decimals=3)

                    # 1) 従来手法（Group2=6cm固定）
                    all_sizes_traditional = np.concatenate([size_label1, size_label2, sizes_group3])
                    unique_sizes_traditional, unique_counts = np.unique(all_sizes_traditional, return_counts=True)
                    unique_sizes_traditional = np.sort(unique_sizes_traditional)
                    cum_counts_traditional = np.array([np.sum(all_sizes_traditional >= s) for s in unique_sizes_traditional])

                    # フィッティング
                    (k_pow_trad, r_pow_trad, R2_pow_trad, N_pow_fit_trad, t_pow_trad, p_pow_trad, se_pow_trad, n_pow_trad, dof_pow_trad), \
                    (k_exp_trad, r_exp_trad, R2_exp_trad, N_exp_fit_trad, t_exp_trad, p_exp_trad, se_exp_trad, n_exp_trad, dof_exp_trad), \
                    D_fit_trad = calc_fitting(unique_sizes_traditional, cum_counts_traditional)

                    # 範囲ラベル作成
                    range_label_parts = []
                    if range_info['time_range']:
                        range_label_parts.append(f"t{time_min_range}-{time_max_range}")
                    if range_info['horizontal_range']:
                        range_label_parts.append(f"x{horizontal_min_range}-{horizontal_max_range}")

                    # 除外フラグに応じてラベルを作成
                    if range_label_parts:
                        base_label = f"Range {range_idx + 1}: {', '.join(range_label_parts)}"
                    else:
                        base_label = f"Range {range_idx + 1}: Full"

                    if exclude_flag:
                        range_label = f"{base_label} (exclude)"
                    else:
                        range_label = base_label

                    # データを保存
                    color = colors[range_idx % len(colors)]
                    all_ranges_data_traditional.append({
                        'x_data': unique_sizes_traditional,
                        'y_data': cum_counts_traditional,
                        'fit_x': D_fit_trad,
                        'fit_y': N_pow_fit_trad,
                        'fit_params': {
                            'k': k_pow_trad,
                            'r': r_pow_trad,
                            'R2': R2_pow_trad,
                            'p_str': format_p_value(p_pow_trad)
                        },
                        'label': range_label,
                        'color': color
                    })

                    all_ranges_data_traditional.append({
                        'x_data': unique_sizes_traditional,
                        'y_data': cum_counts_traditional,
                        'fit_x': D_fit_trad,
                        'fit_y': N_exp_fit_trad,
                        'fit_params': {
                            'k': k_exp_trad,
                            'r': r_exp_trad,
                            'R2': R2_exp_trad,
                            'p_str': format_p_value(p_exp_trad)
                        },
                        'label': range_label,
                        'color': color,
                        'fit_type': 'exponential'
                    })

                    # 2) Group2サイズ推定
                    all_sizes_estimate_group2 = np.concatenate([size_label1, sizes_group2, sizes_group3])
                    unique_sizes_estimate_group2 = np.sort(np.unique(all_sizes_estimate_group2))
                    cum_counts_estimate_group2 = np.array([np.sum(all_sizes_estimate_group2 >= s) for s in unique_sizes_estimate_group2])

                    (k_pow_est_grp2, r_pow_est_grp2, R2_pow_est_grp2, N_pow_fit_est_grp2, t_pow_est_grp2, p_pow_est_grp2, se_pow_est_grp2, n_pow_est_grp2, dof_pow_est_grp2), \
                    (k_exp_est_grp2, r_exp_est_grp2, R2_exp_est_grp2, N_exp_fit_est_grp2, t_exp_est_grp2, p_exp_est_grp2, se_exp_est_grp2, n_exp_est_grp2, dof_exp_est_grp2), \
                    D_fit_est_grp2 = calc_fitting(unique_sizes_estimate_group2, cum_counts_estimate_group2)

                    all_ranges_data_estimate_grp2.append({
                        'x_data': unique_sizes_estimate_group2,
                        'y_data': cum_counts_estimate_group2,
                        'fit_x': D_fit_est_grp2,
                        'fit_y': N_pow_fit_est_grp2,
                        'fit_params': {
                            'k': k_pow_est_grp2,
                            'r': r_pow_est_grp2,
                            'R2': R2_pow_est_grp2,
                            'p_str': format_p_value(p_pow_est_grp2)
                        },
                        'label': range_label,
                        'color': color
                    })

                    all_ranges_data_estimate_grp2.append({
                        'x_data': unique_sizes_estimate_group2,
                        'y_data': cum_counts_estimate_group2,
                        'fit_x': D_fit_est_grp2,
                        'fit_y': N_exp_fit_est_grp2,
                        'fit_params': {
                            'k': k_exp_est_grp2,
                            'r': r_exp_est_grp2,
                            'R2': R2_exp_est_grp2,
                            'p_str': format_p_value(p_exp_est_grp2)
                        },
                        'label': range_label,
                        'color': color,
                        'fit_type': 'exponential'
                    })

                    # 3) Group2-3のみ
                    all_sizes_group2_3 = np.concatenate([sizes_group2, sizes_group3])
                    unique_sizes_group2_3 = np.sort(np.unique(all_sizes_group2_3))
                    cum_counts_group2_3 = np.array([np.sum(all_sizes_group2_3 >= s) for s in unique_sizes_group2_3])

                    (k_pow_grp2_3, r_pow_grp2_3, R2_pow_grp2_3, N_pow_fit_grp2_3, t_pow_grp2_3, p_pow_grp2_3, se_pow_grp2_3, n_pow_grp2_3, dof_pow_grp2_3), \
                    (k_exp_grp2_3, r_exp_grp2_3, R2_exp_grp2_3, N_exp_fit_grp2_3, t_exp_grp2_3, p_exp_grp2_3, se_exp_grp2_3, n_exp_grp2_3, dof_exp_grp2_3), \
                    D_fit_grp2_3 = calc_fitting(unique_sizes_group2_3, cum_counts_group2_3)

                    all_ranges_data_grp2_3.append({
                        'x_data': unique_sizes_group2_3,
                        'y_data': cum_counts_group2_3,
                        'fit_x': D_fit_grp2_3,
                        'fit_y': N_pow_fit_grp2_3,
                        'fit_params': {
                            'k': k_pow_grp2_3,
                            'r': r_pow_grp2_3,
                            'R2': R2_pow_grp2_3,
                            'p_str': format_p_value(p_pow_grp2_3)
                        },
                        'label': range_label,
                        'color': color
                    })

                    all_ranges_data_grp2_3.append({
                        'x_data': unique_sizes_group2_3,
                        'y_data': cum_counts_group2_3,
                        'fit_x': D_fit_grp2_3,
                        'fit_y': N_exp_fit_grp2_3,
                        'fit_params': {
                            'k': k_exp_grp2_3,
                            'r': r_exp_grp2_3,
                            'R2': R2_exp_grp2_3,
                            'p_str': format_p_value(p_exp_grp2_3)
                        },
                        'label': range_label,
                        'color': color,
                        'fit_type': 'exponential'
                    })

                    # 4) 面積規格化 Group2推定
                    area_range = range_info.get('area', 16136)
                    (k_pow_area_est, r_pow_area_est, R2_pow_area_est, N_pow_fit_area_est,
                     t_pow_area_est, p_pow_area_est, se_pow_area_est, n_pow_area_est, dof_pow_area_est), \
                    D_fit_area_est, cum_counts_area_est = calc_fitting_area_normalized(
                        unique_sizes_estimate_group2, cum_counts_estimate_group2, area_range)

                    all_ranges_data_area_normalized_est.append({
                        'x_data': unique_sizes_estimate_group2,
                        'y_data': cum_counts_area_est,
                        'fit_x': D_fit_area_est,
                        'fit_y': N_pow_fit_area_est,
                        'fit_params': {
                            'k': k_pow_area_est,
                            'r': r_pow_area_est,
                            'R2': R2_pow_area_est,
                            'p_str': format_p_value(p_pow_area_est),
                            'area': area_range
                        },
                        'label': range_label,
                        'color': color
                    })

                    # 5) 面積規格化 Group2-3
                    (k_pow_area_2_3, r_pow_area_2_3, R2_pow_area_2_3, N_pow_fit_area_2_3,
                     t_pow_area_2_3, p_pow_area_2_3, se_pow_area_2_3, n_pow_area_2_3, dof_pow_area_2_3), \
                    D_fit_area_2_3, cum_counts_area_2_3 = calc_fitting_area_normalized(
                        unique_sizes_group2_3, cum_counts_group2_3, area_range)

                    all_ranges_data_area_normalized_2_3.append({
                        'x_data': unique_sizes_group2_3,
                        'y_data': cum_counts_area_2_3,
                        'fit_x': D_fit_area_2_3,
                        'fit_y': N_pow_fit_area_2_3,
                        'fit_params': {
                            'k': k_pow_area_2_3,
                            'r': r_pow_area_2_3,
                            'R2': R2_pow_area_2_3,
                            'p_str': format_p_value(p_pow_area_2_3),
                            'area': area_range
                        },
                        'label': range_label,
                        'color': color
                    })

                # プロット生成
                print('\n=== 比較プロット生成中 ===')

                # 従来手法とGroup2推定の比較プロットは削除（フィッティングなしのlinear-linearプロットのみ出力ディレクトリ直下に保存）
                # Group2-3のみの比較プロットと面積規格化比較プロットのみ実施

                # Group2-3のみのプロット
                power_data_grp2_3 = [d for d in all_ranges_data_grp2_3 if 'fit_type' not in d]
                for scale in ['linear', 'semilog', 'loglog']:
                    output_path = os.path.join(output_dir_power_2_3, f'RSFD_power_law_comparison_group2-3_{scale}')
                    create_multi_range_comparison_plot(
                        power_data_grp2_3, 'Rock size [cm]', 'Cumulative number of rocks',
                        output_path, scale_type=scale, fit_type='Power-law',
                        show_plot=False, xlim=(0, 50)
                    )

                # 面積規格化Group2推定のプロット
                for scale in ['linear', 'semilog', 'loglog']:
                    output_path = os.path.join(output_dir_area_normalized_est, f'RSFD_area_normalized_estimate_group2_comparison_{scale}')
                    create_multi_range_comparison_plot(
                        all_ranges_data_area_normalized_est, 'Rock size [cm]', 'Cumulative number of rocks /m²',
                        output_path, scale_type=scale, fit_type='Power-law',
                        show_plot=False, xlim=(0, 50)
                    )

                # 面積規格化Group2-3のプロット
                for scale in ['linear', 'semilog', 'loglog']:
                    output_path = os.path.join(output_dir_area_normalized_2_3, f'RSFD_area_normalized_group2-3_comparison_{scale}')
                    create_multi_range_comparison_plot(
                        all_ranges_data_area_normalized_2_3, 'Rock size [cm]', 'Cumulative number of rocks /m²',
                        output_path, scale_type=scale, fit_type='Power-law',
                        show_plot=False, xlim=(0, 50)
                    )

                print('複数範囲比較プロット完了')
            else:
                # モード1-3の処理
                with open(data_path, 'r') as f:
                    results = json.load(f).get('results', {})

                x   = np.array([v['x']            for v in results.values()])
                y   = np.array([v['y']            for v in results.values()])
                lab = np.array([v['label']        for v in results.values()], dtype=int)
                time_top    = np.array([none_to_nan(v['time_top'])    for v in results.values()], dtype=float)
                time_bottom = np.array([none_to_nan(v['time_bottom']) for v in results.values()], dtype=float)
                print('ラベルデータ読み込み完了:', len(lab), '個')

                # データ範囲フィルタリング
                original_count = len(lab)
                if time_min is not None and time_max is not None:
                    # Group1: y値で判定、Group2-6: time_topで判定
                    mask_group1 = (lab == 1) & (y >= time_min) & (y <= time_max)
                    mask_others = (lab != 1) & (time_top >= time_min) & (time_top <= time_max)
                    time_mask = mask_group1 | mask_others
                
                    # モード3の場合は論理反転（指定範囲を除外）
                    if mode == '3':
                        time_mask = ~time_mask
                
                    x = x[time_mask]
                    y = y[time_mask]
                    lab = lab[time_mask]
                    time_top = time_top[time_mask]
                    time_bottom = time_bottom[time_mask]
                
                    if mode == '2':
                        print(f'時間範囲フィルタリング後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
                    elif mode == '3':
                        print(f'時間範囲除外後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
                
                if horizontal_min is not None and horizontal_max is not None:
                    horizontal_mask = (x >= horizontal_min) & (x <= horizontal_max)
                
                    # モード3の場合は論理反転（指定範囲を除外）
                    if mode == '3':
                        horizontal_mask = ~horizontal_mask
                
                    x = x[horizontal_mask]
                    y = y[horizontal_mask]
                    lab = lab[horizontal_mask]
                    time_top = time_top[horizontal_mask]
                    time_bottom = time_bottom[horizontal_mask]
                
                    if mode == '2':
                        print(f'水平位置範囲フィルタリング後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
                    elif mode == '3':
                        print(f'水平位置範囲除外後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
                
                print(f'フィルタリング完了: {len(lab)}個のデータを使用')
                
                # ------------------------------------------------------------------
                # 3. ラベル別個数をテキスト出力
                # ------------------------------------------------------------------
                counts = {k: int(np.sum(lab == k)) for k in range(1, 7)}
                with open(os.path.join(output_dir, 'RSFD_counts_by_label.txt'), 'w') as f:
                    f.write('# RSFD Label Counts\n')
                    f.write(f'# Original data count: {original_count}\n')
                
                    # モードに応じたフィルタ情報の記録
                    if mode == '1':
                        f.write('# Mode: Full range (no filtering)\n')
                    elif mode == '2':
                        f.write('# Mode: Use only specified range\n')
                        if time_min is not None and time_max is not None:
                            f.write(f'# Time range filter: {time_min} - {time_max} ns\n')
                        if horizontal_min is not None and horizontal_max is not None:
                            f.write(f'# Horizontal range filter: {horizontal_min} - {horizontal_max} m\n')
                    elif mode == '3':
                        f.write('# Mode: Remove specified range\n')
                        if time_min is not None and time_max is not None:
                            f.write(f'# Removed time range: {time_min} - {time_max} ns\n')
                        if horizontal_min is not None and horizontal_max is not None:
                            f.write(f'# Removed horizontal range: {horizontal_min} - {horizontal_max} m\n')
                
                    f.write(f'# Filtered data count: {len(lab)} ({len(lab)/original_count*100:.1f}%)\n')
                    f.write('\n')
                    for k, v in counts.items():
                        f.write(f'Label {k}: {v}\n')
                
                # ------------------------------------------------------------------
                # 4. ラベル1・2・3 → サイズ配列を作成
                # ------------------------------------------------------------------
                size_label1 = np.full(counts[1], 1.0)      # ラベル1：1 cm
                size_label2 = np.full(counts[2], 6.0)      # ラベル2：6 cm
                mask2_valid = (lab == 2) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
                mask3_valid = (lab == 3) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
                er = 9.0
                c  = 299_792_458  # m/s
                sizes_group2 = (time_bottom[mask2_valid] - time_top[mask2_valid]) * 1e-9 * c / np.sqrt(er) * 0.5 * 100  # [cm]
                sizes_group3 = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(er) * 0.5 * 100  # [cm]
                sizes_group3_max = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(6) * 0.5 * 100  # [cm] # grpup3のエラー範囲
                sizes_group3_min = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(15) * 0.5 * 100  # [cm] # group3のエラー範囲
                
                # サイズ値を小数点以下3桁に丸めて、浮動小数点の微小誤差を排除
                sizes_group2 = np.round(sizes_group2, decimals=3)
                sizes_group3 = np.round(sizes_group3, decimals=3)
                
                all_sizes_cm_traditional = np.concatenate([size_label1, size_label2, sizes_group3])
                all_sizes_cm_estimamte_group2 = np.concatenate([size_label1, sizes_group2, sizes_group3]) # Group2も式で計算した場合
                all_sizes_cm_group2_3 = np.concatenate([sizes_group2, sizes_group3]) # Group2-3のみ（推定手法）
                #all_sizes_cm = sizes_group3 # Group3のみだとどうなる？の検証
                if all_sizes_cm_traditional.size == 0:
                    raise RuntimeError('有効なラベル1–3が見つかりませんでした。')
                
                # ------------------------------------------------------------------
                # 5. 累積サイズ‑頻度分布 (≥ size) を計算
                # ------------------------------------------------------------------
                unique_sizes_traditional = np.sort(np.unique(all_sizes_cm_traditional))
                cum_counts_traditional   = np.array([(all_sizes_cm_traditional >= s).sum() for s in unique_sizes_traditional], dtype=int)
                
                unique_sizes_estimate_group2 = np.sort(np.unique(all_sizes_cm_estimamte_group2))
                cum_counts_estimate_group2   = np.array([(all_sizes_cm_estimamte_group2 >= s).sum() for s in unique_sizes_estimate_group2], dtype=int)
                
                unique_sizes_group2_3 = np.sort(np.unique(all_sizes_cm_group2_3))
                cum_counts_group2_3   = np.array([(all_sizes_cm_group2_3 >= s).sum() for s in unique_sizes_group2_3], dtype=int)
                
                # ------------------------------------------------------------------
                # 6. 汎用プロット関数の定義
                # ------------------------------------------------------------------
                def format_p_value(p):
                    """p値のフォーマットを補助する"""
                    if p < 0.001:
                        return "p < 0.001"
                    else:
                        return f"p={p:.3f}"
                
                def create_rsfd_plot(x_data, y_data, xlabel, ylabel, output_path,
                                     scale_type='linear', fit_lines=None,
                                     show_plot=False, dpi_png=300, dpi_pdf=600,
                                     marker='o', linestyle='-', linewidth=1.5, color=None, label=None,
                                     xlim=None):
                    """
                    RSFDプロットを作成・保存する汎用関数
                
                    Parameters:
                    -----------
                    x_data, y_data : array
                        プロットするデータ
                    xlabel, ylabel : str
                        軸ラベル
                    output_path : str
                        出力パス（拡張子なし）
                    scale_type : str
                        'linear', 'semilog', 'loglog'
                    fit_lines : list of dict, optional
                        フィット曲線のリスト [{'x': x, 'y': y, 'label': label, 'color': color, 'linestyle': style}, ...]
                    show_plot : bool
                        プロット表示の有無
                    dpi_png, dpi_pdf : int
                        解像度
                    marker, linestyle, linewidth, color, label :
                        データプロットのスタイル設定
                    xlim : tuple, optional
                        x軸範囲 (xmin, xmax)
                    """
                    plt.figure(figsize=(8, 6))
                
                    # データプロット
                    if marker and linestyle:
                        plot_kwargs = {'marker': marker, 'linestyle': linestyle, 'linewidth': linewidth}
                        if color:
                            plot_kwargs['color'] = color
                        if label:
                            plot_kwargs['label'] = label
                        plt.plot(x_data, y_data, **plot_kwargs)
                    elif marker:  # scatter plot
                        scatter_kwargs = {'marker': marker}
                        if color:
                            scatter_kwargs['color'] = color
                        if label:
                            scatter_kwargs['label'] = label
                        plt.scatter(x_data, y_data, **scatter_kwargs)
                
                    # フィット曲線の追加
                    if fit_lines:
                        for fit_line in fit_lines:
                            plt.plot(fit_line['x'], fit_line['y'],
                                    linestyle=fit_line.get('linestyle', '--'),
                                    linewidth=fit_line.get('linewidth', 1.5),
                                    color=fit_line.get('color', 'red'),
                                    label=fit_line.get('label', ''))
                
                    # 軸スケール設定
                    if scale_type == 'semilog':
                        plt.yscale('log')
                    elif scale_type == 'loglog':
                        plt.xscale('log')
                        plt.yscale('log')
                
                    # x軸範囲の設定（軸スケール設定の直後に実行）
                    if xlim and scale_type == 'loglog':
                        plt.xlim(max(xlim[0], 0.5), xlim[1])  # logスケールで負またはゼロを避ける
                    elif xlim:
                        plt.xlim(xlim)

                    # 軸ラベルとグリッド
                    plt.xlabel(xlabel, fontsize=20)
                    plt.ylabel(ylabel, fontsize=20)
                    plt.tick_params(labelsize=16)
                    plt.grid(True, linestyle='--', alpha=0.5)

                    # y軸のtick設定（最大値1-20の場合は2刻みに固定）
                    ax = plt.gca()
                    ylim = ax.get_ylim()
                    if 1 <= ylim[1] <= 20:
                        ax.yaxis.set_major_locator(MultipleLocator(2))

                    plt.tight_layout()

                    # 保存
                    plt.savefig(f'{output_path}.png', dpi=dpi_png)
                    plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)

                    # 表示
                    if show_plot:
                        plt.show()
                    else:
                        plt.close()

                    print(f'プロット保存: {output_path}.png')

                    # Legend情報の保存（ラベルがある場合のみ）
                    if label or fit_lines:
                        legend_entries = []

                        # データラベル
                        if label:
                            legend_entries.append({
                                'label': label,
                                'color': color if color else 'black',
                                'linestyle': linestyle if linestyle else '',
                                'marker': marker if marker else ''
                            })

                        # フィット曲線ラベル
                        if fit_lines:
                            for fit_line in fit_lines:
                                legend_entries.append({
                                    'label': fit_line.get('label', ''),
                                    'color': fit_line.get('color', 'red'),
                                    'linestyle': fit_line.get('linestyle', '--'),
                                    'linewidth': fit_line.get('linewidth', 1.5)
                                })

                        # TXT形式で保存
                        save_legend_info_to_txt(output_path, legend_entries)

                        # Legend専用PDF出力
                        save_legend_only_pdf(output_path, legend_entries)
                
                # ------------------------------------------------------------------
                # 7. 従来手法（Traditional）のプロット保存
                # ------------------------------------------------------------------
                # フィッティングなしのlinear-linearプロットのみ、出力ディレクトリ直下に保存
                output_path = os.path.join(output_dir, 'RSFD_traditional_linear')
                create_rsfd_plot(
                    unique_sizes_traditional, cum_counts_traditional,
                    'Rock size [cm]', 'Cumulative number of rocks',
                    output_path, scale_type='linear',
                    show_plot=False,
                    xlim=(0, 50)
                )

                # TXT保存
                with open(os.path.join(output_dir, 'RSFD_traditional.txt'), 'w') as f:
                    f.write('# size_cm\tcumulative_count\n')
                    for s, n in zip(unique_sizes_traditional, cum_counts_traditional):
                        f.write(f'{s:.3f}\t{n}\n')
                print('従来手法累積データTXT保存: RSFD_traditional.txt')

                # Label‑3 詳細ダンプ
                if mask3_valid.any():
                    dump_path = os.path.join(output_dir, 'Label3_detail.txt')
                    with open(dump_path, 'w') as f:
                        f.write('#x\t y\t time_top\t time_bottom\n')
                        for xi, yi, tp, bt in zip(x[mask3_valid], y[mask3_valid], time_top[mask3_valid], time_bottom[mask3_valid]):
                            f.write(f'{xi:.6f}\t{yi:.6f}\t{tp:.3f}\t{bt:.3f}\n')
                    print('Label‑3 詳細を保存:', dump_path)

                # Label‑2-3 詳細ダンプ
                if mask2_valid.any() or mask3_valid.any():
                    dump_path = os.path.join(output_dir, 'Label2-3_detail.txt')
                    with open(dump_path, 'w') as f:
                        f.write('#label\t x\t y\t time_top\t time_bottom\t size_cm\n')

                        # Label2とLabel3のデータを統合
                        combined_mask = mask2_valid | mask3_valid
                        x_combined = x[combined_mask]
                        y_combined = y[combined_mask]
                        time_top_combined = time_top[combined_mask]
                        time_bottom_combined = time_bottom[combined_mask]
                        lab_combined = lab[combined_mask]

                        # サイズを計算
                        size_cm_combined = (time_bottom_combined - time_top_combined) * 1e-9 * c / np.sqrt(er) * 0.5 * 100  # [cm]

                        # サイズの昇順でソート
                        sort_indices = np.argsort(size_cm_combined)

                        # ソートされた順序で出力
                        for i in sort_indices:
                            f.write(f'{lab_combined[i]}\t{x_combined[i]:.6f}\t{y_combined[i]:.6f}\t{time_top_combined[i]:.3f}\t{time_bottom_combined[i]:.3f}\t{size_cm_combined[i]:.8f}\n')
                print('Label‑2-3 詳細を保存:', dump_path)
            
            # ------------------------------------------------------------------
            # 8. フィッティング: べき則と指数関数
            # ------------------------------------------------------------------
            def calc_fitting(sizes, counts):
                # 対数変換
                mask = sizes > 0
                log_D = np.log(sizes[mask])
                log_N = np.log(counts[mask])
            
                # 7.1 べき則フィッティング (Power-law: log N = r log D + log k)
                # 定数項 (切片) のために X に '1' の列を追加
                X_pow = sm.add_constant(log_D)
                # OLSモデルの実行
                model_pow = sm.OLS(log_N, X_pow)
                results_pow = model_pow.fit()
            
                # パラメータを results_pow から抽出
                log_k_pow, r_pow = results_pow.params
                k_pow = np.exp(log_k_pow)
                R2_pow = results_pow.rsquared
                # 傾き(r) の統計量を取得
                r_pow_se = results_pow.bse[1]
                r_pow_t = results_pow.tvalues[1]
                r_pow_p = results_pow.pvalues[1]
                dof_pow = results_pow.df_resid
                n_pow = int(results_pow.nobs)
            
                # 7.2 指数関数フィッティング (Exponential: log N = rD + log k)
                # 定数項 (切片) のために X に '1' の列を追加
                X_exp = sm.add_constant(sizes[mask])
                # OLSモデルの実行
                model_exp = sm.OLS(log_N, X_exp)
                results_exp = model_exp.fit()
            
                # パラメータを results_exp から抽出
                log_k_exp, r_exp = results_exp.params
                k_exp = np.exp(log_k_exp)
                R2_exp = results_exp.rsquared
                # 傾き(r) の統計量を取得
                r_exp_se = results_exp.bse[1]
                r_exp_t = results_exp.tvalues[1]
                r_exp_p = results_exp.pvalues[1]
                dof_exp = results_exp.df_resid
                n_exp = int(results_exp.nobs)
            
                # フィット曲線用に滑らかなサンプル点を生成
                D_fit = np.linspace(sizes.min(), sizes.max(), 200)
                N_pow_fit = k_pow * D_fit**r_pow
                N_exp_fit = k_exp * np.exp(r_exp * D_fit)
            
                # べき則の結果, 指数関数の結果, D_fit
                return (k_pow, np.abs(r_pow), R2_pow, N_pow_fit, r_pow_t, r_pow_p, r_pow_se, n_pow, dof_pow), \
                       (k_exp, np.abs(r_exp), R2_exp, N_exp_fit, r_exp_t, r_exp_p, r_exp_se, n_exp, dof_exp), \
                       D_fit
            
            (k_pow_trad, r_pow_trad, R2_pow_trad, N_pow_fit_trad, t_pow_trad, p_pow_trad, se_pow_trad, n_pow_trad, dof_pow_trad),\
                (k_exp_trad, r_exp_trad, R2_exp_trad, N_exp_fit_trad, t_exp_trad, p_exp_trad, se_exp_trad, n_exp_trad, dof_exp_trad), D_fit_trad\
                = calc_fitting(unique_sizes_traditional, cum_counts_traditional)
            
            (k_pow_est_grp2, r_pow_est_grp2, R2_pow_est_grp2, N_pow_fit_est_grp2, t_pow_est_grp2, p_pow_est_grp2, se_pow_est_grp2, n_pow_est_grp2, dof_pow_est_grp2),\
                (k_exp_est_grp2, r_exp_est_grp2, R2_exp_est_grp2, N_exp_fit_est_grp2, t_exp_est_grp2, p_exp_est_grp2, se_exp_est_grp2, n_exp_est_grp2, dof_exp_est_grp2), D_fit_est_grp2\
                = calc_fitting(unique_sizes_estimate_group2, cum_counts_estimate_group2)
            
            (k_pow_grp2_3, r_pow_grp2_3, R2_pow_grp2_3, N_pow_fit_grp2_3, t_pow_grp2_3, p_pow_grp2_3, se_pow_grp2_3, n_pow_grp2_3, dof_pow_grp2_3),\
                (k_exp_grp2_3, r_exp_grp2_3, R2_exp_grp2_3, N_exp_fit_grp2_3, t_exp_grp2_3, p_exp_grp2_3, se_exp_grp2_3, n_exp_grp2_3, dof_exp_grp2_3), D_fit_grp2_3\
                = calc_fitting(unique_sizes_group2_3, cum_counts_group2_3)
            
            # ------------------------------------------------------------------
            # 9. プロット: Group2推定とGroup2-3のフィッティング（3種類のスケール）
            # ------------------------------------------------------------------
            # 従来手法のフィッティングプロットは削除（フィッティングなしのlinear-linearプロットのみ出力ディレクトリ直下に保存済み）

            # 9.1 べき則フィット（Group2推定）
            p_str_pow_est_grp2 = format_p_value(p_pow_est_grp2)
            fit_lines_pow_est_grp2 = [{
                'x': D_fit_est_grp2, 'y': N_pow_fit_est_grp2,
                'label': f'Power-law: k={k_pow_est_grp2:.2e}, r={r_pow_est_grp2:.3f}, R²={R2_pow_est_grp2:.4f}, {p_str_pow_est_grp2}',
                'color': 'red', 'linestyle': '--'
            }]

            # 9.2 指数関数フィット（Group2推定）
            p_str_exp_est_grp2 = format_p_value(p_exp_est_grp2)
            fit_lines_exp_est_grp2 = [{
                'x': D_fit_est_grp2, 'y': N_exp_fit_est_grp2,
                'label': f'Exponential: k={k_exp_est_grp2:.2e}, r={r_exp_est_grp2:.3f}, R²={R2_exp_est_grp2:.4f}, {p_str_exp_est_grp2}',
                'color': 'green', 'linestyle': '--'
            }]

            # ------------------------------------------------------------------
            # 10. プロット: フィッティング比較（3種類のスケール）
            # ------------------------------------------------------------------
            # 従来手法の比較プロットは削除（フィッティングプロット自体を削除したため）
            # Group2推定の比較プロットのみ実施
            
            # ------------------------------------------------------------------
            # 11. プロット: Group2-3のみのフィッティング（3種類のスケール）
            # ------------------------------------------------------------------
            # 11.1 べき則フィット（Group2-3のみ）
            p_str_pow_grp2_3 = format_p_value(p_pow_grp2_3)  # p値の書式設定
            fit_lines_pow_grp2_3 = [{
                'x': D_fit_grp2_3, 'y': N_pow_fit_grp2_3,
                'label': f'Power-law: k={k_pow_grp2_3:.2e}, r={r_pow_grp2_3:.3f}, R²={R2_pow_grp2_3:.4f}, {p_str_pow_grp2_3}',
                'color': 'red', 'linestyle': '--'
            }]
            
            for scale in ['linear', 'semilog', 'loglog']:
                output_path = os.path.join(output_dir_power_2_3, f'RSFD_power_law_fit_group2-3_{scale}')
                create_rsfd_plot(
                    unique_sizes_group2_3, cum_counts_group2_3,
                    'Rock size [cm]', 'Cumulative number of rocks',
                    output_path, scale_type=scale,
                    fit_lines=fit_lines_pow_grp2_3,
                    marker='o', linestyle='', label='Data (Group2-3)',
                    show_plot=False,
                    xlim=(0, 50)
                )

            # 11.2 Group比較プロット（Group1-3 vs Group2-3、area normalizeなし）
            for scale in ['linear', 'semilog', 'loglog']:
                output_path = os.path.join(output_dir_group_comparison, f'RSFD_group_comparison_{scale}')

                plt.figure(figsize=(10, 8))

                # データ点（Group1-3のみ表示）
                plt.plot(unique_sizes_estimate_group2, cum_counts_estimate_group2,
                        marker='o', linestyle='', color='black', label='Data')

                # Group 1-3のフィット線
                plt.plot(D_fit_est_grp2, N_pow_fit_est_grp2,
                        linestyle='--', linewidth=1.5, color='blue',
                        label=f'Group1-3 fit: k={k_pow_est_grp2:.2e}, r={r_pow_est_grp2:.3f}, R²={R2_pow_est_grp2:.4f}, {p_str_pow_est_grp2}')

                # Group 2-3のフィット線
                plt.plot(D_fit_grp2_3, N_pow_fit_grp2_3,
                        linestyle='--', linewidth=1.5, color='red',
                        label=f'Group2-3 fit: k={k_pow_grp2_3:.2e}, r={r_pow_grp2_3:.3f}, R²={R2_pow_grp2_3:.4f}, {p_str_pow_grp2_3}')

                # 軸スケール設定
                if scale == 'semilog':
                    plt.yscale('log')
                elif scale == 'loglog':
                    plt.xscale('log')
                    plt.yscale('log')

                # x軸範囲
                if scale == 'loglog':
                    plt.xlim(max(0, 0.5), 50)
                else:
                    plt.xlim(0, 50)

                # 軸ラベルとグリッド
                plt.xlabel('Rock size [cm]', fontsize=20)
                plt.ylabel('Cumulative number of rocks', fontsize=20)
                plt.tick_params(labelsize=16)
                plt.grid(True, linestyle='--', alpha=0.5)

                # y軸のtick設定（最大値1-20の場合は2刻みに固定）
                ax = plt.gca()
                ylim = ax.get_ylim()
                if 1 <= ylim[1] <= 20:
                    ax.yaxis.set_major_locator(MultipleLocator(2))

                plt.tight_layout()

                # 保存
                plt.savefig(f'{output_path}.png', dpi=300)
                plt.savefig(f'{output_path}.pdf', dpi=600)
                plt.close()

                print(f'Group比較プロット保存: {output_path}.png')

                # Legend情報の保存
                legend_entries = [
                    {'label': 'Data', 'color': 'black', 'marker': 'o', 'linestyle': ''},
                    {'label': f'Group1-3 fit: k={k_pow_est_grp2:.2e}, r={r_pow_est_grp2:.3f}, R²={R2_pow_est_grp2:.4f}, {p_str_pow_est_grp2}',
                     'color': 'blue', 'linestyle': '--', 'linewidth': 1.5},
                    {'label': f'Group2-3 fit: k={k_pow_grp2_3:.2e}, r={r_pow_grp2_3:.3f}, R²={R2_pow_grp2_3:.4f}, {p_str_pow_grp2_3}',
                     'color': 'red', 'linestyle': '--', 'linewidth': 1.5}
                ]
                save_legend_info_to_txt(output_path, legend_entries)
                save_legend_only_pdf(output_path, legend_entries)

            # TXT保存（Group2-3）
            with open(os.path.join(output_dir, 'RSFD_linear_group2-3.txt'), 'w') as f:
                f.write('# size_cm\tcumulative_count\n')
                for s, n in zip(unique_sizes_group2_3, cum_counts_group2_3):
                    f.write(f'{s:.3f}\t{n}\n')
            print('Group2-3累積データTXT保存: RSFD_linear_group2-3.txt')

            # ------------------------------------------------------------------
            # 11.4 面積規格化フィッティング（Group2推定）
            # ------------------------------------------------------------------
            (k_pow_area_est, r_pow_area_est, R2_pow_area_est, N_pow_fit_area_est,
             t_pow_area_est, p_pow_area_est, se_pow_area_est, n_pow_area_est, dof_pow_area_est), \
            D_fit_area_est, cum_counts_area_est = calc_fitting_area_normalized(
                unique_sizes_estimate_group2, cum_counts_estimate_group2, area)

            p_str_pow_area_est = format_p_value(p_pow_area_est)
            fit_lines_pow_area_est = [{
                'x': D_fit_area_est, 'y': N_pow_fit_area_est,
                'label': f'Power-law: k={k_pow_area_est:.2e}, r={r_pow_area_est:.3f}, R²={R2_pow_area_est:.4f}, {p_str_pow_area_est}',
                'color': 'red', 'linestyle': '--'
            }]

            for scale in ['linear', 'semilog', 'loglog']:
                output_path = os.path.join(output_dir_area_normalized_est, f'RSFD_area_normalized_estimate_group2_{scale}')
                create_rsfd_plot(
                    unique_sizes_estimate_group2, cum_counts_area_est,
                    'Rock size [cm]', 'Cumulative number of rocks /m²',
                    output_path, scale_type=scale,
                    fit_lines=fit_lines_pow_area_est,
                    marker='o', linestyle='', label='Data (Estimate Group2, Area-normalized)',
                    show_plot=(scale == 'linear'),
                    xlim=(0, 50)
                )

            # TXT保存（面積規格化Group2推定）
            with open(os.path.join(output_dir, 'RSFD_area_normalized_estimate_group2.txt'), 'w') as f:
                f.write(f'# Area: {area} m²\n')
                f.write('# size_cm\tcumulative_count_per_m2\n')
                for s, n in zip(unique_sizes_estimate_group2, cum_counts_area_est):
                    f.write(f'{s:.3f}\t{n:.6f}\n')
            print('面積規格化Group2推定累積データTXT保存: RSFD_area_normalized_estimate_group2.txt')

            # ------------------------------------------------------------------
            # 11.5 面積規格化フィッティング（Group2-3）
            # ------------------------------------------------------------------
            (k_pow_area_2_3, r_pow_area_2_3, R2_pow_area_2_3, N_pow_fit_area_2_3,
             t_pow_area_2_3, p_pow_area_2_3, se_pow_area_2_3, n_pow_area_2_3, dof_pow_area_2_3), \
            D_fit_area_2_3, cum_counts_area_2_3 = calc_fitting_area_normalized(
                unique_sizes_group2_3, cum_counts_group2_3, area)

            p_str_pow_area_2_3 = format_p_value(p_pow_area_2_3)
            fit_lines_pow_area_2_3 = [{
                'x': D_fit_area_2_3, 'y': N_pow_fit_area_2_3,
                'label': f'Power-law: k={k_pow_area_2_3:.2e}, r={r_pow_area_2_3:.3f}, R²={R2_pow_area_2_3:.4f}, {p_str_pow_area_2_3}',
                'color': 'red', 'linestyle': '--'
            }]

            for scale in ['linear', 'semilog', 'loglog']:
                output_path = os.path.join(output_dir_area_normalized_2_3, f'RSFD_area_normalized_group2-3_{scale}')
                create_rsfd_plot(
                    unique_sizes_group2_3, cum_counts_area_2_3,
                    'Rock size [cm]', 'Cumulative number of rocks /m²',
                    output_path, scale_type=scale,
                    fit_lines=fit_lines_pow_area_2_3,
                    marker='o', linestyle='', label='Data (Group2-3, Area-normalized)',
                    show_plot=(scale == 'linear'),
                    xlim=(0, 50)
                )

            # TXT保存（面積規格化Group2-3）
            with open(os.path.join(output_dir, 'RSFD_area_normalized_group2-3.txt'), 'w') as f:
                f.write(f'# Area: {area} m²\n')
                f.write('# size_cm\tcumulative_count_per_m2\n')
                for s, n in zip(unique_sizes_group2_3, cum_counts_area_2_3):
                    f.write(f'{s:.3f}\t{n:.6f}\n')
            print('面積規格化Group2-3累積データTXT保存: RSFD_area_normalized_group2-3.txt')

            # ------------------------------------------------------------------
            # 11.6 Group比較プロット（Group1-3 vs Group2-3、area normalizeあり）
            # ------------------------------------------------------------------
            for scale in ['linear', 'semilog', 'loglog']:
                output_path = os.path.join(output_dir_group_comparison_area, f'RSFD_group_comparison_area_normalized_{scale}')

                plt.figure(figsize=(10, 8))

                # データ点（Group1-3のみ表示、面積規格化）
                plt.plot(unique_sizes_estimate_group2, cum_counts_area_est,
                        marker='o', linestyle='', color='black', label='Data')

                # Group 1-3のフィット線（面積規格化）
                plt.plot(D_fit_area_est, N_pow_fit_area_est,
                        linestyle='--', linewidth=1.5, color='blue',
                        label=f'Group1-3 fit: k={k_pow_area_est:.2e}, r={r_pow_area_est:.3f}, R²={R2_pow_area_est:.4f}, {p_str_pow_area_est}')

                # Group 2-3のフィット線（面積規格化）
                plt.plot(D_fit_area_2_3, N_pow_fit_area_2_3,
                        linestyle='--', linewidth=1.5, color='red',
                        label=f'Group2-3 fit: k={k_pow_area_2_3:.2e}, r={r_pow_area_2_3:.3f}, R²={R2_pow_area_2_3:.4f}, {p_str_pow_area_2_3}')

                # 軸スケール設定
                if scale == 'semilog':
                    plt.yscale('log')
                elif scale == 'loglog':
                    plt.xscale('log')
                    plt.yscale('log')

                # x軸範囲
                if scale == 'loglog':
                    plt.xlim(max(0, 0.5), 50)
                else:
                    plt.xlim(0, 50)

                # 軸ラベルとグリッド
                plt.xlabel('Rock size [cm]', fontsize=20)
                plt.ylabel('Cumulative number of rocks /m²', fontsize=20)
                plt.tick_params(labelsize=16)
                plt.grid(True, linestyle='--', alpha=0.5)

                # y軸のtick設定（最大値1-20の場合は2刻みに固定）
                ax = plt.gca()
                ylim = ax.get_ylim()
                if 1 <= ylim[1] <= 20:
                    ax.yaxis.set_major_locator(MultipleLocator(2))

                plt.tight_layout()

                # 保存
                plt.savefig(f'{output_path}.png', dpi=300)
                plt.savefig(f'{output_path}.pdf', dpi=600)
                plt.close()

                print(f'Group比較プロット（面積規格化）保存: {output_path}.png')

                # Legend情報の保存
                legend_entries = [
                    {'label': 'Data', 'color': 'black', 'marker': 'o', 'linestyle': ''},
                    {'label': f'Group1-3 fit: k={k_pow_area_est:.2e}, r={r_pow_area_est:.3f}, R²={R2_pow_area_est:.4f}, {p_str_pow_area_est}',
                     'color': 'blue', 'linestyle': '--', 'linewidth': 1.5},
                    {'label': f'Group2-3 fit: k={k_pow_area_2_3:.2e}, r={r_pow_area_2_3:.3f}, R²={R2_pow_area_2_3:.4f}, {p_str_pow_area_2_3}',
                     'color': 'red', 'linestyle': '--', 'linewidth': 1.5}
                ]
                save_legend_info_to_txt(output_path, legend_entries)
                save_legend_only_pdf(output_path, legend_entries)

            # ------------------------------------------------------------------
            # 12. フィッティングサマリーファイルの出力
            # ------------------------------------------------------------------
            summary_file_path = os.path.join(output_dir, 'RSFD_fitting_summary.txt')
            with open(summary_file_path, 'w') as f:
                # ヘッダー (タブ区切り)
                f.write('DataSet\tModel\tk\tr\tR_squared\tr_StdErr\tr_t_value\tr_p_value\tN_points\tDOF\n')
            
                # 1. 従来手法 (Traditional)
                f.write(f'Traditional\tPower\t{k_pow_trad:.4e}\t{r_pow_trad:.4f}\t{R2_pow_trad:.4f}\t'
                        f'{se_pow_trad:.4f}\t{t_pow_trad:.3f}\t{p_pow_trad:.4e}\t{n_pow_trad}\t{dof_pow_trad}\n')
                f.write(f'Traditional\tExponential\t{k_exp_trad:.4e}\t{r_exp_trad:.4f}\t{R2_exp_trad:.4f}\t'
                        f'{se_exp_trad:.4f}\t{t_exp_trad:.3f}\t{p_exp_trad:.4e}\t{n_exp_trad}\t{dof_exp_trad}\n')
            
                # 2. Group2 推定 (Estimate_Group2)
                f.write(f'Estimate_Group2\tPower\t{k_pow_est_grp2:.4e}\t{r_pow_est_grp2:.4f}\t{R2_pow_est_grp2:.4f}\t'
                        f'{se_pow_est_grp2:.4f}\t{t_pow_est_grp2:.3f}\t{p_pow_est_grp2:.4e}\t{n_pow_est_grp2}\t{dof_pow_est_grp2}\n')
                f.write(f'Estimate_Group2\tExponential\t{k_exp_est_grp2:.4e}\t{r_exp_est_grp2:.4f}\t{R2_exp_est_grp2:.4f}\t'
                        f'{se_exp_est_grp2:.4f}\t{t_exp_est_grp2:.3f}\t{p_exp_est_grp2:.4e}\t{n_exp_est_grp2}\t{dof_exp_est_grp2}\n')
            
                # 3. Group2-3 のみ (Group2-3_Only)
                f.write(f'Group2-3_Only\tPower\t{k_pow_grp2_3:.4e}\t{r_pow_grp2_3:.4f}\t{R2_pow_grp2_3:.4f}\t'
                        f'{se_pow_grp2_3:.4f}\t{t_pow_grp2_3:.3f}\t{p_pow_grp2_3:.4e}\t{n_pow_grp2_3}\t{dof_pow_grp2_3}\n')
                f.write(f'Group2-3_Only\tExponential\t{k_exp_grp2_3:.4e}\t{r_exp_grp2_3:.4f}\t{R2_exp_grp2_3:.4f}\t'
                        f'{se_exp_grp2_3:.4f}\t{t_exp_grp2_3:.3f}\t{p_exp_grp2_3:.4e}\t{n_exp_grp2_3}\t{dof_exp_grp2_3}\n')

                # 4. 面積規格化 Estimate_Group2 (Area_Normalized_Est_Grp2)
                f.write(f'Area_Normalized_Est_Grp2\tPower\t{k_pow_area_est:.4e}\t{r_pow_area_est:.4f}\t{R2_pow_area_est:.4f}\t'
                        f'{se_pow_area_est:.4f}\t{t_pow_area_est:.3f}\t{p_pow_area_est:.4e}\t{n_pow_area_est}\t{dof_pow_area_est}\n')

                # 5. 面積規格化 Group2-3 (Area_Normalized_Grp2-3)
                f.write(f'Area_Normalized_Grp2-3\tPower\t{k_pow_area_2_3:.4e}\t{r_pow_area_2_3:.4f}\t{R2_pow_area_2_3:.4f}\t'
                        f'{se_pow_area_2_3:.4f}\t{t_pow_area_2_3:.3f}\t{p_pow_area_2_3:.4e}\t{n_pow_area_2_3}\t{dof_pow_area_2_3}\n')

                # 面積情報
                f.write(f'\n# Area: {area} m²\n')

            print(f'フィッティングサマリー保存: {summary_file_path}')
            print(f'✓ 処理 {record_idx} 完了\n')

        except Exception as e:
            print(f'✗ 処理 {record_idx} でエラーが発生しました: {e}')
            print(f'次の処理に進みます...\n')
            continue

    print('\n全ての再作成処理が完了しました！')
    exit(0)

# ------------------------------------------------------------------
# 共通処理: JSON 読み込み（startup_mode == '1'の場合）
# ------------------------------------------------------------------
if mode != '4':
    # モード1-3の処理
    with open(data_path, 'r') as f:
        results = json.load(f).get('results', {})

    x   = np.array([v['x']            for v in results.values()])
    y   = np.array([v['y']            for v in results.values()])
    lab = np.array([v['label']        for v in results.values()], dtype=int)
    time_top    = np.array([none_to_nan(v['time_top'])    for v in results.values()], dtype=float)
    time_bottom = np.array([none_to_nan(v['time_bottom']) for v in results.values()], dtype=float)
    print('ラベルデータ読み込み完了:', len(lab), '個')
    
    # データ範囲フィルタリング
    original_count = len(lab)
    if time_min is not None and time_max is not None:
        # Group1: y値で判定、Group2-6: time_topで判定
        mask_group1 = (lab == 1) & (y >= time_min) & (y <= time_max)
        mask_others = (lab != 1) & (time_top >= time_min) & (time_top <= time_max)
        time_mask = mask_group1 | mask_others
    
        # モード3の場合は論理反転（指定範囲を除外）
        if mode == '3':
            time_mask = ~time_mask
    
        x = x[time_mask]
        y = y[time_mask]
        lab = lab[time_mask]
        time_top = time_top[time_mask]
        time_bottom = time_bottom[time_mask]
    
        if mode == '2':
            print(f'時間範囲フィルタリング後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
        elif mode == '3':
            print(f'時間範囲除外後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
    
    if horizontal_min is not None and horizontal_max is not None:
        horizontal_mask = (x >= horizontal_min) & (x <= horizontal_max)
    
        # モード3の場合は論理反転（指定範囲を除外）
        if mode == '3':
            horizontal_mask = ~horizontal_mask
    
        x = x[horizontal_mask]
        y = y[horizontal_mask]
        lab = lab[horizontal_mask]
        time_top = time_top[horizontal_mask]
        time_bottom = time_bottom[horizontal_mask]
    
        if mode == '2':
            print(f'水平位置範囲フィルタリング後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
        elif mode == '3':
            print(f'水平位置範囲除外後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
    
    print(f'フィルタリング完了: {len(lab)}個のデータを使用')
    
    
    # ------------------------------------------------------------------
    # 3. ラベル別個数をテキスト出力
    # ------------------------------------------------------------------
    # 3. ラベル別個数をテキスト出力
    # ------------------------------------------------------------------
    counts = {k: int(np.sum(lab == k)) for k in range(1, 7)}
    with open(os.path.join(output_dir, 'RSFD_counts_by_label.txt'), 'w') as f:
        f.write('# RSFD Label Counts\n')
        f.write(f'# Original data count: {original_count}\n')
    
        # モードに応じたフィルタ情報の記録
        if mode == '1':
            f.write('# Mode: Full range (no filtering)\n')
        elif mode == '2':
            f.write('# Mode: Use only specified range\n')
            if time_min is not None and time_max is not None:
                f.write(f'# Time range filter: {time_min} - {time_max} ns\n')
            if horizontal_min is not None and horizontal_max is not None:
                f.write(f'# Horizontal range filter: {horizontal_min} - {horizontal_max} m\n')
        elif mode == '3':
            f.write('# Mode: Remove specified range\n')
            if time_min is not None and time_max is not None:
                f.write(f'# Removed time range: {time_min} - {time_max} ns\n')
            if horizontal_min is not None and horizontal_max is not None:
                f.write(f'# Removed horizontal range: {horizontal_min} - {horizontal_max} m\n')
    
        f.write(f'# Filtered data count: {len(lab)} ({len(lab)/original_count*100:.1f}%)\n')
        f.write('\n')
        for k, v in counts.items():
            f.write(f'Label {k}: {v}\n')
    
    # ------------------------------------------------------------------
    # 4. ラベル1・2・3 → サイズ配列を作成
    # ------------------------------------------------------------------
    size_label1 = np.full(counts[1], 1.0)      # ラベル1：1 cm
    size_label2 = np.full(counts[2], 6.0)      # ラベル2：6 cm
    mask2_valid = (lab == 2) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
    mask3_valid = (lab == 3) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
    er = 9.0
    c  = 299_792_458  # m/s
    sizes_group2 = (time_bottom[mask2_valid] - time_top[mask2_valid]) * 1e-9 * c / np.sqrt(er) * 0.5 * 100  # [cm]
    sizes_group3 = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(er) * 0.5 * 100  # [cm]
    sizes_group3_max = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(6) * 0.5 * 100  # [cm] # grpup3のエラー範囲
    sizes_group3_min = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(15) * 0.5 * 100  # [cm] # group3のエラー範囲
    
    # サイズ値を小数点以下3桁に丸めて、浮動小数点の微小誤差を排除
    sizes_group2 = np.round(sizes_group2, decimals=3)
    sizes_group3 = np.round(sizes_group3, decimals=3)
    
    all_sizes_cm_traditional = np.concatenate([size_label1, size_label2, sizes_group3])
    all_sizes_cm_estimamte_group2 = np.concatenate([size_label1, sizes_group2, sizes_group3]) # Group2も式で計算した場合
    all_sizes_cm_group2_3 = np.concatenate([sizes_group2, sizes_group3]) # Group2-3のみ（推定手法）
    #all_sizes_cm = sizes_group3 # Group3のみだとどうなる？の検証
    if all_sizes_cm_traditional.size == 0:
        raise RuntimeError('有効なラベル1–3が見つかりませんでした。')
    
    # ------------------------------------------------------------------
    # 5. 累積サイズ‑頻度分布 (≥ size) を計算
    # ------------------------------------------------------------------
    unique_sizes_traditional = np.sort(np.unique(all_sizes_cm_traditional))
    cum_counts_traditional   = np.array([(all_sizes_cm_traditional >= s).sum() for s in unique_sizes_traditional], dtype=int)
    
    unique_sizes_estimate_group2 = np.sort(np.unique(all_sizes_cm_estimamte_group2))
    cum_counts_estimate_group2   = np.array([(all_sizes_cm_estimamte_group2 >= s).sum() for s in unique_sizes_estimate_group2], dtype=int)
    
    unique_sizes_group2_3 = np.sort(np.unique(all_sizes_cm_group2_3))
    cum_counts_group2_3   = np.array([(all_sizes_cm_group2_3 >= s).sum() for s in unique_sizes_group2_3], dtype=int)
    
    # ------------------------------------------------------------------
    # 6. 汎用プロット関数の定義
    # ------------------------------------------------------------------
    def format_p_value(p):
        """p値のフォーマットを補助する"""
        if p < 0.001:
            return "p < 0.001"
        else:
            return f"p={p:.3f}"
    
    def create_rsfd_plot(x_data, y_data, xlabel, ylabel, output_path,
                         scale_type='linear', fit_lines=None,
                         show_plot=False, dpi_png=300, dpi_pdf=600,
                         marker='o', linestyle='-', linewidth=1.5, color=None, label=None,
                         xlim=None):
        """
        RSFDプロットを作成・保存する汎用関数
    
        Parameters:
        -----------
        x_data, y_data : array
            プロットするデータ
        xlabel, ylabel : str
            軸ラベル
        output_path : str
            出力パス（拡張子なし）
        scale_type : str
            'linear', 'semilog', 'loglog'
        fit_lines : list of dict, optional
            フィット曲線のリスト [{'x': x, 'y': y, 'label': label, 'color': color, 'linestyle': style}, ...]
        show_plot : bool
            プロット表示の有無
        dpi_png, dpi_pdf : int
            解像度
        marker, linestyle, linewidth, color, label :
            データプロットのスタイル設定
        xlim : tuple, optional
            x軸範囲 (xmin, xmax)
        """
        plt.figure(figsize=(8, 6))
    
        # データプロット
        if marker and linestyle:
            plot_kwargs = {'marker': marker, 'linestyle': linestyle, 'linewidth': linewidth}
            if color:
                plot_kwargs['color'] = color
            if label:
                plot_kwargs['label'] = label
            plt.plot(x_data, y_data, **plot_kwargs)
        elif marker:  # scatter plot
            scatter_kwargs = {'marker': marker}
            if color:
                scatter_kwargs['color'] = color
            if label:
                scatter_kwargs['label'] = label
            plt.scatter(x_data, y_data, **scatter_kwargs)
    
        # フィット曲線の追加
        if fit_lines:
            for fit_line in fit_lines:
                plt.plot(fit_line['x'], fit_line['y'],
                        linestyle=fit_line.get('linestyle', '--'),
                        linewidth=fit_line.get('linewidth', 1.5),
                        color=fit_line.get('color', 'red'),
                        label=fit_line.get('label', ''))
    
        # 軸スケール設定
        if scale_type == 'semilog':
            plt.yscale('log')
        elif scale_type == 'loglog':
            plt.xscale('log')
            plt.yscale('log')
    
        # x軸範囲の設定（軸スケール設定の直後に実行）
        if xlim and scale_type == 'loglog':
            plt.xlim(max(xlim[0], 0.5), xlim[1])  # logスケールで負またはゼロを避ける
        elif xlim:
            plt.xlim(xlim)
    
        # 軸ラベルとグリッド
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.tick_params(labelsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)

        # y軸のtick設定（最大値1-20の場合は2刻みに固定）
        ax = plt.gca()
        ylim = ax.get_ylim()
        if 1 <= ylim[1] <= 20:
            ax.yaxis.set_major_locator(MultipleLocator(2))

        plt.tight_layout()

        # 保存
        plt.savefig(f'{output_path}.png', dpi=dpi_png)
        plt.savefig(f'{output_path}.pdf', dpi=dpi_pdf)

        # 表示
        if show_plot:
            plt.show()
        else:
            plt.close()

        print(f'プロット保存: {output_path}.png')

        # Legend情報の保存（ラベルがある場合のみ）
        if label or fit_lines:
            legend_entries = []

            # データラベル
            if label:
                legend_entries.append({
                    'label': label,
                    'color': color if color else 'black',
                    'linestyle': linestyle if linestyle else '',
                    'marker': marker if marker else ''
                })

            # フィット曲線ラベル
            if fit_lines:
                for fit_line in fit_lines:
                    legend_entries.append({
                        'label': fit_line.get('label', ''),
                        'color': fit_line.get('color', 'red'),
                        'linestyle': fit_line.get('linestyle', '--'),
                        'linewidth': fit_line.get('linewidth', 1.5)
                    })

            # TXT形式で保存
            save_legend_info_to_txt(output_path, legend_entries)

            # Legend専用PDF出力
            save_legend_only_pdf(output_path, legend_entries)
    
    # ------------------------------------------------------------------
    # 7. 従来手法（Traditional）のプロット保存
    # ------------------------------------------------------------------
    # フィッティングなしのlinear-linearプロットのみ、出力ディレクトリ直下に保存
    output_path = os.path.join(output_dir, 'RSFD_traditional_linear')
    create_rsfd_plot(
        unique_sizes_traditional, cum_counts_traditional,
        'Rock size [cm]', 'Cumulative number of rocks',
        output_path, scale_type='linear',
        show_plot=False,
        xlim=(0, 50)
    )

    # TXT保存
    with open(os.path.join(output_dir, 'RSFD_traditional.txt'), 'w') as f:
        f.write('# size_cm\tcumulative_count\n')
        for s, n in zip(unique_sizes_traditional, cum_counts_traditional):
            f.write(f'{s:.3f}\t{n}\n')
    print('従来手法累積データTXT保存: RSFD_traditional.txt')

    # Label‑3 詳細ダンプ
    if mask3_valid.any():
        dump_path = os.path.join(output_dir, 'Label3_detail.txt')
        with open(dump_path, 'w') as f:
            f.write('#x\t y\t time_top\t time_bottom\n')
            for xi, yi, tp, bt in zip(x[mask3_valid], y[mask3_valid], time_top[mask3_valid], time_bottom[mask3_valid]):
                f.write(f'{xi:.6f}\t{yi:.6f}\t{tp:.3f}\t{bt:.3f}\n')
        print('Label‑3 詳細を保存:', dump_path)
    
    # Label‑3 詳細ダンプ
    if mask2_valid.any() or mask3_valid.any():
        dump_path = os.path.join(output_dir, 'Label2-3_detail.txt')
        with open(dump_path, 'w') as f:
            f.write('#label\t x\t y\t time_top\t time_bottom\t size_cm\n')
    
            # Label2とLabel3のデータを統合
            combined_mask = mask2_valid | mask3_valid
            x_combined = x[combined_mask]
            y_combined = y[combined_mask]
            time_top_combined = time_top[combined_mask]
            time_bottom_combined = time_bottom[combined_mask]
            lab_combined = lab[combined_mask]
    
            # サイズを計算
            size_cm_combined = (time_bottom_combined - time_top_combined) * 1e-9 * c / np.sqrt(er) * 0.5 * 100  # [cm]
    
            # サイズの昇順でソート
            sort_indices = np.argsort(size_cm_combined)
    
            # ソートされた順序で出力
            for i in sort_indices:
                f.write(f'{lab_combined[i]}\t{x_combined[i]:.6f}\t{y_combined[i]:.6f}\t{time_top_combined[i]:.3f}\t{time_bottom_combined[i]:.3f}\t{size_cm_combined[i]:.8f}\n')
        print('Label‑2-3 詳細を保存:', dump_path)
    
    # ------------------------------------------------------------------
    # 8. フィッティング: べき則と指数関数
    # ------------------------------------------------------------------
    def calc_fitting(sizes, counts):
        # 対数変換
        mask = sizes > 0
        log_D = np.log(sizes[mask])
        log_N = np.log(counts[mask])
    
        # 7.1 べき則フィッティング (Power-law: log N = r log D + log k)
        # 定数項 (切片) のために X に '1' の列を追加
        X_pow = sm.add_constant(log_D)
        # OLSモデルの実行
        model_pow = sm.OLS(log_N, X_pow)
        results_pow = model_pow.fit()
    
        # パラメータを results_pow から抽出
        log_k_pow, r_pow = results_pow.params
        k_pow = np.exp(log_k_pow)
        R2_pow = results_pow.rsquared
        # 傾き(r) の統計量を取得
        r_pow_se = results_pow.bse[1]
        r_pow_t = results_pow.tvalues[1]
        r_pow_p = results_pow.pvalues[1]
        dof_pow = results_pow.df_resid
        n_pow = int(results_pow.nobs)
    
        # 7.2 指数関数フィッティング (Exponential: log N = rD + log k)
        # 定数項 (切片) のために X に '1' の列を追加
        X_exp = sm.add_constant(sizes[mask])
        # OLSモデルの実行
        model_exp = sm.OLS(log_N, X_exp)
        results_exp = model_exp.fit()
    
        # パラメータを results_exp から抽出
        log_k_exp, r_exp = results_exp.params
        k_exp = np.exp(log_k_exp)
        R2_exp = results_exp.rsquared
        # 傾き(r) の統計量を取得
        r_exp_se = results_exp.bse[1]
        r_exp_t = results_exp.tvalues[1]
        r_exp_p = results_exp.pvalues[1]
        dof_exp = results_exp.df_resid
        n_exp = int(results_exp.nobs)
    
        # フィット曲線用に滑らかなサンプル点を生成
        D_fit = np.linspace(sizes.min(), sizes.max(), 200)
        N_pow_fit = k_pow * D_fit**r_pow
        N_exp_fit = k_exp * np.exp(r_exp * D_fit)
    
        # べき則の結果, 指数関数の結果, D_fit
        return (k_pow, np.abs(r_pow), R2_pow, N_pow_fit, r_pow_t, r_pow_p, r_pow_se, n_pow, dof_pow), \
               (k_exp, np.abs(r_exp), R2_exp, N_exp_fit, r_exp_t, r_exp_p, r_exp_se, n_exp, dof_exp), \
               D_fit
    
    (k_pow_trad, r_pow_trad, R2_pow_trad, N_pow_fit_trad, t_pow_trad, p_pow_trad, se_pow_trad, n_pow_trad, dof_pow_trad),\
        (k_exp_trad, r_exp_trad, R2_exp_trad, N_exp_fit_trad, t_exp_trad, p_exp_trad, se_exp_trad, n_exp_trad, dof_exp_trad), D_fit_trad\
        = calc_fitting(unique_sizes_traditional, cum_counts_traditional)
    
    (k_pow_est_grp2, r_pow_est_grp2, R2_pow_est_grp2, N_pow_fit_est_grp2, t_pow_est_grp2, p_pow_est_grp2, se_pow_est_grp2, n_pow_est_grp2, dof_pow_est_grp2),\
        (k_exp_est_grp2, r_exp_est_grp2, R2_exp_est_grp2, N_exp_fit_est_grp2, t_exp_est_grp2, p_exp_est_grp2, se_exp_est_grp2, n_exp_est_grp2, dof_exp_est_grp2), D_fit_est_grp2\
        = calc_fitting(unique_sizes_estimate_group2, cum_counts_estimate_group2)
    
    (k_pow_grp2_3, r_pow_grp2_3, R2_pow_grp2_3, N_pow_fit_grp2_3, t_pow_grp2_3, p_pow_grp2_3, se_pow_grp2_3, n_pow_grp2_3, dof_pow_grp2_3),\
        (k_exp_grp2_3, r_exp_grp2_3, R2_exp_grp2_3, N_exp_fit_grp2_3, t_exp_grp2_3, p_exp_grp2_3, se_exp_grp2_3, n_exp_grp2_3, dof_exp_grp2_3), D_fit_grp2_3\
        = calc_fitting(unique_sizes_group2_3, cum_counts_group2_3)
    
    # ------------------------------------------------------------------
    # 9. プロット: Group2-3のみのフィッティング（3種類のスケール）
    # ------------------------------------------------------------------
    # 従来手法のフィッティングプロットは削除（フィッティングなしのlinear-linearプロットのみ出力ディレクトリ直下に保存済み）
    # Group2推定のフィッティング比較プロットも削除

    # 9.1 べき則フィット（Group2-3のみ）
    p_str_pow_grp2_3 = format_p_value(p_pow_grp2_3)  # p値の書式設定
    fit_lines_pow_grp2_3 = [{
        'x': D_fit_grp2_3, 'y': N_pow_fit_grp2_3,
        'label': f'Power-law: k={k_pow_grp2_3:.2e}, r={r_pow_grp2_3:.3f}, R²={R2_pow_grp2_3:.4f}, {p_str_pow_grp2_3}',
        'color': 'red', 'linestyle': '--'
    }]

    for scale in ['linear', 'semilog', 'loglog']:
        output_path = os.path.join(output_dir_power_2_3, f'RSFD_power_law_fit_group2-3_{scale}')
        create_rsfd_plot(
            unique_sizes_group2_3, cum_counts_group2_3,
            'Rock size [cm]', 'Cumulative number of rocks',
            output_path, scale_type=scale,
            fit_lines=fit_lines_pow_grp2_3,
            marker='o', linestyle='', label='Data (Group2-3)',
            show_plot=False,
            xlim=(0, 50)
        )


    # 9.2 Group比較プロット（Group1-3 vs Group2-3、area normalizeなし）
    for scale in ['linear', 'semilog', 'loglog']:
        output_path = os.path.join(output_dir_group_comparison, f'RSFD_group_comparison_{scale}')

        plt.figure(figsize=(10, 8))

        # データ点（Group1-3のみ表示）
        plt.plot(unique_sizes_estimate_group2, cum_counts_estimate_group2,
                marker='o', linestyle='', color='black', label='Data')

        # Group 1-3のフィット線
        plt.plot(D_fit_est_grp2, N_pow_fit_est_grp2,
                linestyle='--', linewidth=1.5, color='blue',
                label=f'Group1-3 fit: k={k_pow_est_grp2:.2e}, r={r_pow_est_grp2:.3f}, R²={R2_pow_est_grp2:.4f}, {p_str_pow_est_grp2}')

        # Group 2-3のフィット線
        plt.plot(D_fit_grp2_3, N_pow_fit_grp2_3,
                linestyle='--', linewidth=1.5, color='red',
                label=f'Group2-3 fit: k={k_pow_grp2_3:.2e}, r={r_pow_grp2_3:.3f}, R²={R2_pow_grp2_3:.4f}, {p_str_pow_grp2_3}')

        # 軸スケール設定
        if scale == 'semilog':
            plt.yscale('log')
        elif scale == 'loglog':
            plt.xscale('log')
            plt.yscale('log')

        # x軸範囲
        if scale == 'loglog':
            plt.xlim(max(0, 0.5), 50)
        else:
            plt.xlim(0, 50)

        # 軸ラベルとグリッド
        plt.xlabel('Rock size [cm]', fontsize=20)
        plt.ylabel('Cumulative number of rocks', fontsize=20)
        plt.tick_params(labelsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)

        # y軸のtick設定（最大値1-20の場合は2刻みに固定）
        ax = plt.gca()
        ylim = ax.get_ylim()
        if 1 <= ylim[1] <= 20:
            ax.yaxis.set_major_locator(MultipleLocator(2))

        plt.tight_layout()

        # 保存
        plt.savefig(f'{output_path}.png', dpi=300)
        plt.savefig(f'{output_path}.pdf', dpi=600)
        plt.close()

        print(f'Group比較プロット保存: {output_path}.png')

        # Legend情報の保存
        legend_entries = [
            {'label': 'Data', 'color': 'black', 'marker': 'o', 'linestyle': ''},
            {'label': f'Group1-3 fit: k={k_pow_est_grp2:.2e}, r={r_pow_est_grp2:.3f}, R²={R2_pow_est_grp2:.4f}, {p_str_pow_est_grp2}',
             'color': 'blue', 'linestyle': '--', 'linewidth': 1.5},
            {'label': f'Group2-3 fit: k={k_pow_grp2_3:.2e}, r={r_pow_grp2_3:.3f}, R²={R2_pow_grp2_3:.4f}, {p_str_pow_grp2_3}',
             'color': 'red', 'linestyle': '--', 'linewidth': 1.5}
        ]
        save_legend_info_to_txt(output_path, legend_entries)
        save_legend_only_pdf(output_path, legend_entries)

    # TXT保存（Group2-3）
    with open(os.path.join(output_dir, 'RSFD_linear_group2-3.txt'), 'w') as f:
        f.write('# size_cm\tcumulative_count\n')
        for s, n in zip(unique_sizes_group2_3, cum_counts_group2_3):
            f.write(f'{s:.3f}\t{n}\n')
    print('Group2-3累積データTXT保存: RSFD_linear_group2-3.txt')

    # ------------------------------------------------------------------
    # 9.4 面積規格化フィッティング（Group2推定）
    # ------------------------------------------------------------------
    (k_pow_area_est, r_pow_area_est, R2_pow_area_est, N_pow_fit_area_est,
     t_pow_area_est, p_pow_area_est, se_pow_area_est, n_pow_area_est, dof_pow_area_est), \
    D_fit_area_est, cum_counts_area_est = calc_fitting_area_normalized(
        unique_sizes_estimate_group2, cum_counts_estimate_group2, area)

    p_str_pow_area_est = format_p_value(p_pow_area_est)
    fit_lines_pow_area_est = [{
        'x': D_fit_area_est, 'y': N_pow_fit_area_est,
        'label': f'Power-law: k={k_pow_area_est:.2e}, r={r_pow_area_est:.3f}, R²={R2_pow_area_est:.4f}, {p_str_pow_area_est}',
        'color': 'red', 'linestyle': '--'
    }]

    for scale in ['linear', 'semilog', 'loglog']:
        output_path = os.path.join(output_dir_area_normalized_est, f'RSFD_area_normalized_estimate_group2_{scale}')
        create_rsfd_plot(
            unique_sizes_estimate_group2, cum_counts_area_est,
            'Rock size [cm]', 'Cumulative number of rocks /m²',
            output_path, scale_type=scale,
            fit_lines=fit_lines_pow_area_est,
            marker='o', linestyle='', label='Data (Estimate Group2, Area-normalized)',
            show_plot=(scale == 'linear'),
            xlim=(0, 50)
        )

    # TXT保存（面積規格化Group2推定）
    with open(os.path.join(output_dir, 'RSFD_area_normalized_estimate_group2.txt'), 'w') as f:
        f.write(f'# Area: {area} m²\n')
        f.write('# size_cm\tcumulative_count_per_m2\n')
        for s, n in zip(unique_sizes_estimate_group2, cum_counts_area_est):
            f.write(f'{s:.3f}\t{n:.6f}\n')
    print('面積規格化Group2推定累積データTXT保存: RSFD_area_normalized_estimate_group2.txt')

    # ------------------------------------------------------------------
    # 9.5 面積規格化フィッティング（Group2-3）
    # ------------------------------------------------------------------
    (k_pow_area_2_3, r_pow_area_2_3, R2_pow_area_2_3, N_pow_fit_area_2_3,
     t_pow_area_2_3, p_pow_area_2_3, se_pow_area_2_3, n_pow_area_2_3, dof_pow_area_2_3), \
    D_fit_area_2_3, cum_counts_area_2_3 = calc_fitting_area_normalized(
        unique_sizes_group2_3, cum_counts_group2_3, area)

    p_str_pow_area_2_3 = format_p_value(p_pow_area_2_3)
    fit_lines_pow_area_2_3 = [{
        'x': D_fit_area_2_3, 'y': N_pow_fit_area_2_3,
        'label': f'Power-law: k={k_pow_area_2_3:.2e}, r={r_pow_area_2_3:.3f}, R²={R2_pow_area_2_3:.4f}, {p_str_pow_area_2_3}',
        'color': 'red', 'linestyle': '--'
    }]

    for scale in ['linear', 'semilog', 'loglog']:
        output_path = os.path.join(output_dir_area_normalized_2_3, f'RSFD_area_normalized_group2-3_{scale}')
        create_rsfd_plot(
            unique_sizes_group2_3, cum_counts_area_2_3,
            'Rock size [cm]', 'Cumulative number of rocks /m²',
            output_path, scale_type=scale,
            fit_lines=fit_lines_pow_area_2_3,
            marker='o', linestyle='', label='Data (Group2-3, Area-normalized)',
            show_plot=(scale == 'linear'),
            xlim=(0, 50)
        )

    # TXT保存（面積規格化Group2-3）
    with open(os.path.join(output_dir, 'RSFD_area_normalized_group2-3.txt'), 'w') as f:
        f.write(f'# Area: {area} m²\n')
        f.write('# size_cm\tcumulative_count_per_m2\n')
        for s, n in zip(unique_sizes_group2_3, cum_counts_area_2_3):
            f.write(f'{s:.3f}\t{n:.6f}\n')
    print('面積規格化Group2-3累積データTXT保存: RSFD_area_normalized_group2-3.txt')

    # ------------------------------------------------------------------
    # 9.6 Group比較プロット（Group1-3 vs Group2-3、area normalizeあり）
    # ------------------------------------------------------------------
    for scale in ['linear', 'semilog', 'loglog']:
        output_path = os.path.join(output_dir_group_comparison_area, f'RSFD_group_comparison_area_normalized_{scale}')

        plt.figure(figsize=(10, 8))

        # データ点（Group1-3のみ表示、面積規格化）
        plt.plot(unique_sizes_estimate_group2, cum_counts_area_est,
                marker='o', linestyle='', color='black', label='Data')

        # Group 1-3のフィット線（面積規格化）
        plt.plot(D_fit_area_est, N_pow_fit_area_est,
                linestyle='--', linewidth=1.5, color='blue',
                label=f'Group1-3 fit: k={k_pow_area_est:.2e}, r={r_pow_area_est:.3f}, R²={R2_pow_area_est:.4f}, {p_str_pow_area_est}')

        # Group 2-3のフィット線（面積規格化）
        plt.plot(D_fit_area_2_3, N_pow_fit_area_2_3,
                linestyle='--', linewidth=1.5, color='red',
                label=f'Group2-3 fit: k={k_pow_area_2_3:.2e}, r={r_pow_area_2_3:.3f}, R²={R2_pow_area_2_3:.4f}, {p_str_pow_area_2_3}')

        # 軸スケール設定
        if scale == 'semilog':
            plt.yscale('log')
        elif scale == 'loglog':
            plt.xscale('log')
            plt.yscale('log')

        # x軸範囲
        if scale == 'loglog':
            plt.xlim(max(0, 0.5), 50)
        else:
            plt.xlim(0, 50)

        # 軸ラベルとグリッド
        plt.xlabel('Rock size [cm]', fontsize=20)
        plt.ylabel('Cumulative number of rocks /m²', fontsize=20)
        plt.tick_params(labelsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)

        # y軸のtick設定（最大値1-20の場合は2刻みに固定）
        ax = plt.gca()
        ylim = ax.get_ylim()
        if 1 <= ylim[1] <= 20:
            ax.yaxis.set_major_locator(MultipleLocator(2))

        plt.tight_layout()

        # 保存
        plt.savefig(f'{output_path}.png', dpi=300)
        plt.savefig(f'{output_path}.pdf', dpi=600)
        plt.close()

        print(f'Group比較プロット（面積規格化）保存: {output_path}.png')

        # Legend情報の保存
        legend_entries = [
            {'label': 'Data', 'color': 'black', 'marker': 'o', 'linestyle': ''},
            {'label': f'Group1-3 fit: k={k_pow_area_est:.2e}, r={r_pow_area_est:.3f}, R²={R2_pow_area_est:.4f}, {p_str_pow_area_est}',
             'color': 'blue', 'linestyle': '--', 'linewidth': 1.5},
            {'label': f'Group2-3 fit: k={k_pow_area_2_3:.2e}, r={r_pow_area_2_3:.3f}, R²={R2_pow_area_2_3:.4f}, {p_str_pow_area_2_3}',
             'color': 'red', 'linestyle': '--', 'linewidth': 1.5}
        ]
        save_legend_info_to_txt(output_path, legend_entries)
        save_legend_only_pdf(output_path, legend_entries)

    # ------------------------------------------------------------------
    # 12. フィッティングサマリーファイルの出力
    # ------------------------------------------------------------------
    summary_file_path = os.path.join(output_dir, 'RSFD_fitting_summary.txt')
    with open(summary_file_path, 'w') as f:
        # ヘッダー (タブ区切り)
        f.write('DataSet\tModel\tk\tr\tR_squared\tr_StdErr\tr_t_value\tr_p_value\tN_points\tDOF\n')
    
        # 1. 従来手法 (Traditional)
        f.write(f'Traditional\tPower\t{k_pow_trad:.4e}\t{r_pow_trad:.4f}\t{R2_pow_trad:.4f}\t'
                f'{se_pow_trad:.4f}\t{t_pow_trad:.3f}\t{p_pow_trad:.4e}\t{n_pow_trad}\t{dof_pow_trad}\n')
        f.write(f'Traditional\tExponential\t{k_exp_trad:.4e}\t{r_exp_trad:.4f}\t{R2_exp_trad:.4f}\t'
                f'{se_exp_trad:.4f}\t{t_exp_trad:.3f}\t{p_exp_trad:.4e}\t{n_exp_trad}\t{dof_exp_trad}\n')
    
        # 2. Group2 推定 (Estimate_Group2)
        f.write(f'Estimate_Group2\tPower\t{k_pow_est_grp2:.4e}\t{r_pow_est_grp2:.4f}\t{R2_pow_est_grp2:.4f}\t'
                f'{se_pow_est_grp2:.4f}\t{t_pow_est_grp2:.3f}\t{p_pow_est_grp2:.4e}\t{n_pow_est_grp2}\t{dof_pow_est_grp2}\n')
        f.write(f'Estimate_Group2\tExponential\t{k_exp_est_grp2:.4e}\t{r_exp_est_grp2:.4f}\t{R2_exp_est_grp2:.4f}\t'
                f'{se_exp_est_grp2:.4f}\t{t_exp_est_grp2:.3f}\t{p_exp_est_grp2:.4e}\t{n_exp_est_grp2}\t{dof_exp_est_grp2}\n')
    
        # 3. Group2-3 のみ (Group2-3_Only)
        f.write(f'Group2-3_Only\tPower\t{k_pow_grp2_3:.4e}\t{r_pow_grp2_3:.4f}\t{R2_pow_grp2_3:.4f}\t'
                f'{se_pow_grp2_3:.4f}\t{t_pow_grp2_3:.3f}\t{p_pow_grp2_3:.4e}\t{n_pow_grp2_3}\t{dof_pow_grp2_3}\n')
        f.write(f'Group2-3_Only\tExponential\t{k_exp_grp2_3:.4e}\t{r_exp_grp2_3:.4f}\t{R2_exp_grp2_3:.4f}\t'
                f'{se_exp_grp2_3:.4f}\t{t_exp_grp2_3:.3f}\t{p_exp_grp2_3:.4e}\t{n_exp_grp2_3}\t{dof_exp_grp2_3}\n')

        # 4. 面積規格化 Estimate_Group2 (Area_Normalized_Est_Grp2)
        f.write(f'Area_Normalized_Est_Grp2\tPower\t{k_pow_area_est:.4e}\t{r_pow_area_est:.4f}\t{R2_pow_area_est:.4f}\t'
                f'{se_pow_area_est:.4f}\t{t_pow_area_est:.3f}\t{p_pow_area_est:.4e}\t{n_pow_area_est}\t{dof_pow_area_est}\n')

        # 5. 面積規格化 Group2-3 (Area_Normalized_Grp2-3)
        f.write(f'Area_Normalized_Grp2-3\tPower\t{k_pow_area_2_3:.4e}\t{r_pow_area_2_3:.4f}\t{R2_pow_area_2_3:.4f}\t'
                f'{se_pow_area_2_3:.4f}\t{t_pow_area_2_3:.3f}\t{p_pow_area_2_3:.4e}\t{n_pow_area_2_3}\t{dof_pow_area_2_3}\n')

        # 面積情報
        f.write(f'\n# Area: {area} m²\n')

    print(f'フィッティングサマリー保存: {summary_file_path}')

else:  # mode == '4'
    # ------------------------------------------------------------------
    # モード4: 複数範囲比較処理
    # ------------------------------------------------------------------
    print('\n=== 複数範囲比較処理を開始 ===')

    # 色のリスト（各範囲に割り当て）
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
              'olive', 'cyan', 'magenta', 'yellow', 'navy', 'teal', 'maroon']

    # 各範囲のデータを格納するリスト
    all_ranges_data_traditional = []  # 従来手法（Group2=6cm固定）
    all_ranges_data_estimate_grp2 = []  # Group2サイズ推定
    all_ranges_data_grp2_3 = []  # Group2-3のみ
    all_ranges_data_area_normalized_est = []  # 面積規格化Group2推定
    all_ranges_data_area_normalized_2_3 = []  # 面積規格化Group2-3

    # JSONデータ読み込み（一度だけ）
    with open(data_path, 'r') as f:
        results = json.load(f).get('results', {})

    x_all = np.array([v['x'] for v in results.values()])
    y_all = np.array([v['y'] for v in results.values()])
    lab_all = np.array([v['label'] for v in results.values()], dtype=int)
    time_top_all = np.array([none_to_nan(v['time_top']) for v in results.values()], dtype=float)
    time_bottom_all = np.array([none_to_nan(v['time_bottom']) for v in results.values()], dtype=float)
    print(f'ラベルデータ読み込み完了: {len(lab_all)}個')

    # 各範囲について処理
    for range_idx, range_info in enumerate(ranges_list):
        print(f'\n--- 範囲 {range_idx + 1}/{num_ranges} の処理 ---')

        # データをコピー
        x = x_all.copy()
        y = y_all.copy()
        lab = lab_all.copy()
        time_top = time_top_all.copy()
        time_bottom = time_bottom_all.copy()

        # データ範囲フィルタリング
        original_count = len(lab)
        time_min = range_info['time_min']
        time_max = range_info['time_max']
        horizontal_min = range_info['horizontal_min']
        horizontal_max = range_info['horizontal_max']
        exclude_flag = range_info.get('exclude', False)

        if time_min is not None and time_max is not None:
            mask_group1 = (lab == 1) & (y >= time_min) & (y <= time_max)
            mask_others = (lab != 1) & (time_top >= time_min) & (time_top <= time_max)
            time_mask = mask_group1 | mask_others

            # 除外フラグが立っている場合は論理反転
            if exclude_flag:
                time_mask = ~time_mask

            x = x[time_mask]
            y = y[time_mask]
            lab = lab[time_mask]
            time_top = time_top[time_mask]
            time_bottom = time_bottom[time_mask]

            if exclude_flag:
                print(f'時間範囲除外後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
            else:
                print(f'時間範囲フィルタリング後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')

        if horizontal_min is not None and horizontal_max is not None:
            horizontal_mask = (x >= horizontal_min) & (x <= horizontal_max)

            # 除外フラグが立っている場合は論理反転
            if exclude_flag:
                horizontal_mask = ~horizontal_mask

            x = x[horizontal_mask]
            y = y[horizontal_mask]
            lab = lab[horizontal_mask]
            time_top = time_top[horizontal_mask]
            time_bottom = time_bottom[horizontal_mask]

            if exclude_flag:
                print(f'水平位置範囲除外後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')
            else:
                print(f'水平位置範囲フィルタリング後: {len(lab)}個 (元データの{len(lab)/original_count*100:.1f}%)')

        print(f'フィルタリング完了: {len(lab)}個のデータを使用')

        # サイズ配列を作成
        counts = {k: int(np.sum(lab == k)) for k in range(1, 7)}
        size_label1 = np.full(counts[1], 1.0)
        size_label2 = np.full(counts[2], 6.0)
        mask2_valid = (lab == 2) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
        mask3_valid = (lab == 3) & (~np.isnan(time_top)) & (~np.isnan(time_bottom))
        er = 9.0
        c = 299_792_458
        sizes_group2 = (time_bottom[mask2_valid] - time_top[mask2_valid]) * 1e-9 * c / np.sqrt(er) * 0.5 * 100
        sizes_group3 = (time_bottom[mask3_valid] - time_top[mask3_valid]) * 1e-9 * c / np.sqrt(er) * 0.5 * 100

        # サイズ値を小数点以下3桁に丸めて、浮動小数点の微小誤差を排除
        sizes_group2 = np.round(sizes_group2, decimals=3)
        sizes_group3 = np.round(sizes_group3, decimals=3)

        # 1) 従来手法（Group2=6cm固定）
        all_sizes_traditional = np.concatenate([size_label1, size_label2, sizes_group3])
        unique_sizes_traditional, unique_counts = np.unique(all_sizes_traditional, return_counts=True)
        unique_sizes_traditional = np.sort(unique_sizes_traditional)
        cum_counts_traditional = np.array([np.sum(all_sizes_traditional >= s) for s in unique_sizes_traditional])

        # フィッティング
        (k_pow_trad, r_pow_trad, R2_pow_trad, N_pow_fit_trad, t_pow_trad, p_pow_trad, se_pow_trad, n_pow_trad, dof_pow_trad), \
        (k_exp_trad, r_exp_trad, R2_exp_trad, N_exp_fit_trad, t_exp_trad, p_exp_trad, se_exp_trad, n_exp_trad, dof_exp_trad), \
        D_fit_trad = calc_fitting(unique_sizes_traditional, cum_counts_traditional)

        # 範囲ラベル作成
        range_label_parts = []
        if range_info['time_range']:
            range_label_parts.append(f"t{time_min}-{time_max}")
        if range_info['horizontal_range']:
            range_label_parts.append(f"x{horizontal_min}-{horizontal_max}")

        # 除外フラグに応じてラベルを作成
        if range_label_parts:
            base_label = f"Range {range_idx + 1}: {', '.join(range_label_parts)}"
        else:
            base_label = f"Range {range_idx + 1}: Full"

        if exclude_flag:
            range_label = f"{base_label} (exclude)"
        else:
            range_label = base_label

        # データを保存
        color = colors[range_idx % len(colors)]
        all_ranges_data_traditional.append({
            'x_data': unique_sizes_traditional,
            'y_data': cum_counts_traditional,
            'fit_x': D_fit_trad,
            'fit_y': N_pow_fit_trad,
            'fit_params': {
                'k': k_pow_trad,
                'r': r_pow_trad,
                'R2': R2_pow_trad,
                'p_str': format_p_value(p_pow_trad)
            },
            'label': range_label,
            'color': color
        })

        # Exponential用のデータ（別途保存）
        all_ranges_data_traditional.append({
            'x_data': unique_sizes_traditional,
            'y_data': cum_counts_traditional,
            'fit_x': D_fit_trad,
            'fit_y': N_exp_fit_trad,
            'fit_params': {
                'k': k_exp_trad,
                'r': r_exp_trad,
                'R2': R2_exp_trad,
                'p_str': format_p_value(p_exp_trad)
            },
            'label': range_label,
            'color': color,
            'fit_type': 'exponential'
        })

        # 2) Group2サイズ推定
        all_sizes_estimate_group2 = np.concatenate([size_label1, sizes_group2, sizes_group3])
        unique_sizes_estimate_group2 = np.sort(np.unique(all_sizes_estimate_group2))
        cum_counts_estimate_group2 = np.array([np.sum(all_sizes_estimate_group2 >= s) for s in unique_sizes_estimate_group2])

        (k_pow_est_grp2, r_pow_est_grp2, R2_pow_est_grp2, N_pow_fit_est_grp2, t_pow_est_grp2, p_pow_est_grp2, se_pow_est_grp2, n_pow_est_grp2, dof_pow_est_grp2), \
        (k_exp_est_grp2, r_exp_est_grp2, R2_exp_est_grp2, N_exp_fit_est_grp2, t_exp_est_grp2, p_exp_est_grp2, se_exp_est_grp2, n_exp_est_grp2, dof_exp_est_grp2), \
        D_fit_est_grp2 = calc_fitting(unique_sizes_estimate_group2, cum_counts_estimate_group2)

        all_ranges_data_estimate_grp2.append({
            'x_data': unique_sizes_estimate_group2,
            'y_data': cum_counts_estimate_group2,
            'fit_x': D_fit_est_grp2,
            'fit_y': N_pow_fit_est_grp2,
            'fit_params': {
                'k': k_pow_est_grp2,
                'r': r_pow_est_grp2,
                'R2': R2_pow_est_grp2,
                'p_str': format_p_value(p_pow_est_grp2)
            },
            'label': range_label,
            'color': color
        })

        all_ranges_data_estimate_grp2.append({
            'x_data': unique_sizes_estimate_group2,
            'y_data': cum_counts_estimate_group2,
            'fit_x': D_fit_est_grp2,
            'fit_y': N_exp_fit_est_grp2,
            'fit_params': {
                'k': k_exp_est_grp2,
                'r': r_exp_est_grp2,
                'R2': R2_exp_est_grp2,
                'p_str': format_p_value(p_exp_est_grp2)
            },
            'label': range_label,
            'color': color,
            'fit_type': 'exponential'
        })

        # 3) Group2-3のみ
        all_sizes_group2_3 = np.concatenate([sizes_group2, sizes_group3])
        unique_sizes_group2_3 = np.sort(np.unique(all_sizes_group2_3))
        cum_counts_group2_3 = np.array([np.sum(all_sizes_group2_3 >= s) for s in unique_sizes_group2_3])

        (k_pow_grp2_3, r_pow_grp2_3, R2_pow_grp2_3, N_pow_fit_grp2_3, t_pow_grp2_3, p_pow_grp2_3, se_pow_grp2_3, n_pow_grp2_3, dof_pow_grp2_3), \
        (k_exp_grp2_3, r_exp_grp2_3, R2_exp_grp2_3, N_exp_fit_grp2_3, t_exp_grp2_3, p_exp_grp2_3, se_exp_grp2_3, n_exp_grp2_3, dof_exp_grp2_3), \
        D_fit_grp2_3 = calc_fitting(unique_sizes_group2_3, cum_counts_group2_3)

        all_ranges_data_grp2_3.append({
            'x_data': unique_sizes_group2_3,
            'y_data': cum_counts_group2_3,
            'fit_x': D_fit_grp2_3,
            'fit_y': N_pow_fit_grp2_3,
            'fit_params': {
                'k': k_pow_grp2_3,
                'r': r_pow_grp2_3,
                'R2': R2_pow_grp2_3,
                'p_str': format_p_value(p_pow_grp2_3)
            },
            'label': range_label,
            'color': color
        })

        all_ranges_data_grp2_3.append({
            'x_data': unique_sizes_group2_3,
            'y_data': cum_counts_group2_3,
            'fit_x': D_fit_grp2_3,
            'fit_y': N_exp_fit_grp2_3,
            'fit_params': {
                'k': k_exp_grp2_3,
                'r': r_exp_grp2_3,
                'R2': R2_exp_grp2_3,
                'p_str': format_p_value(p_exp_grp2_3)
            },
            'label': range_label,
            'color': color,
            'fit_type': 'exponential'
        })

        # 4) 面積規格化 Group2推定
        area_range = range_info.get('area', 16136)
        (k_pow_area_est, r_pow_area_est, R2_pow_area_est, N_pow_fit_area_est,
         t_pow_area_est, p_pow_area_est, se_pow_area_est, n_pow_area_est, dof_pow_area_est), \
        D_fit_area_est, cum_counts_area_est = calc_fitting_area_normalized(
            unique_sizes_estimate_group2, cum_counts_estimate_group2, area_range)

        all_ranges_data_area_normalized_est.append({
            'x_data': unique_sizes_estimate_group2,
            'y_data': cum_counts_area_est,
            'fit_x': D_fit_area_est,
            'fit_y': N_pow_fit_area_est,
            'fit_params': {
                'k': k_pow_area_est,
                'r': r_pow_area_est,
                'R2': R2_pow_area_est,
                'p_str': format_p_value(p_pow_area_est),
                'area': area_range
            },
            'label': range_label,
            'color': color
        })

        # 5) 面積規格化 Group2-3
        (k_pow_area_2_3, r_pow_area_2_3, R2_pow_area_2_3, N_pow_fit_area_2_3,
         t_pow_area_2_3, p_pow_area_2_3, se_pow_area_2_3, n_pow_area_2_3, dof_pow_area_2_3), \
        D_fit_area_2_3, cum_counts_area_2_3 = calc_fitting_area_normalized(
            unique_sizes_group2_3, cum_counts_group2_3, area_range)

        all_ranges_data_area_normalized_2_3.append({
            'x_data': unique_sizes_group2_3,
            'y_data': cum_counts_area_2_3,
            'fit_x': D_fit_area_2_3,
            'fit_y': N_pow_fit_area_2_3,
            'fit_params': {
                'k': k_pow_area_2_3,
                'r': r_pow_area_2_3,
                'R2': R2_pow_area_2_3,
                'p_str': format_p_value(p_pow_area_2_3),
                'area': area_range
            },
            'label': range_label,
            'color': color
        })

    # ------------------------------------------------------------------
    # プロット生成
    # ------------------------------------------------------------------
    print('\n=== 比較プロット生成中 ===')

    # 従来手法とGroup2推定の比較プロットは削除（フィッティングなしのlinear-linearプロットのみ出力ディレクトリ直下に保存）
    # Group2-3のみの比較プロットと面積規格化比較プロットのみ実施

    # Group2-3のみのプロット
    power_data_grp2_3 = [d for d in all_ranges_data_grp2_3 if 'fit_type' not in d]
    for scale in ['linear', 'semilog', 'loglog']:
        output_path = os.path.join(output_dir_power_2_3, f'RSFD_power_law_comparison_group2-3_{scale}')
        create_multi_range_comparison_plot(
            power_data_grp2_3, 'Rock size [cm]', 'Cumulative number of rocks',
            output_path, scale_type=scale, fit_type='Power-law',
            show_plot=False, xlim=(0, 50)
        )

    # 面積規格化Group2推定のプロット
    for scale in ['linear', 'semilog', 'loglog']:
        output_path = os.path.join(output_dir_area_normalized_est, f'RSFD_area_normalized_estimate_group2_comparison_{scale}')
        create_multi_range_comparison_plot(
            all_ranges_data_area_normalized_est, 'Rock size [cm]', 'Cumulative number of rocks /m²',
            output_path, scale_type=scale, fit_type='Power-law',
            show_plot=False, xlim=(0, 50)
        )

    # 面積規格化Group2-3のプロット
    for scale in ['linear', 'semilog', 'loglog']:
        output_path = os.path.join(output_dir_area_normalized_2_3, f'RSFD_area_normalized_group2-3_comparison_{scale}')
        create_multi_range_comparison_plot(
            all_ranges_data_area_normalized_2_3, 'Rock size [cm]', 'Cumulative number of rocks /m²',
            output_path, scale_type=scale, fit_type='Power-law',
            show_plot=False, xlim=(0, 50)
        )

    # 両方のフィッティング比較プロットは複雑になるため、Power lawのみで実施
    print('複数範囲比較プロット完了')

# ------------------------------------------------------------------
# 13. 設定ファイルの保存（新規処理の場合のみ）
# ------------------------------------------------------------------
if startup_mode == '1':
    config_path = os.path.join(base_dir, 'processing_config.json')
    if mode == '4':
        save_processing_config(
            config_path, data_path, mode, time_range, horizontal_range,
            time_min, time_max, horizontal_min, horizontal_max, output_dir,
            num_ranges=num_ranges, ranges_list=ranges_list, area=None
        )
    else:
        save_processing_config(
            config_path, data_path, mode, time_range, horizontal_range,
            time_min, time_max, horizontal_min, horizontal_max, output_dir,
            area=area
        )

print('\nすべて完了しました！')