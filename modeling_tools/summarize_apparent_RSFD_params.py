import json
import glob
import os
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DIR = input('summary.jsonを含むベースディレクトリのパス：').strip()

# LPR観測値
# 'range' は対応する解析深さ範囲 (d_start, d_end)。None は全深さ範囲を表す。
# 読み込んだ summary JSON の深さ範囲と一致する観測値のみを比較に使用する。
LPR_obs = [
    {'label': 'Full range', 'range': None,        'r': 1.08, 'k': 7.21e-3},
    {'label': '0-3 m',      'range': (0.0, 3.0),  'r': 1.92, 'k': 19.3e-3},
    {'label': '3-12 m',     'range': (3.0, 12.0), 'r': 1.00, 'k': 5.68e-3},
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


def parse_depth_range(suffix):
    """
    JSONファイル名の接尾辞から解析深さ範囲を取得する。
      ""            -> None            (全深さ範囲)
      "_0.0-3.0m"   -> (0.0, 3.0)
    解釈できない場合は 'unknown' を返す。
    """
    if suffix == '':
        return None
    m = re.fullmatch(r'_([0-9.]+)-([0-9.]+)m', suffix)
    if not m:
        return 'unknown'
    return (float(m.group(1)), float(m.group(2)))


def select_observation(depth_range):
    """解析深さ範囲に対応する LPR 観測値を1つだけ返す。無ければ None。"""
    if depth_range == 'unknown':
        return None
    for obs in LPR_obs:
        if obs['range'] is None and depth_range is None:
            return obs
        if obs['range'] is not None and depth_range is not None:
            if (abs(obs['range'][0] - depth_range[0]) < 1e-6 and
                    abs(obs['range'][1] - depth_range[1]) < 1e-6):
                return obs
    return None


DEPTH_RANGE = parse_depth_range(suffix)
OBS = select_observation(DEPTH_RANGE)

output_dir = os.path.join(BASE_DIR, plot_dir_name)
print(f"\n選択ファイル : {json_file_path}")
print(f"出力ディレクトリ: {output_dir}")
if DEPTH_RANGE == 'unknown':
    print("[警告] ファイル名から深さ範囲を判定できませんでした。"
          "apparent_r_vs_k プロットはスキップされます。")
elif OBS is None:
    rng = "全深さ範囲" if DEPTH_RANGE is None else f"{DEPTH_RANGE[0]}-{DEPTH_RANGE[1]} m"
    print(f"[警告] 深さ範囲 {rng} に対応する LPR 観測値が LPR_obs にありません。"
          "apparent_r_vs_k プロットはスキップされます。")
else:
    print(f"比較する観測値: {OBS['label']} (r={OBS['r']}, k={OBS['k']:.3e})")
os.makedirs(output_dir, exist_ok=True)

with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 横軸となる r の値を取得し、数値として昇順にソートする
r_keys = sorted(data.keys(), key=float)


def detect_input_mode(data):
    """
    JSON から入力モード（'N' or 'K'）を判定する。
      1. エントリに input_mode フィールドがあればそれを使う（新形式）
      2. 無ければケースキーの形式から推定する
         "1000" のように全て整数として解釈できれば N系、
         "0.005" のような非整数が含まれれば K系（旧形式へのフォールバック）
    """
    for r in data:
        for entry in data[r].values():
            if isinstance(entry, dict) and 'input_mode' in entry:
                return entry['input_mode']

    for r in data:
        for key in data[r].keys():
            try:
                int(key)
            except ValueError:
                return 'K'
    return 'N'


INPUT_MODE = detect_input_mode(data)
CASE_SYMBOL = 'k' if INPUT_MODE == 'K' else 'N'
print(f"入力モード   : {INPUT_MODE} "
      f"({'k_true 入力系' if INPUT_MODE == 'K' else 'N_true 入力系'})")


def case_label(key):
    """凡例ラベル。N系は 'N = 1000'、K系は 'k = 0.005' となる。"""
    return f"{CASE_SYMBOL} = {key}"


# ケース（N または k）のリストを取得し、数値として昇順にソートする
# 全 r のキーを横断して収集（深さ範囲JSONでは r によって存在するケースが異なるため）
all_case_keys = set()
for r in r_keys:
    for case_key in data[r].keys():
        try:
            float(case_key)
        except ValueError:
            continue
        all_case_keys.add(case_key)
case_keys = sorted(all_case_keys, key=float)
print(f"ケース ({CASE_SYMBOL}) : {case_keys}")

# Calculation stop の有無を確認
calculation_stops = []
for r in r_keys:
    for case_key in case_keys:
        if case_key in data[r] and data[r][case_key] == "Calculation stop":
            calculation_stops.append((r, case_key))
print(f"Calculation stops found for: {calculation_stops}")


# プロット線用のリスト
line_colors  = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange']
line_markers = ['o', 's', 'D', '^', 'v', 'P', '*']
line_styles  = ['-', '--', '-.', ':']

# 色の割り当て
#   r に関するプロット（横軸 r_true）では、同一ケース(k または N)を同じ色でまとめる
#   k に関するプロット（横軸 k_true）では、同一 r を同じ色でまとめる
case_colors = {c: line_colors[i % len(line_colors)] for i, c in enumerate(case_keys)}
r_colors    = {r: line_colors[i % len(line_colors)] for i, r in enumerate(r_keys)}
case_markers = {c: line_markers[i % len(line_markers)] for i, c in enumerate(case_keys)}
r_markers    = {r: line_markers[i % len(line_markers)] for i, r in enumerate(r_keys)}


def get_entry(r_key, case_key):
    """有効な (r, ケース) のエントリを返す。欠損・計算中止・NaN の場合は None。"""
    if case_key not in data[r_key]:
        return None
    if (r_key, case_key) in calculation_stops:
        return None
    entry = data[r_key][case_key]
    if not isinstance(entry, dict):
        return None
    return entry


def r_line_label(r_key):
    return f"r = {float(r_key):.1f}"


# =========================================================
# 1-4. true vs apparent プロット
#      param='r': 横軸 r_true、線＝同一ケース(k または N) を同じ色
#      param='k': 横軸 k_true、線＝同一 r を同じ色
#      ratio=True で縦軸を apparent / true にする
# =========================================================
def plot_true_vs_apparent(param, ratio, out_name):
    if param == 'r':
        line_keys = case_keys                       # 1本の線＝1つのケース(k または N)
        x_keys    = r_keys                          # 線に沿って並ぶのは r
        fetch     = lambda line_key, x_key: get_entry(x_key, line_key)
        colors, markers = case_colors, case_markers
        label_of  = case_label
        log_axis  = False
        axis_name = 'r'
    else:
        line_keys = r_keys                          # 1本の線＝1つの r
        x_keys    = case_keys                       # 線に沿って並ぶのはケース(k または N)
        fetch     = lambda line_key, x_key: get_entry(line_key, x_key)
        colors, markers = r_colors, r_markers
        label_of  = r_line_label
        log_axis  = True
        axis_name = 'k'

    plt.figure(figsize=(8, 6))
    all_x, all_y = [], []

    for idx, line_key in enumerate(line_keys):
        x_vals, xerrs, y_vals, yerrs = [], [], [], []

        for x_key in x_keys:
            entry = fetch(line_key, x_key)
            if entry is None:
                continue

            x_mean = entry[f"{axis_name}_true_mean"]
            x_std  = entry[f"{axis_name}_true_std"]
            y_mean = entry[f"{axis_name}_apparent_mean"]
            y_std  = entry[f"{axis_name}_apparent_std"]

            if x_mean is None or y_mean is None:
                continue
            if not (x_mean == x_mean and y_mean == y_mean):   # NaN 判定
                continue

            if ratio:
                if x_mean == 0:
                    continue
                y_mean = y_mean / x_mean
                y_std  = y_std / x_mean

            x_vals.append(x_mean); xerrs.append(x_std)
            y_vals.append(y_mean); yerrs.append(y_std)

        if not x_vals:
            continue

        order = sorted(range(len(x_vals)), key=lambda i: x_vals[i])
        x_vals = [x_vals[i] for i in order]; xerrs = [xerrs[i] for i in order]
        y_vals = [y_vals[i] for i in order]; yerrs = [yerrs[i] for i in order]

        all_x += x_vals
        all_y += y_vals

        plt.errorbar(
            x_vals, y_vals, xerr=xerrs, yerr=yerrs,
            label=label_of(line_key),
            marker=markers[line_key],
            capsize=5,
            linestyle=line_styles[idx % len(line_styles)],
            color=colors[line_key]
        )

    # 参照線
    if ratio:
        plt.axhline(1, color='k', linestyle='--', label='Apparent / True = 1')
    elif all_x and all_y:
        lo = min(min(all_x), min(all_y)) * 0.8
        hi = max(max(all_x), max(all_y)) * 1.2
        plt.plot([lo, hi], [lo, hi], 'k--', label='Apparent = True')

    if log_axis:
        plt.xscale('log')
        if not ratio:
            plt.yscale('log')

    unit = ' [1/m$^2$]' if axis_name == 'k' else ''
    plt.xlabel(f'True {axis_name}{unit}', fontsize=16)
    if ratio:
        plt.ylabel(f'Apparent {axis_name} / True {axis_name}', fontsize=16)
    else:
        plt.ylabel(f'Apparent {axis_name}{unit}', fontsize=16)

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{out_name}.png'))
    plt.savefig(os.path.join(output_dir, f'{out_name}.pdf'))
    plt.show()


# 1. r_true vs r_apparent            （線＝同一ケース）
plot_true_vs_apparent('r', ratio=False, out_name='r_apparent_plot')
# 2. r_true vs r_apparent / r_true   （線＝同一ケース）
plot_true_vs_apparent('r', ratio=True,  out_name='r_apparent_ratio_plot')
# 3. k_true vs k_apparent            （線＝同一 r）
plot_true_vs_apparent('k', ratio=False, out_name='k_apparent_plot')
# 4. k_true vs k_apparent / k_true   （線＝同一 r）
plot_true_vs_apparent('k', ratio=True,  out_name='k_apparent_ratio_plot')


# =========================================================
# 5. apparent r vs apparent k
#    読み込んだ summary JSON の深さ範囲に対応する LPR 観測値のみを描画する
#    色: input_r ごと、マーカー: ケース(N または k) ごと
# =========================================================
if OBS is None:
    print("apparent_r_vs_k プロットはスキップしました（対応する観測値なし）。")
else:
    r_obs = OBS['r']
    k_obs = OBS['k']

    # 軸範囲: 従来の既定値を基準に、データと観測値が収まるよう必要に応じて拡張する
    x_lo, x_hi = 0.2, 4.0
    y_lo, y_hi = 3e-4, 0.5
    _r_all, _k_all = [r_obs], [k_obs]
    for r in r_keys:
        for c in case_keys:
            entry = get_entry(r, c)
            if entry is None:
                continue
            _r_all.append(entry["r_apparent_mean"])
            _k_all.append(entry["k_apparent_mean"])
    _r_all = [v for v in _r_all if v == v]
    _k_all = [v for v in _k_all if v == v and v > 0]
    if _r_all:
        x_lo = min(x_lo, min(_r_all) * 0.9)
        x_hi = max(x_hi, max(_r_all) * 1.1)
    if _k_all:
        y_lo = min(y_lo, min(_k_all) * 0.5)
        y_hi = max(y_hi, max(_k_all) * 2.0)

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in r_keys:
        color = r_colors[r]

        # 同じ input_r の点を共通線スタイルで接続
        pts = [(get_entry(r, c)["r_apparent_mean"], get_entry(r, c)["k_apparent_mean"])
               for c in case_keys if get_entry(r, c) is not None]
        pts = [(x, y) for x, y in pts if x == x and y == y]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=color, linestyle='-', linewidth=1.5, zorder=1)

        # 各点をマーカー＋エラーバーでプロット
        for case_key in case_keys:
            entry = get_entry(r, case_key)
            if entry is None:
                continue
            ax.errorbar(
                entry["r_apparent_mean"],
                entry["k_apparent_mean"],
                xerr=entry["r_apparent_std"],
                yerr=entry["k_apparent_std"],
                marker=case_markers[case_key],
                markersize=8,
                color=color,
                capsize=5,
                linestyle='none',
                zorder=2
            )

    ax.axhline(k_obs, color='k', linestyle='--', linewidth=2.5)
    ax.axvline(r_obs, color='k', linestyle='--', linewidth=2.5)

    # カスタム凡例: 色→input_r、マーカー→ケース(N または k)
    legend_r   = [Line2D([0], [0], color=r_colors[r], linestyle='-', linewidth=1.5,
                         label=r_line_label(r)) for r in r_keys]
    legend_n   = [Line2D([0], [0], color='gray', marker=case_markers[c], linestyle='none',
                         markersize=8, label=case_label(c)) for c in case_keys]
    legend_obs = [Line2D([0], [0], color='k', linestyle='--', linewidth=2.5,
                         label=f"Observed \n(r={r_obs}, k={k_obs:.2e})")]
    handles_list = legend_r + legend_n + legend_obs
    ax.legend(handles=handles_list, fontsize=14,
              bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

    ax.set_xlabel('Apparent r', fontsize=16)
    ax.set_ylabel('Apparent k [/m$^2$]', fontsize=16)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'apparent_r_vs_k.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'apparent_r_vs_k.pdf'), bbox_inches='tight')
    plt.show()

    # --- 追加: ラベル（凡例）なしプロット (figsize=(8, 6)) ---
    fig_nl, ax_nl = plt.subplots(figsize=(8, 6))
    for r in r_keys:
        color = r_colors[r]
        pts = [(get_entry(r, c)["r_apparent_mean"], get_entry(r, c)["k_apparent_mean"])
               for c in case_keys if get_entry(r, c) is not None]
        pts = [(x, y) for x, y in pts if x == x and y == y]
        if pts:
            ax_nl.plot([p[0] for p in pts], [p[1] for p in pts],
                       color=color, linestyle='-', linewidth=1.5, zorder=1)

        for case_key in case_keys:
            entry = get_entry(r, case_key)
            if entry is None:
                continue
            ax_nl.errorbar(
                entry["r_apparent_mean"],
                entry["k_apparent_mean"],
                xerr=entry["r_apparent_std"],
                yerr=entry["k_apparent_std"],
                marker=case_markers[case_key],
                markersize=8,
                color=color,
                capsize=5,
                linestyle='none',
                zorder=2
            )

    ax_nl.axhline(k_obs, color='k', linestyle='--', linewidth=2.5)
    ax_nl.axvline(r_obs, color='k', linestyle='--', linewidth=2.5)
    ax_nl.set_xlabel('Apparent r', fontsize=16)
    ax_nl.set_ylabel('Apparent k [1/m$^2$]', fontsize=16)
    ax_nl.set_xlim(x_lo, x_hi)
    ax_nl.set_ylim(y_lo, y_hi)
    ax_nl.tick_params(axis='both', which='major', labelsize=14)
    ax_nl.set_yscale('log')
    ax_nl.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'apparent_r_vs_k_no_legend.png'))
    plt.savefig(os.path.join(output_dir, 'apparent_r_vs_k_no_legend.pdf'))
    plt.show()

    # --- 追加: ラベル（凡例）のみの画像 ---
    fig_leg = plt.figure(figsize=(4, 6))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis('off')
    ax_leg.legend(handles=handles_list, fontsize=14, loc='center', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'apparent_r_vs_k_legend_only.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'apparent_r_vs_k_legend_only.pdf'), bbox_inches='tight')
    plt.show()