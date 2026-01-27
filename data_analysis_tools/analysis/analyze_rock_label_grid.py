#!/usr/bin/env python3
"""
岩石ラベルグリッド解析ツール
JSONファイルから岩石ラベルを読み込み、指定されたウィンドウサイズでグリッド分割し、
各グリッドセル内でのラベル分布を円グラフ/棒グラフで可視化

機能:
- 2次元グリッドベースのラベル集計
- 各セルでの円グラフ表示（ラベル数を直接表示）
- 3パターンの集計（岩石のみ1-3、非岩石のみ4-6、全ラベル1-6）
- 統計データのテキスト出力
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_label_data(json_path):
    """
    JSONファイルからラベルデータを読み込む
    
    Parameters:
    -----------
    json_path : str
        JSONファイルのパス
    
    Returns:
    --------
    dict: {'x': array, 'y': array, 'label': array}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', {})
    if not results:
        raise ValueError("JSONファイルに'results'データが見つかりません")
    
    x_coords = []
    y_coords = []
    labels = []
    
    for value in results.values():
        x_coords.append(value['x'])
        y_coords.append(value['y'])
        labels.append(value['label'])
    
    return {
        'x': np.array(x_coords),
        'y': np.array(y_coords),
        'label': np.array(labels, dtype=int)
    }

def load_bscan_data(bscan_path):
    """
    B-scanデータファイルを読み込む
    
    Parameters:
    -----------
    bscan_path : str
        B-scanデータファイルのパス
    
    Returns:
    --------
    np.ndarray: B-scanデータ (time x traces)
    """
    if not os.path.exists(bscan_path):
        raise FileNotFoundError(f"B-scanファイルが見つかりません: {bscan_path}")
    
    print(f"B-scanデータを読み込み中: {bscan_path}")
    data = np.loadtxt(bscan_path, delimiter=' ')
    print("B-scanデータの読み込みが完了しました。")
    
    # NaN値の統計を表示
    nan_count = np.sum(np.isnan(data))
    total_count = data.size
    if nan_count > 0:
        print(f"NaN値検出: {nan_count} / {total_count} ({nan_count/total_count*100:.2f}%)")
    else:
        print("NaN値は検出されませんでした。")
    
    return data

def find_bscan_path(json_path):
    """
    JSONファイルのパスからB-scanデータファイルのパスを推定する
    
    Parameters:
    -----------
    json_path : str
        JSONファイルのパス
    
    Returns:
    --------
    str or None: B-scanデータファイルのパス（見つからない場合はNone）
    """
    # JSONファイルが格納されているディレクトリの親ディレクトリを取得
    json_dir = os.path.dirname(json_path)
    parent_dir = os.path.dirname(json_dir)
    
    # 親ディレクトリ内でB-scanデータファイルを探す
    # パターン: ディレクトリ名+.txt
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path):
            # ディレクトリ名と同じ名前の.txtファイルを探す
            potential_bscan = os.path.join(parent_dir, f"{item}.txt")
            if os.path.exists(potential_bscan):
                return potential_bscan
    
    return None

def get_bscan_data_path(json_path):
    """
    B-scanデータファイルのパスを取得する（自動検出 + 手動指定オプション）
    
    Parameters:
    -----------
    json_path : str
        JSONファイルのパス
    
    Returns:
    --------
    str or None: B-scanデータファイルのパス（None=スキップ）
    """
    # 自動検出を試行
    auto_path = find_bscan_path(json_path)
    
    if auto_path:
        print(f"B-scanデータファイルを自動検出しました: {auto_path}")
        use_auto = input("このファイルを使用しますか？ (y/n/skip): ").strip().lower()
        
        if use_auto == 'y':
            return auto_path
        elif use_auto == 'skip':
            print("B-scan背景表示をスキップします")
            return None
    else:
        print("B-scanデータファイルの自動検出に失敗しました")
    
    # 手動指定
    print("B-scanデータファイルのパスを手動で指定してください")
    manual_path = input("B-scanデータファイルのパス（空=スキップ）: ").strip()
    
    if not manual_path:
        print("B-scan背景表示をスキップします")
        return None
    
    if not os.path.exists(manual_path):
        print(f"警告: 指定されたファイルが見つかりません: {manual_path}")
        return None
    
    return manual_path

def time_to_depth(time_ns, epsilon_r=4.5):
    """
    時間[ns]を深さ[m]に変換
    
    Parameters:
    -----------
    time_ns : array
        時間 [ns]
    epsilon_r : float
        比誘電率（デフォルト: 4.5）
    
    Returns:
    --------
    array: 深さ [m]
    """
    c = 299792458  # 光速 [m/s]
    depth_m = time_ns * 1e-9 * c / np.sqrt(epsilon_r) * 0.5
    return depth_m

def depth_to_time(depth_m, epsilon_r=4.5):
    """
    深さ[m]を時間[ns]に変換
    
    Parameters:
    -----------
    depth_m : array
        深さ [m]
    epsilon_r : float
        比誘電率（デフォルト: 4.5）
    
    Returns:
    --------
    array: 時間 [ns]
    """
    c = 299792458  # 光速 [m/s]
    time_ns = depth_m * 2 * np.sqrt(epsilon_r) / c * 1e9
    return time_ns

def get_label_color(label):
    """
    ラベル番号に対応する色を返す
    
    Parameters:
    -----------
    label : int
        ラベル番号 (1-6)
    
    Returns:
    --------
    str: 色名
    """
    color_map = {
        1: 'red',        # red
        2: 'green',      # green
        3: 'blue',       # blue
        4: 'cyan',       # cyan
        5: 'magenta',    # magenta
        6: 'yellow'      # yellow
    }
    return color_map.get(label, 'gray')  # デフォルトはgray

def calculate_adaptive_font_size(window_x, window_t, fig_width, fig_height, num_x_grids, num_t_grids):
    """
    グリッドサイズに応じた適応的フォントサイズを計算
    
    Parameters:
    -----------
    window_x : float
        水平方向ウィンドウサイズ [m]
    window_t : float  
        時間方向ウィンドウサイズ [ns]
    fig_width : float
        図の幅 [inch]
    fig_height : float
        図の高さ [inch]
    num_x_grids : int
        水平方向グリッド数
    num_t_grids : int
        時間方向グリッド数
    
    Returns:
    --------
    dict: フォントサイズ設定 {'total': int, 'detail': int}
    """
    # 1グリッドあたりの画面上のサイズ（インチ）を計算
    grid_width_inch = fig_width / num_x_grids
    grid_height_inch = fig_height / num_t_grids
    
    # より小さい方の次元を基準にフォントサイズを決定
    min_grid_size_inch = min(grid_width_inch, grid_height_inch)
    
    # フォントサイズ計算（経験的な値）
    # 総数表示用（大きめ）
    total_font_size = max(6, min(20, int(min_grid_size_inch * 20)))
    
    # 詳細表示用（小さめ）
    detail_font_size = max(4, min(12, int(min_grid_size_inch * 12)))
    
    return {
        'total': total_font_size,
        'detail': detail_font_size
    }

def get_text_color_for_background(background_colors, alpha_values):
    """
    背景色に応じて最適な文字色を決定
    
    Parameters:
    -----------
    background_colors : list
        背景色のリスト（matplotlib color format）
    alpha_values : list
        各色のアルファ値のリスト
    
    Returns:
    --------
    str: 最適な文字色 ('white' or 'black')
    """
    import matplotlib.colors as mcolors
    
    # 背景色の明度を計算（アルファ値も考慮）
    total_luminance = 0
    total_weight = 0
    
    for color, alpha in zip(background_colors, alpha_values):
        # 色をRGB値に変換
        rgb = mcolors.to_rgb(color)
        
        # 輝度計算（ITU-R BT.709）
        luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        
        # アルファ値で重み付け
        total_luminance += luminance * alpha
        total_weight += alpha
    
    # 背景がない場合（アルファ値の合計が小さい）は黒文字
    if total_weight < 0.3:
        return 'black'
    
    # 平均輝度を計算
    avg_luminance = total_luminance / total_weight
    
    # 輝度が0.5以上なら黒文字、それ以下なら白文字
    return 'black' if avg_luminance > 0.5 else 'white'

def add_count_text_to_grid(ax, grid_result, font_sizes, show_details=True, bscan_data=None):
    """
    各グリッドセルに岩石数の数値を表示
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        プロット軸
    grid_result : dict
        グリッド解析結果
    font_sizes : dict
        フォントサイズ設定 {'total': int, 'detail': int}
    show_details : bool
        詳細内訳を表示するかどうか
    bscan_data : np.ndarray or None
        B-scanデータ（文字色決定用）
    """
    x_centers = grid_result['x_centers']
    depth_centers = grid_result['depth_centers']
    grid_counts = grid_result['grid_counts']
    window_x = grid_result['window_x']
    window_t = grid_result['window_t']
    
    for i, x_center in enumerate(x_centers):
        for j, depth_center in enumerate(depth_centers):
            counts = grid_counts.get((i, j), {})
            
            # 深さ座標を時間座標に変換
            time_center = depth_to_time(depth_center)
            time_window_t = window_t  # 時間ウィンドウサイズは直接使用
            
            if counts:
                total_count = sum(counts.values())
                
                # 背景色の情報を取得（棒グラフの色）
                sorted_labels = sorted(counts.keys())
                background_colors = [get_label_color(label) for label in sorted_labels]
                alpha_values = [counts[label] / total_count for label in sorted_labels]
                
                # 最適な文字色を決定
                text_color = get_text_color_for_background(background_colors, alpha_values)
                
                # B-scan背景がある場合の文字色調整
                if bscan_data is not None:
                    # B-scan背景がある場合は白文字を優先（見やすさのため）
                    text_color = 'white'
                
                # 総数を中央上部に表示（時間軸ベース）
                ax.text(x_center, time_center - time_window_t * 0.15, 
                       str(total_count),
                       ha='center', va='center',
                       fontsize=font_sizes['total'],
                       fontweight='bold',
                       color=text_color,
                       bbox=dict(boxstyle='round,pad=0.1', 
                                facecolor='black', alpha=0.3, edgecolor='none') if bscan_data is not None else None)
                
                # 詳細内訳を下部に表示（オプション）
                if show_details and len(counts) > 1:
                    detail_text = ' '.join([f'{label}:{count}' for label, count in sorted(counts.items())])
                    ax.text(x_center, time_center + time_window_t * 0.25,
                           detail_text,
                           ha='center', va='center',
                           fontsize=font_sizes['detail'],
                           color=text_color,
                           bbox=dict(boxstyle='round,pad=0.1', 
                                    facecolor='black', alpha=0.2, edgecolor='none') if bscan_data is not None else None)
            else:
                # データがない場合は「-」を表示（B-scan背景がない場合のみ）
                if bscan_data is None:
                    ax.text(x_center, time_center, '-',
                           ha='center', va='center',
                           fontsize=font_sizes['total'],
                           color='gray', alpha=0.5)

def create_grid_analysis(data, window_x, window_t, label_filter=None, filter_name="all"):
    """
    グリッドベースのラベル解析を実行
    
    Parameters:
    -----------
    data : dict
        ラベルデータ
    window_x : float
        水平方向ウィンドウサイズ [m]
    window_t : float
        時間方向ウィンドウサイズ [ns]
    label_filter : list or None
        含めるラベル番号のリスト（None=全て）
    filter_name : str
        フィルタ名（ファイル名用）
    
    Returns:
    --------
    dict: グリッド解析結果
    """
    x_coords = data['x']
    y_coords = data['y']  # これは時間[ns]データ
    labels = data['label']
    
    # ラベルフィルタリング
    if label_filter is not None:
        mask = np.isin(labels, label_filter)
        x_coords = x_coords[mask]
        y_coords = y_coords[mask]
        labels = labels[mask]
    
    if len(labels) == 0:
        print(f"警告: フィルタリング後のデータが空です (フィルタ: {label_filter})")
        return None
    
    # グリッド範囲設定
    x_max = data['x'].max()
    time_min, time_max = data['y'].min(), data['y'].max()
    
    # x軸グリッドビン作成（0基準）
    x_start = 0
    x_end = np.ceil(x_max / window_x) * window_x  # 最大値を含む最小の倍数
    x_bins = np.arange(x_start, x_end + window_x, window_x)
    
    # 時間軸グリッドビン作成（0基準、負の値も含む）
    # 最小値と最大値を0基準のウィンドウサイズの倍数に拡張
    time_start = np.floor(time_min / window_t) * window_t  # 最小値を含む最大の倍数
    time_end = np.ceil(time_max / window_t) * window_t    # 最大値を含む最小の倍数
    time_bins = np.arange(time_start, time_end + window_t, window_t)
    
    # グリッドセンター計算
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    # 時間センターを深さセンターに変換（表示用）
    depth_centers = time_to_depth(time_centers)
    
    # 時間ビンを深さビンに変換（表示用）
    depth_bins = time_to_depth(time_bins)
    
    # 各グリッドセルでのラベル集計
    grid_counts = {}
    for i in range(len(x_centers)):
        for j in range(len(time_centers)):
            # セル範囲内のデータを抽出（時間ベース）
            x_mask = (x_coords >= x_bins[i]) & (x_coords < x_bins[i+1])
            t_mask = (y_coords >= time_bins[j]) & (y_coords < time_bins[j+1])
            cell_mask = x_mask & t_mask
            
            if np.any(cell_mask):
                cell_labels = labels[cell_mask]
                unique_labels, counts = np.unique(cell_labels, return_counts=True)
                grid_counts[(i, j)] = dict(zip(unique_labels, counts))
            else:
                grid_counts[(i, j)] = {}
    
    return {
        'x_centers': x_centers,
        'depth_centers': depth_centers,  # 表示用（深さ変換後）
        'time_centers': time_centers,    # 計算用（時間ベース）
        'x_bins': x_bins,
        'depth_bins': depth_bins,        # 表示用（深さ変換後）
        'time_bins': time_bins,          # 計算用（時間ベース）
        'grid_counts': grid_counts,
        'window_x': window_x,
        'window_t': window_t,            # 時間ベースのウィンドウサイズ
        'filter_name': filter_name,
        'total_points': len(labels)
    }

def calculate_optimal_figure_size(x_bins, depth_bins):
    """
    データ範囲に基づいて最適な図サイズを計算
    
    Parameters:
    -----------
    x_bins : array
        水平方向のビン
    depth_bins : array
        深さ方向のビン
    
    Returns:
    --------
    tuple: (width, height) 図のサイズ
    """
    # データ範囲
    x_range = x_bins[-1] - x_bins[0]
    depth_range = depth_bins[-1] - depth_bins[0]
    
    # アスペクト比
    aspect_ratio = x_range / depth_range
    
    # 基準サイズ（深さ方向）
    base_height = 6  # インチ
    
    # アスペクト比に基づいて幅を調整
    if aspect_ratio > 10:  # 極端に横長の場合
        # 最大幅を制限し、複数行に分割することを考慮
        max_width = 20
        width = min(max_width, base_height * aspect_ratio * 0.3)
        height = base_height
    elif aspect_ratio > 5:  # 横長の場合
        width = base_height * aspect_ratio * 0.4
        height = base_height
    else:  # 通常の場合
        width = base_height * aspect_ratio
        height = base_height
    
    # 最小サイズの確保
    width = max(width, 8)
    height = max(height, 4)
    
    return width, height

def find_time_zero_index(data):
    """
    x=0（最初のトレース）において最初にNaNではない値をとるインデックスを見つける
    
    Parameters:
    -----------
    data : np.ndarray
        B-scanデータ (time x traces)
    
    Returns:
    --------
    time_zero_index : int
        t=0として設定するインデックス番号
    """
    if data.shape[1] == 0:
        return 0
    
    first_trace = data[:, 0]  # x=0のトレース
    
    # NaNではない最初のインデックスを探す
    valid_indices = np.where(~np.isnan(first_trace))[0]
    
    if len(valid_indices) == 0:
        print("警告: x=0のトレースにNaNではない値が見つかりません。t=0をインデックス0に設定します。")
        return 0
    
    time_zero_index = valid_indices[0]
    print(f"t=0として設定: インデックス {time_zero_index}")
    
    return time_zero_index

def prepare_bscan_background(bscan_data):
    """
    B-scanデータを背景表示用に準備する
    
    Parameters:
    -----------
    bscan_data : np.ndarray
        B-scanデータ (time x traces)
    
    Returns:
    --------
    dict: B-scan背景表示用データ
    """
    # CLAUDE.mdの標準パラメータ
    sample_interval = 0.312500e-9  # [s] - Time sampling interval
    trace_interval = 3.6e-2        # [m] - Spatial trace interval
    
    # NaN値を保持したまま処理（NaN領域は白背景で表示するため）
    bscan_display = bscan_data.copy()
    
    # B-scanの座標軸を計算
    x_bscan_end = bscan_data.shape[1] * trace_interval
    
    # t=0補正: x=0で最初にNaNではない値をとるインデックスを見つける
    time_zero_index = find_time_zero_index(bscan_data)
    
    # 時間軸から深さ軸への変換（t=0補正を適用）
    # 時間軸（ns単位、t=0補正適用）
    y_start = -time_zero_index * sample_interval / 1e-9  # 負の時間も含む
    y_end = (bscan_data.shape[0] - time_zero_index) * sample_interval / 1e-9
    
    # 深さ軸（m単位）
    epsilon_r = 4.5
    c = 299792458  # [m/s]
    
    # 時間軸から深さ軸への変換
    time_ns_array = np.arange(-time_zero_index, bscan_data.shape[0] - time_zero_index) * sample_interval / 1e-9
    depth_m_array = time_ns_array * 1e-9 * c / np.sqrt(epsilon_r) * 0.5
    
    # B-scanの範囲
    y_bscan_start = depth_m_array[0]
    y_bscan_end = depth_m_array[-1]
    
    print(f"B-scan時間軸範囲: {y_start:.2f} ns ～ {y_end:.2f} ns")
    print(f"B-scan深さ軸範囲: {y_bscan_start:.3f} m ～ {y_bscan_end:.3f} m")
    
    return {
        'data': bscan_display,
        'x_start': 0,
        'x_end': x_bscan_end,
        'y_start': y_bscan_start,
        'y_end': y_bscan_end,
        'extent': [0, x_bscan_end, y_bscan_end, y_bscan_start]  # [left, right, bottom, top]
    }

def plot_grid_analysis(grid_result, output_dir, suffix="", bscan_data=None, show_counts=True, show_details=True):
    """
    グリッド解析結果をプロット
    
    Parameters:
    -----------
    grid_result : dict
        グリッド解析結果
    output_dir : str
        出力ディレクトリ
    suffix : str
        ファイル名のサフィックス
    bscan_data : np.ndarray or None
        B-scanデータ（背景表示用、None=背景表示なし）
    show_counts : bool
        数値表示を行うかどうか
    show_details : bool
        詳細内訳を表示するかどうか
    """
    if grid_result is None:
        return
    
    x_centers = grid_result['x_centers']
    depth_centers = grid_result['depth_centers']
    x_bins = grid_result['x_bins']
    depth_bins = grid_result['depth_bins']
    grid_counts = grid_result['grid_counts']
    window_x = grid_result['window_x']
    window_t = grid_result['window_t']
    
    # 最適な図サイズを計算
    fig_width, fig_height = calculate_optimal_figure_size(x_bins, depth_bins)
    
    # CLAUDE.mdの標準フォントサイズを定義
    font_large = 20      # Titles and labels
    font_medium = 18     # Axis labels  
    font_small = 16      # Tick labels
    
    # 数値表示用の適応的フォントサイズを計算
    adaptive_font_sizes = None
    if show_counts:
        num_x_grids = len(x_centers)
        num_t_grids = len(depth_centers)  # depth_centersの数がtime_gridsの数と同じ
        adaptive_font_sizes = calculate_adaptive_font_size(
            window_x, window_t, fig_width, fig_height, num_x_grids, num_t_grids)
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 第2縦軸（深さ軸）を作成
    ax2 = ax.twinx()
    
    # B-scan背景表示
    if bscan_data is not None:
        bscan_bg = prepare_bscan_background(bscan_data)
        
        # グレースケールでB-scanを背景表示（NaN領域は白背景）
        # カラーバー範囲の計算（NaN値を無視）
        bscan_data_abs_max = np.nanmax(np.abs(bscan_bg['data']))
        if np.isnan(bscan_data_abs_max) or bscan_data_abs_max == 0:
            bscan_data_abs_max = 1.0  # デフォルト値
        
        # B-scanのextentを時間軸ベースに変換
        bscan_extent_time = [
            bscan_bg['extent'][0],  # x_start (変更なし)
            bscan_bg['extent'][1],  # x_end (変更なし)
            depth_to_time(bscan_bg['extent'][2]),  # y_bottom (深さ→時間)
            depth_to_time(bscan_bg['extent'][3])   # y_top (深さ→時間)
        ]
        
        # NaN値を白背景で表示するためのカラーマップ設定
        # グレースケールカラーマップをベースに、NaN値を白に設定
        cmap = plt.cm.gray.copy()
        cmap.set_bad('white')  # NaN値を白色で表示
        
        ax.imshow(bscan_bg['data'],
                 extent=bscan_extent_time, 
                 aspect='auto', 
                 cmap=cmap,
                 alpha=1.0,  # 透明度を設定
                 origin='upper',
                 interpolation='nearest',
                 vmin=-bscan_data_abs_max/15, 
                 vmax=bscan_data_abs_max/15)
        
        print(f"B-scan背景表示: {bscan_bg['x_start']:.1f} - {bscan_bg['x_end']:.1f} m, {bscan_bg['y_start']:.3f} - {bscan_bg['y_end']:.3f} m")
    
    # 各グリッドセルに円グラフを描画
    for i, x_center in enumerate(x_centers):
        for j, depth_center in enumerate(depth_centers):
            counts = grid_counts.get((i, j), {})
            
            # 深さ座標を時間座標に変換
            time_center = depth_to_time(depth_center)
            time_window_t = window_t  # 時間ウィンドウサイズは直接使用
            
            if counts:
                # 積み上げ棒グラフの設定
                total_count = sum(counts.values())
                
                # データ範囲に基づいた基本サイズ計算
                x_range = x_bins[-1] - x_bins[0]
                depth_range = depth_bins[-1] - depth_bins[0]
                physical_aspect_ratio = x_range / depth_range
                
                # 棒グラフのパディング設定
                padding_x = window_x * 0.1  # 左右のパディング
                padding_t = time_window_t * 0.1  # 時間方向のパディング
                
                # 棒グラフの実際のサイズ
                bar_width = window_x - 2 * padding_x
                bar_height = time_window_t - 2 * padding_t
                
                # 棒グラフの開始位置（時間軸ベース）
                bar_x = x_center - bar_width / 2
                bar_y = time_center - bar_height / 2
                
                # ラベルをソートして一貫した表示順序を保つ
                sorted_labels = sorted(counts.keys())
                
                # 横方向の積み上げ棒グラフを描画
                current_width = 0
                for label in sorted_labels:
                    count = counts[label]
                    # 各セグメントの幅を計算（比例配分）
                    segment_width = (count / total_count) * bar_width
                    
                    # 矩形を描画
                    rect = plt.Rectangle((bar_x + current_width, bar_y),
                                       segment_width, bar_height,
                                       facecolor=get_label_color(label),
                                       edgecolor='black',
                                       linewidth=0.5,
                                       alpha=0.9)  # グリッドの透明度を上げてB-scanと区別
                    ax.add_patch(rect)
                    
                    current_width += segment_width
            else:
                # データがない場合は白い四角を描画（B-scan背景がある場合は透明に）
                if bscan_data is None:
                    rect = plt.Rectangle((x_center - window_x/2, time_center - time_window_t/2),
                                       window_x, time_window_t, 
                                       facecolor='white', edgecolor='lightgray', linewidth=0.5)
                    ax.add_patch(rect)
                else:
                    # B-scan背景がある場合は透明な境界線のみ
                    rect = plt.Rectangle((x_center - window_x/2, time_center - time_window_t/2),
                                       window_x, time_window_t, 
                                       facecolor='none', edgecolor='lightgray', linewidth=0.3, alpha=0.7)
                    ax.add_patch(rect)
    
    # グリッドライン描画（B-scan背景がある場合は少し目立たせる）
    grid_alpha = 0.6 if bscan_data is not None else 0.3
    for x_bin in x_bins:
        ax.axvline(x_bin, color='white' if bscan_data is not None else 'gray', 
                  linestyle='-', alpha=grid_alpha, linewidth=0.5)
    for depth_bin in depth_bins:
        time_bin = depth_to_time(depth_bin)
        ax.axhline(time_bin, color='white' if bscan_data is not None else 'gray', 
                  linestyle='-', alpha=grid_alpha, linewidth=0.5)
    
    # 数値オーバーレイを追加
    if show_counts and adaptive_font_sizes is not None:
        add_count_text_to_grid(ax, grid_result, adaptive_font_sizes, show_details, bscan_data)
    
    # 軸設定（アスペクト比に応じた調整）
    ax.set_xlim(x_bins[0], x_bins[-1])
    
    # 第1縦軸（時間軸）の設定
    ax_time_min = depth_to_time(depth_bins[0])   # 浅い方が小さな時間
    ax_time_max = depth_to_time(depth_bins[-1])  # 深い方が大きな時間
    ax.set_ylim(ax_time_min, ax_time_max)
    
    # 第2縦軸（深さ軸）の設定
    ax2.set_ylim(depth_bins[0], depth_bins[-1])
    
    # 物理的な距離の比率を計算
    x_range = x_bins[-1] - x_bins[0]
    depth_range = depth_bins[-1] - depth_bins[0]
    physical_aspect_ratio = x_range / depth_range
    
    # 軸ラベル設定
    ax.set_xlabel('Horizontal position [m]', fontsize=font_medium)
    ax.set_ylabel('Time [ns]', fontsize=font_medium)
    ax2.set_ylabel(r'Depth [m] in $\varepsilon_r = 4.5$', fontsize=font_medium)
    
    # 軸のticks設定
    # 水平方向のticks（適切な間隔で設定）
    x_ticks = np.arange(x_bins[0], x_bins[-1], 200)
    ax.set_xticks(x_ticks)
    
    # 時間軸（第1縦軸）のticks設定
    time_range = ax_time_max - ax_time_min
    # 時間範囲に応じて適切な間隔を設定
    if time_range <= 200:
        time_interval = 25  # 25ns刻み
    elif time_range <= 500:
        time_interval = 50  # 50ns刻み
    elif time_range <= 1000:
        time_interval = 100  # 100ns刻み
    else:
        time_interval = 200  # 200ns刻み
    
    # 時間ticksを生成（整数値で開始・終了）
    time_start = int(np.ceil(ax_time_min / time_interval) * time_interval)
    time_end = int(np.floor(ax_time_max / time_interval) * time_interval)
    time_ticks = np.arange(time_start, time_end + time_interval, time_interval)
    ax.set_yticks(time_ticks)
    
    # 深さ軸（第2縦軸）のticks設定
    depth_interval = 2.0  # 2m刻み
    depth_start = np.ceil(depth_bins[0] / depth_interval) * depth_interval
    depth_end = np.floor(depth_bins[-1] / depth_interval) * depth_interval
    depth_ticks = np.arange(depth_start, depth_end + depth_interval, depth_interval)
    ax2.set_yticks(depth_ticks)
    
    # ticksのフォーマット設定
    ax.tick_params(axis='both', which='major', labelsize=font_small)
    ax2.tick_params(axis='y', which='major', labelsize=font_small)
    
    # 両方の軸を反転して深い方を下にする
    ax.invert_yaxis()   # 時間軸も反転（大きな時間が下）
    ax2.invert_yaxis()  # 深さ軸も反転（深い方が下）
    
    # 等アスペクト比の設定（物理的な距離の比率に基づいて判定）
    # 物理的な比率が極端に横長でない場合のみ等アスペクト比を適用
    if physical_aspect_ratio <= 5:
        ax.set_aspect('equal', adjustable='box')
    else:
        # 横長データの場合は自動調整
        ax.set_aspect('auto')
    
    # 凡例作成
    if grid_result['filter_name'] == "rocks_only":
        legend_labels = [1, 2, 3]
    elif grid_result['filter_name'] == "non_rocks_only":
        legend_labels = [4, 5, 6]
    else:
        legend_labels = [1, 2, 3, 4, 5, 6]
    
    legend_patches = []
    for label in legend_labels:
        patch = mpatches.Patch(color=get_label_color(label), label=f'Group {label}' if label <= 3 else f'Label {label}')
        legend_patches.append(patch)
    
    # 凡例のフォントサイズ（標準）
    ax.legend(handles=legend_patches, fontsize=font_medium, 
              bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=len(legend_patches))
    
    # タイトル（標準フォントサイズ）
    filter_text = {
        "rocks_only": "Rocks only (Labels 1-3)",
        "non_rocks_only": "Non-rocks only (Labels 4-6)",
        "all": "All labels (Labels 1-6)"
    }
    title = f"Grid Analysis - {filter_text[grid_result['filter_name']]}"
    title += f" (Window: {window_x}m x {window_t}ns)"
    
    # B-scan背景表示の有無をタイトルに追加
    if bscan_data is not None:
        title += " with B-scan background"
    
    ax.set_title(title, fontsize=font_large, pad=20)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    filename = f'grid_analysis_{grid_result["filter_name"]}{suffix}'
    if bscan_data is not None:
        filename += '_with_bscan'
    if show_counts:
        if show_details:
            filename += '_with_counts_and_details'
        else:
            filename += '_with_counts'
    png_path = os.path.join(output_dir, f'{filename}.png')
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"グリッド解析プロット保存: {png_path}")

def save_grid_statistics(grid_results, output_dir, window_x, window_t):
    """
    グリッド統計をテキストファイルに保存
    
    Parameters:
    -----------
    grid_results : dict
        各フィルタのグリッド解析結果
    output_dir : str
        出力ディレクトリ
    window_x : float
        水平方向ウィンドウサイズ [m]
    window_t : float
        時間方向ウィンドウサイズ [ns]
    """
    stats_path = os.path.join(output_dir, 'grid_statistics.txt')
    
    with open(stats_path, 'w') as f:
        f.write("# Grid-based rock label statistics\n")
        f.write(f"# Window size: {window_x}m (horizontal) x {window_t}ns (time)\n")
        f.write(f"# Grid cells: {len(grid_results['all']['x_centers'])} x {len(grid_results['all']['depth_centers'])}\n")
        f.write("\n")
        
        # 各フィルタごとの統計
        for filter_name, result in grid_results.items():
            if result is None:
                continue
                
            f.write(f"## {filter_name.replace('_', ' ').title()}\n")
            f.write(f"Total filtered points: {result['total_points']}\n")
            
            # グリッドセルごとの詳細
            f.write("Grid_X_Center[m]\tGrid_Depth_Center[m]\t")
            if filter_name == "rocks_only":
                f.write("Label_1\tLabel_2\tLabel_3\tTotal\n")
            elif filter_name == "non_rocks_only":
                f.write("Label_4\tLabel_5\tLabel_6\tTotal\n")
            else:
                f.write("Label_1\tLabel_2\tLabel_3\tLabel_4\tLabel_5\tLabel_6\tTotal\n")
            
            for i, x_center in enumerate(result['x_centers']):
                for j, depth_center in enumerate(result['depth_centers']):
                    counts = result['grid_counts'].get((i, j), {})
                    f.write(f"{x_center:.2f}\t{depth_center:.3f}\t")
                    
                    if filter_name == "rocks_only":
                        labels_to_check = [1, 2, 3]
                    elif filter_name == "non_rocks_only":
                        labels_to_check = [4, 5, 6]
                    else:
                        labels_to_check = [1, 2, 3, 4, 5, 6]
                    
                    total = 0
                    for label in labels_to_check:
                        count = counts.get(label, 0)
                        f.write(f"{count}\t")
                        total += count
                    f.write(f"{total}\n")
            
            f.write("\n")
    
    print(f"グリッド統計データ保存: {stats_path}")

def main():
    """
    メイン処理
    """
    print("岩石ラベルグリッド解析ツール")
    print("=" * 50)
    
    # ファイル入力
    json_path = input("JSONファイルのパスを入力してください: ").strip()
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {json_path}")
    
    if not json_path.lower().endswith('.json'):
        raise ValueError("JSONファイルを指定してください")
    
    # ウィンドウサイズ入力
    window_x = float(input("水平方向ウィンドウサイズ [m] を入力してください: ").strip())
    window_t = float(input("時間方向ウィンドウサイズ [ns] を入力してください: ").strip())
    
    # 数値表示オプション
    print("\n数値表示オプション:")
    print("1: 数値表示なし（従来通り）")
    print("2: 総数のみ表示")
    print("3: 総数+詳細内訳表示")
    show_option = input("選択 (1-3): ").strip()
    
    show_counts = show_option in ['2', '3']
    show_details = show_option == '3'
    
    # B-scanデータの取得
    print("\nB-scan背景表示の設定...")
    bscan_path = get_bscan_data_path(json_path)
    bscan_data = None
    
    if bscan_path:
        try:
            bscan_data = load_bscan_data(bscan_path)
            print(f"B-scanデータ形状: {bscan_data.shape}")
        except Exception as e:
            print(f"B-scanデータ読み込みエラー: {e}")
            print("B-scan背景表示なしで続行します")
            bscan_data = None
    
    # 出力ディレクトリ設定
    base_dir = os.path.dirname(os.path.dirname(json_path))
    filename = os.path.splitext(os.path.basename(json_path))[0]
    base_output_dir = os.path.join(base_dir, f'label_statistics_grid/{filename}_x{window_x}_t{window_t}')
    
    # B-scan有り/無しで分けたディレクトリを作成
    output_dir_without_bscan = os.path.join(base_output_dir, 'without_bscan')
    output_dir_with_bscan = os.path.join(base_output_dir, 'with_bscan')
    
    # データ読み込み
    print("\nデータを読み込み中...")
    data = load_label_data(json_path)
    
    print(f"総ラベル数: {len(data['label'])}")
    print(f"水平位置範囲: {data['x'].min():.2f} - {data['x'].max():.2f} m")
    print(f"時間範囲: {data['y'].min():.2f} - {data['y'].max():.2f} ns")
    depth_min = time_to_depth(data['y'].min())
    depth_max = time_to_depth(data['y'].max())
    print(f"深さ範囲: {depth_min:.3f} - {depth_max:.3f} m")
    print(f"グリッド範囲: x=0 - {np.ceil(data['x'].max() / window_x) * window_x:.0f} m, time={np.floor(data['y'].min() / window_t) * window_t:.0f} - {np.ceil(data['y'].max() / window_t) * window_t:.0f} ns")
    
    # 3パターンの解析実行
    print("\nグリッド解析を実行中...")
    
    # 1. 岩石のみ（ラベル1-3）
    print("岩石のみ（ラベル1-3）の解析中...")
    rocks_result = create_grid_analysis(data, window_x, window_t, [1, 2, 3], "rocks_only")
    
    # 2. 非岩石のみ（ラベル4-6）
    print("非岩石のみ（ラベル4-6）の解析中...")
    non_rocks_result = create_grid_analysis(data, window_x, window_t, [4, 5, 6], "non_rocks_only")
    
    # 3. 全ラベル（ラベル1-6）
    print("全ラベル（ラベル1-6）の解析中...")
    all_result = create_grid_analysis(data, window_x, window_t, None, "all")
    
    # プロット作成
    print("\nプロットを作成中...")
    
    # B-scan背景なしのプロット（常に作成）
    print("B-scan背景なしのプロットを作成中...")
    if rocks_result:
        plot_grid_analysis(rocks_result, output_dir_without_bscan, bscan_data=None, 
                          show_counts=show_counts, show_details=show_details)
    if non_rocks_result:
        plot_grid_analysis(non_rocks_result, output_dir_without_bscan, bscan_data=None, 
                          show_counts=show_counts, show_details=show_details)
    if all_result:
        plot_grid_analysis(all_result, output_dir_without_bscan, bscan_data=None, 
                          show_counts=show_counts, show_details=show_details)
    
    # B-scan背景ありのプロット（B-scanデータがある場合のみ）
    if bscan_data is not None:
        print("B-scan背景ありのプロットを作成中...")
        if rocks_result:
            plot_grid_analysis(rocks_result, output_dir_with_bscan, bscan_data=bscan_data, 
                              show_counts=show_counts, show_details=show_details)
        if non_rocks_result:
            plot_grid_analysis(non_rocks_result, output_dir_with_bscan, bscan_data=bscan_data, 
                              show_counts=show_counts, show_details=show_details)
        if all_result:
            plot_grid_analysis(all_result, output_dir_with_bscan, bscan_data=bscan_data, 
                              show_counts=show_counts, show_details=show_details)
    
    # 統計データ保存（両方のディレクトリに保存）
    print("\n統計データを保存中...")
    grid_results = {
        'rocks_only': rocks_result,
        'non_rocks_only': non_rocks_result,
        'all': all_result
    }
    save_grid_statistics(grid_results, output_dir_without_bscan, window_x, window_t)
    
    if bscan_data is not None:
        save_grid_statistics(grid_results, output_dir_with_bscan, window_x, window_t)
    
    print(f"\n出力ディレクトリ:")
    print(f"  B-scan背景なし: {output_dir_without_bscan}")
    if bscan_data is not None:
        print(f"  B-scan背景あり: {output_dir_with_bscan}")
    print("処理完了!")

if __name__ == "__main__":
    main()