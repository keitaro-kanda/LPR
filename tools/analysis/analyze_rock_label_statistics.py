#!/usr/bin/env python3
"""
岩石ラベルの統計的性質プロットツール
JSONファイルから岩石の分布を分析し、深さおよび水平位置のヒストグラムを生成

新機能:
- 深さ計測データ(plot_viewer_depth_measurement.py)による規格化ヒストグラム
- 1mあたりの平均岩石個数を算出する深さ規格化機能
- x座標の自動マッチング機能（許容誤差付き）
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

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
        1: 'r',        # red
        2: 'g',        # green
        3: 'b',        # blue
        4: 'cyan',     # cyan
        5: 'magenta',  # magenta
        6: 'yellow'    # yellow
    }
    return color_map.get(label, 'gray')  # デフォルトはgray

def create_depth_histogram(data, bin_size_m=1.0, output_dir='rock_statics', label_filter=None, suffix=''):
    """
    深さごとの岩石ラベルヒストグラムを作成
    
    Parameters:
    -----------
    data : dict
        ラベルデータ
    bin_size_m : float
        深さビンサイズ [m]
    output_dir : str
        出力ディレクトリ
    label_filter : list or None
        含めるラベル番号のリスト（None=全て）
    suffix : str
        ファイル名のサフィックス
    """
    y_coords = data['y']
    labels = data['label']
    
    # ラベルフィルタリング
    if label_filter is not None:
        mask = np.isin(labels, label_filter)
        y_coords = y_coords[mask]
        labels = labels[mask]
    
    if len(labels) == 0:
        print(f"警告: フィルタリング後のデータが空です (フィルタ: {label_filter})")
        return
    
    # 深さ変換
    depths = time_to_depth(y_coords)
    
    # 深さビン設定（0.5の倍数で区切り）
    all_depths = time_to_depth(data['y'])
    depth_min, depth_max = all_depths.min(), all_depths.max()
    
    # 0.5の倍数でビン境界を設定
    bin_start = np.floor(depth_min / bin_size_m) * bin_size_m
    bin_end = np.ceil(depth_max / bin_size_m) * bin_size_m
    bins = np.arange(bin_start, bin_end + bin_size_m, bin_size_m)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # ラベル別にヒストグラム計算
    unique_labels = np.unique(labels)
    label_counts = {}
    
    for label in unique_labels:
        mask = labels == label
        counts, _ = np.histogram(depths[mask], bins=bins)
        label_counts[label] = counts
    
    # プロット作成
    plt.figure(figsize=(8, 10))
    
    # 積み上げヒストグラム
    bottom = np.zeros(len(bin_centers))
    
    for label in sorted(unique_labels):
        plt.barh(bin_centers, label_counts[label], 
                height=bin_size_m * 0.8, 
                left=bottom, 
                label=f'Group {label}' if label <= 3 else f'Label {label}',
                color=get_label_color(label), alpha=0.7)
        bottom += label_counts[label]
    
    plt.xlabel('Number of echoes', fontsize=20)
    plt.ylabel(r'Depth [m] in $\varepsilon_r = 4.5$', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    title_text = 'Rock distribution by depth'
    if label_filter is not None:
        title_text += f' (Labels {label_filter})'
    # Use 2 columns for legend if 6 labels are present
    ncol = 2 if len(unique_labels) == 6 else 1
    plt.legend(fontsize=18, ncol=ncol)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # 深い方を下に
    
    # Set x-axis limit to max number + 1 for depth histogram
    # Calculate the maximum stacked count across all bins
    max_count = np.max(bottom) if len(bottom) > 0 else 1
    plt.xlim(0, max_count + 1)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    filename = f'depth_histogram{suffix}'
    png_path = os.path.join(output_dir, f'{filename}.png')
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"深さヒストグラム保存: {png_path}")
    
    # 統計データ保存
    stats_path = os.path.join(output_dir, f'depth_statistics{suffix}.txt')
    with open(stats_path, 'w') as f:
        f.write("# Depth statistics\n")
        f.write(f"# Bin size: {bin_size_m} m\n")
        if label_filter is not None:
            f.write(f"# Label filter: {label_filter}\n")
        f.write("# Depth_center[m]\t")
        for label in sorted(unique_labels):
            f.write(f"Label_{label}\t")
        f.write("Total\n")
        
        for i, depth in enumerate(bin_centers):
            f.write(f"{depth:.3f}\t")
            total = 0
            for label in sorted(unique_labels):
                count = label_counts[label][i]
                f.write(f"{count}\t")
                total += count
            f.write(f"{total}\n")
    
    print(f"深さ統計データ保存: {stats_path}")

def create_time_histogram(data, bin_size_ns=10.0, output_dir='rock_statics', label_filter=None, suffix=''):
    """
    深さごとの岩石ラベルヒストグラムを作成（時間[ns]単位）
    
    Parameters:
    -----------
    data : dict
        ラベルデータ
    bin_size_ns : float
        時間ビンサイズ [ns]
    output_dir : str
        出力ディレクトリ
    label_filter : list or None
        含めるラベル番号のリスト（None=全て）
    suffix : str
        ファイル名のサフィックス
    """
    y_coords = data['y']
    labels = data['label']
    
    # ラベルフィルタリング
    if label_filter is not None:
        mask = np.isin(labels, label_filter)
        y_coords = y_coords[mask]
        labels = labels[mask]
    
    if len(labels) == 0:
        print(f"警告: フィルタリング後のデータが空です (フィルタ: {label_filter})")
        return
    
    # 時間ビン設定
    all_times = data['y']
    time_min, time_max = all_times.min(), all_times.max()
    
    # ビン境界を設定
    bin_start = np.floor(time_min / bin_size_ns) * bin_size_ns
    bin_end = np.ceil(time_max / bin_size_ns) * bin_size_ns
    bins = np.arange(bin_start, bin_end + bin_size_ns, bin_size_ns)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # ラベル別にヒストグラム計算
    unique_labels = np.unique(labels)
    label_counts = {}
    
    for label in unique_labels:
        mask = labels == label
        counts, _ = np.histogram(y_coords[mask], bins=bins)
        label_counts[label] = counts
    
    # プロット作成
    fig, ax1 = plt.subplots(figsize=(8, 10))
    
    # 積み上げヒストグラム
    bottom = np.zeros(len(bin_centers))
    
    for label in sorted(unique_labels):
        ax1.barh(bin_centers, label_counts[label], 
                height=bin_size_ns * 0.8, 
                left=bottom, 
                label=f'Group {label}' if label <= 3 else f'Label {label}',
                color=get_label_color(label), alpha=0.7)
        bottom += label_counts[label]
    
    ax1.set_xlabel('Number of echoes', fontsize=20)
    ax1.set_ylabel('Time [ns]', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    title_text = 'Rock distribution by time'
    if label_filter is not None:
        title_text += f' (Labels {label_filter})'
    # Use 2 columns for legend if 6 labels are present
    ncol = 2 if len(unique_labels) == 6 else 1
    ax1.legend(fontsize=18, ncol=ncol)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # 深い方を下に
    ax1.set_yticks(np.arange(0, len(bin_centers) * bin_size_ns, 20))

    # Set x-axis limit to max number + 1 for time histogram
    # Calculate the maximum stacked count across all bins
    max_count = np.max(bottom) if len(bottom) > 0 else 1
    ax1.set_xlim(0, max_count + 1)
    
    # 第二縦軸（深さ[m]）を追加
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Depth [m] in $\varepsilon_r = 4.5$', fontsize=20)
    
    # 時間軸の範囲を取得して深さに変換
    time_min, time_max = ax1.get_ylim()
    depth_min = time_to_depth(time_max)  # y軸が反転しているため
    depth_max = time_to_depth(time_min)
    ax2.set_ylim(depth_min, depth_max)
    ax2.invert_yaxis()  # 深い方を下に
    ax2.tick_params(axis='y', which='major', labelsize=16)
    
    fig.tight_layout()
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    filename = f'time_histogram{suffix}'
    png_path = os.path.join(output_dir, f'{filename}.png')
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"時間ヒストグラム保存: {png_path}")
    
    # 統計データ保存
    stats_path = os.path.join(output_dir, f'time_statistics{suffix}.txt')
    with open(stats_path, 'w') as f:
        f.write("# Time statistics\n")
        f.write(f"# Bin size: {bin_size_ns} ns\n")
        if label_filter is not None:
            f.write(f"# Label filter: {label_filter}\n")
        f.write("# Time_center[ns]\t")
        for label in sorted(unique_labels):
            f.write(f"Label_{label}\t")
        f.write("Total\n")
        
        for i, time in enumerate(bin_centers):
            f.write(f"{time:.1f}\t")
            total = 0
            for label in sorted(unique_labels):
                count = label_counts[label][i]
                f.write(f"{count}\t")
                total += count
            f.write(f"{total}\n")
    
    print(f"時間統計データ保存: {stats_path}")

def create_horizontal_histogram(data, bin_size_m=50.0, output_dir='rock_statics', label_filter=None, suffix=''):
    """
    水平位置ごとの岩石ラベルヒストグラムを作成
    
    Parameters:
    -----------
    data : dict
        ラベルデータ
    bin_size_m : float
        水平位置ビンサイズ [m]
    output_dir : str
        出力ディレクトリ
    label_filter : list or None
        含めるラベル番号のリスト（None=全て）
    suffix : str
        ファイル名のサフィックス
    """
    x_coords = data['x']
    labels = data['label']
    
    # ラベルフィルタリング
    if label_filter is not None:
        mask = np.isin(labels, label_filter)
        x_coords_filtered = x_coords[mask]
        labels_filtered = labels[mask]
    else:
        x_coords_filtered = x_coords
        labels_filtered = labels
    
    if len(labels_filtered) == 0:
        print(f"警告: フィルタリング後のデータが空です (フィルタ: {label_filter})")
        return
    
    # ビン設定（0スタート）
    x_max = data['x'].max()
    bin_start = 0
    bin_end = np.ceil(x_max / bin_size_m) * bin_size_m
    bins = np.arange(bin_start, bin_end + bin_size_m, bin_size_m)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # ラベル別にヒストグラム計算
    unique_labels = np.unique(labels_filtered)
    label_counts = {}
    
    for label in unique_labels:
        mask = labels_filtered == label
        counts, _ = np.histogram(x_coords_filtered[mask], bins=bins)
        label_counts[label] = counts
    
    # プロット作成
    plt.figure(figsize=(10, 8))
    
    # 積み上げヒストグラム
    bottom = np.zeros(len(bin_centers))
    
    for label in sorted(unique_labels):
        plt.bar(bin_centers, label_counts[label], 
               width=bin_size_m * 0.8, 
               bottom=bottom,
               label=f'Group {label}' if label <= 3 else f'Label {label}',
               color=get_label_color(label), alpha=0.7)
        bottom += label_counts[label]
    
    plt.xlabel('Horizontal position [m]', fontsize=20)
    plt.ylabel('Number of echoes', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    # Use 2 columns for legend if 6 labels are present
    ncol = 2 if len(unique_labels) == 6 else 1
    plt.legend(fontsize=18, ncol=ncol)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limit to max number + 1 for horizontal histogram
    # Calculate the maximum stacked count across all bins
    max_count = np.max(bottom) if len(bottom) > 0 else 1
    plt.ylim(0, max_count + 1)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    filename = f'horizontal_histogram{suffix}'
    png_path = os.path.join(output_dir, f'{filename}.png')
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"水平位置ヒストグラム保存: {png_path}")
    
    # 統計データ保存
    stats_path = os.path.join(output_dir, f'horizontal_statistics{suffix}.txt')
    with open(stats_path, 'w') as f:
        f.write("# Horizontal position statistics\n")
        f.write(f"# Bin size: {bin_size_m} m\n")
        if label_filter is not None:
            f.write(f"# Label filter: {label_filter}\n")
        f.write("# Position_center[m]\t")
        for label in sorted(unique_labels):
            f.write(f"Label_{label}\t")
        f.write("Total\n")
        
        for i, pos in enumerate(bin_centers):
            f.write(f"{pos:.2f}\t")
            total = 0
            for label in sorted(unique_labels):
                count = label_counts[label][i]
                f.write(f"{count}\t")
                total += count
            f.write(f"{total}\n")
    
    print(f"水平位置統計データ保存: {stats_path}")

def create_depth_normalized_horizontal_histogram(data, depth_data, bin_size_m=50.0, output_dir='rock_statics', label_filter=None, suffix=''):
    """
    深さで規格化した水平位置ごとの岩石ラベルヒストグラムを作成
    
    Parameters:
    -----------
    data : dict
        ラベルデータ
    depth_data : dict
        深さ計測データ {'x': array, 'depth': array}
    bin_size_m : float
        水平位置ビンサイズ [m]
    output_dir : str
        出力ディレクトリ
    label_filter : list or None
        含めるラベル番号のリスト（None=全て）
    suffix : str
        ファイル名のサフィックス
    """
    if depth_data is None:
        print("警告: 深さ計測データが提供されていません。深さ規格化ヒストグラムをスキップします。")
        return
    
    x_coords = data['x']
    labels = data['label']
    
    # ラベルフィルタリング
    if label_filter is not None:
        mask = np.isin(labels, label_filter)
        x_coords_filtered = x_coords[mask]
        labels_filtered = labels[mask]
    else:
        x_coords_filtered = x_coords
        labels_filtered = labels
    
    if len(labels_filtered) == 0:
        print(f"警告: フィルタリング後のデータが空です (フィルタ: {label_filter})")
        return
    
    # ビン設定（0スタート）
    x_max = data['x'].max()
    bin_start = 0
    bin_end = np.ceil(x_max / bin_size_m) * bin_size_m
    bins = np.arange(bin_start, bin_end + bin_size_m, bin_size_m)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 各ビンに対応する深さを取得（ビン下限基準）
    bin_depths = []
    for bin_lower in bins[:-1]:
        depth = find_nearest_depth(bin_lower, depth_data, tolerance=bin_size_m/2)
        bin_depths.append(depth)
    
    # 深さが見つからないビンの数をカウント
    missing_depth_count = sum(1 for d in bin_depths if d is None)
    if missing_depth_count > 0:
        print(f"警告: {missing_depth_count}個のビンで深さデータが見つかりません")
    
    # ラベル別にヒストグラム計算
    unique_labels = np.unique(labels_filtered)
    label_counts = {}
    label_densities = {}
    
    for label in unique_labels:
        mask = labels_filtered == label
        counts, _ = np.histogram(x_coords_filtered[mask], bins=bins)
        label_counts[label] = counts
        
        # 深さで規格化した密度を計算
        densities = np.zeros(len(counts))
        for i, (count, depth) in enumerate(zip(counts, bin_depths)):
            if depth is not None and depth > 0:
                densities[i] = count / depth  # 個/m
            else:
                densities[i] = 0  # 深さデータがない場合は0
        
        label_densities[label] = densities
    
    # プロット作成
    plt.figure(figsize=(10, 8))
    
    # 積み上げヒストグラム
    bottom = np.zeros(len(bin_centers))
    
    for label in sorted(unique_labels):
        plt.bar(bin_centers, label_densities[label], 
               width=bin_size_m * 0.8, 
               bottom=bottom,
               label=f'Group {label}' if label <= 3 else f'Label {label}',
               color=get_label_color(label), alpha=0.7)
        bottom += label_densities[label]
    
    plt.xlabel('Horizontal position [m]', fontsize=20)
    plt.ylabel('Rock density [count/m depth]', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    # Use 2 columns for legend if 6 labels are present
    ncol = 2 if len(unique_labels) == 6 else 1
    plt.legend(fontsize=18, ncol=ncol)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limit to max density + 10%
    max_density = np.max(bottom) if len(bottom) > 0 else 1
    plt.ylim(0, max_density * 1.1)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    filename = f'normalized_horizontal_histogram{suffix}'
    png_path = os.path.join(output_dir, f'{filename}.png')
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"深さ規格化水平位置ヒストグラム保存: {png_path}")
    
    # 統計データ保存
    stats_path = os.path.join(output_dir, f'normalized_horizontal_statistics{suffix}.txt')
    with open(stats_path, 'w') as f:
        f.write("# Depth-normalized horizontal position statistics\n")
        f.write(f"# Bin size: {bin_size_m} m\n")
        if label_filter is not None:
            f.write(f"# Label filter: {label_filter}\n")
        f.write("# Position_center[m]\tBin_lower[m]\tBin_depth[m]\t")
        for label in sorted(unique_labels):
            f.write(f"Label_{label}_count\tLabel_{label}_density[count/m]\t")
        f.write("Total_count\tTotal_density[count/m]\n")
        
        for i, (pos, bin_lower, depth) in enumerate(zip(bin_centers, bins[:-1], bin_depths)):
            depth_str = f"{depth:.3f}" if depth is not None else "N/A"
            f.write(f"{pos:.2f}\t{bin_lower:.2f}\t{depth_str}\t")
            total_count = 0
            total_density = 0
            for label in sorted(unique_labels):
                count = label_counts[label][i]
                density = label_densities[label][i]
                f.write(f"{count}\t{density:.3f}\t")
                total_count += count
                total_density += density
            f.write(f"{total_count}\t{total_density:.3f}\n")
    
    print(f"深さ規格化統計データ保存: {stats_path}")

def load_depth_measurements(json_path):
    """
    深さ計測結果のJSONファイルを読み込む
    
    Parameters:
    -----------
    json_path : str
        深さ計測結果のJSONファイルパス
    
    Returns:
    --------
    dict: {'x': array, 'depth': array} or None
    """
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        measurements = data.get('measurements', [])
        if not measurements:
            return None
        
        x_positions = []
        depths = []
        
        for measurement in measurements:
            x_positions.append(measurement['x_position'])
            depths.append(measurement['depth_m'])
        
        return {
            'x': np.array(x_positions),
            'depth': np.array(depths)
        }
    except Exception as e:
        print(f"深さ計測データ読み込みエラー: {e}")
        return None

def find_nearest_depth(x_target, depth_data, tolerance=1.0):
    """
    指定されたx座標に最も近い深さデータを探す
    
    Parameters:
    -----------
    x_target : float
        目標とするx座標
    depth_data : dict
        深さ計測データ {'x': array, 'depth': array}
    tolerance : float
        許容誤差 [m]
    
    Returns:
    --------
    float or None: 対応する深さ [m]
    """
    if depth_data is None:
        return None
    
    x_positions = depth_data['x']
    depths = depth_data['depth']
    
    # 最も近いx座標を探す
    distances = np.abs(x_positions - x_target)
    min_idx = np.argmin(distances)
    
    if distances[min_idx] <= tolerance:
        return depths[min_idx]
    else:
        return None

def depth_to_time(depth_m, epsilon_r=4.5):
    """
    深さ[m]を時間[ns]に変換
    
    Parameters:
    -----------
    depth_m : float
        深さ [m]
    epsilon_r : float
        比誘電率（デフォルト: 4.5）
    
    Returns:
    --------
    float: 時間 [ns]
    """
    c = 299792458  # 光速 [m/s]
    time_s = depth_m * 2 * np.sqrt(epsilon_r) / c
    return time_s * 1e9  # [ns]

def create_thin_layer_normalized_horizontal_histogram(data, depth_data, bin_size_m=50.0, output_dir='rock_statics', label_filter=None, suffix=''):
    """
    最も薄い層の厚みまでのデータを用いた深さ規格化水平位置ヒストグラムを作成
    
    Parameters:
    -----------
    data : dict
        ラベルデータ
    depth_data : dict
        深さ計測データ {'x': array, 'depth': array}
    bin_size_m : float
        水平位置ビンサイズ [m]
    output_dir : str
        出力ディレクトリ
    label_filter : list or None
        含めるラベル番号のリスト（None=全て）
    suffix : str
        ファイル名のサフィックス
    """
    if depth_data is None:
        print("警告: 深さ計測データが提供されていません。薄い層規格化ヒストグラムをスキップします。")
        return
    
    # 最も薄い層の厚みを取得
    min_depth = np.min(depth_data['depth'])
    max_time_ns = depth_to_time(min_depth)
    
    print(f"最も薄い層の厚み: {min_depth:.3f} m (時間: {max_time_ns:.1f} ns)")
    
    x_coords = data['x']
    y_coords = data['y']
    labels = data['label']
    
    # 薄い層までのデータでフィルタリング
    depth_mask = y_coords <= max_time_ns
    x_coords_depth_filtered = x_coords[depth_mask]
    labels_depth_filtered = labels[depth_mask]
    
    print(f"薄い層フィルタリング後のデータ: {len(labels_depth_filtered)}個 (全体の{len(labels_depth_filtered)/len(labels)*100:.1f}%)")
    
    # ラベルフィルタリング
    if label_filter is not None:
        mask = np.isin(labels_depth_filtered, label_filter)
        x_coords_filtered = x_coords_depth_filtered[mask]
        labels_filtered = labels_depth_filtered[mask]
    else:
        x_coords_filtered = x_coords_depth_filtered
        labels_filtered = labels_depth_filtered
    
    if len(labels_filtered) == 0:
        print(f"警告: フィルタリング後のデータが空です (フィルタ: {label_filter})")
        return
    
    # ビン設定（0スタート）
    x_max = data['x'].max()
    bin_start = 0
    bin_end = np.ceil(x_max / bin_size_m) * bin_size_m
    bins = np.arange(bin_start, bin_end + bin_size_m, bin_size_m)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 各ビンに対応する深さを取得（ビン下限基準）
    bin_depths = []
    for bin_lower in bins[:-1]:
        depth = find_nearest_depth(bin_lower, depth_data, tolerance=bin_size_m/2)
        bin_depths.append(depth)
    
    # 深さが見つからないビンの数をカウント
    missing_depth_count = sum(1 for d in bin_depths if d is None)
    if missing_depth_count > 0:
        print(f"警告: {missing_depth_count}個のビンで深さデータが見つかりません")
    
    # ラベル別にヒストグラム計算
    unique_labels = np.unique(labels_filtered)
    label_counts = {}
    label_densities = {}
    
    for label in unique_labels:
        mask = labels_filtered == label
        counts, _ = np.histogram(x_coords_filtered[mask], bins=bins)
        label_counts[label] = counts
        
        # 薄い層の深さで規格化した密度を計算
        densities = np.zeros(len(counts))
        for i, (count, depth) in enumerate(zip(counts, bin_depths)):
            if depth is not None and depth > 0:
                # 薄い層の厚みを使用
                effective_depth = min(depth, min_depth)
                densities[i] = count / effective_depth  # 個/m
            else:
                densities[i] = 0  # 深さデータがない場合は0
        
        label_densities[label] = densities
    
    # プロット作成
    plt.figure(figsize=(10, 8))
    
    # 積み上げヒストグラム
    bottom = np.zeros(len(bin_centers))
    
    for label in sorted(unique_labels):
        plt.bar(bin_centers, label_densities[label], 
               width=bin_size_m * 0.8, 
               bottom=bottom,
               label=f'Group {label}' if label <= 3 else f'Label {label}',
               color=get_label_color(label), alpha=0.7)
        bottom += label_densities[label]
    
    plt.xlabel('Horizontal position [m]', fontsize=20)
    plt.ylabel(f'Rock density [count/m depth] (depth ≤ {min_depth:.3f}m)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    # Use 2 columns for legend if 6 labels are present
    ncol = 2 if len(unique_labels) == 6 else 1
    plt.legend(fontsize=18, ncol=ncol)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limit to max density + 10%
    max_density = np.max(bottom) if len(bottom) > 0 else 1
    plt.ylim(0, max_density * 1.1)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    filename = f'normalized_horizontal_histogram_thin{suffix}'
    png_path = os.path.join(output_dir, f'{filename}.png')
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"薄い層規格化水平位置ヒストグラム保存: {png_path}")
    
    # 統計データ保存
    stats_path = os.path.join(output_dir, f'normalized_horizontal_statistics_thin{suffix}.txt')
    with open(stats_path, 'w') as f:
        f.write(f"# Thin layer normalized horizontal position statistics (depth ≤ {min_depth:.3f}m)\n")
        f.write(f"# Bin size: {bin_size_m} m\n")
        f.write(f"# Max depth used: {min_depth:.3f} m (time: {max_time_ns:.1f} ns)\n")
        f.write(f"# Data after depth filtering: {len(labels_depth_filtered)} / {len(labels)} ({len(labels_depth_filtered)/len(labels)*100:.1f}%)\n")
        if label_filter is not None:
            f.write(f"# Label filter: {label_filter}\n")
        f.write("# Position_center[m]\tBin_lower[m]\tBin_depth[m]\tEffective_depth[m]\t")
        for label in sorted(unique_labels):
            f.write(f"Label_{label}_count\tLabel_{label}_density[count/m]\t")
        f.write("Total_count\tTotal_density[count/m]\n")
        
        for i, (pos, bin_lower, depth) in enumerate(zip(bin_centers, bins[:-1], bin_depths)):
            depth_str = f"{depth:.3f}" if depth is not None else "N/A"
            effective_depth = min(depth, min_depth) if depth is not None else min_depth
            effective_depth_str = f"{effective_depth:.3f}" if depth is not None else "N/A"
            f.write(f"{pos:.2f}\t{bin_lower:.2f}\t{depth_str}\t{effective_depth_str}\t")
            total_count = 0
            total_density = 0
            for label in sorted(unique_labels):
                count = label_counts[label][i]
                density = label_densities[label][i]
                f.write(f"{count}\t{density:.3f}\t")
                total_count += count
                total_density += density
            f.write(f"{total_count}\t{total_density:.3f}\n")
    
    print(f"薄い層規格化統計データ保存: {stats_path}")

def create_depth_ratio_histogram(data, bin_size_m=1.0, output_dir='rock_statics', label_filter=None, suffix=''):
    """
    深さごとの岩石ラベルの割合ヒストグラムを作成
    
    Parameters:
    -----------
    data : dict
        ラベルデータ
    bin_size_m : float
        深さビンサイズ [m]
    output_dir : str
        出力ディレクトリ
    label_filter : list or None
        含めるラベル番号のリスト（None=全て）
    suffix : str
        ファイル名のサフィックス
    """
    y_coords = data['y']
    labels = data['label']
    
    # ラベルフィルタリング
    if label_filter is not None:
        mask = np.isin(labels, label_filter)
        y_coords = y_coords[mask]
        labels = labels[mask]
    
    if len(labels) == 0:
        print(f"警告: フィルタリング後のデータが空です (フィルタ: {label_filter})")
        return
    
    # 深さ変換
    depths = time_to_depth(y_coords)
    
    # 深さビン設定（0.5の倍数で区切り）
    all_depths = time_to_depth(data['y'])
    depth_min, depth_max = all_depths.min(), all_depths.max()
    
    # 0.5の倍数でビン境界を設定
    bin_start = np.floor(depth_min / bin_size_m) * bin_size_m
    bin_end = np.ceil(depth_max / bin_size_m) * bin_size_m
    bins = np.arange(bin_start, bin_end + bin_size_m, bin_size_m)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # ラベル別にヒストグラム計算
    unique_labels = np.unique(labels)
    label_counts = {}
    label_ratios = {}
    
    for label in unique_labels:
        mask = labels == label
        counts, _ = np.histogram(depths[mask], bins=bins)
        label_counts[label] = counts
    
    # 各ビンでの割合を計算
    for label in unique_labels:
        ratios = np.zeros(len(bin_centers))
        for i in range(len(bin_centers)):
            total_count = sum(label_counts[l][i] for l in unique_labels)
            if total_count > 0:
                ratios[i] = (label_counts[label][i] / total_count) * 100  # パーセント
            else:
                ratios[i] = 0
        label_ratios[label] = ratios
    
    # プロット作成
    plt.figure(figsize=(8, 10))
    
    # 積み上げヒストグラム（割合）
    bottom = np.zeros(len(bin_centers))
    
    for label in sorted(unique_labels):
        plt.barh(bin_centers, label_ratios[label], 
                height=bin_size_m * 0.8, 
                left=bottom, 
                label=f'Group {label}' if label <= 3 else f'Label {label}',
                color=get_label_color(label), alpha=0.7)
        bottom += label_ratios[label]
    
    plt.xlabel('Percentage [%]', fontsize=20)
    plt.ylabel(r'Depth [m] in $\varepsilon_r = 4.5$', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    title_text = 'Rock ratio distribution by depth'
    if label_filter is not None:
        title_text += f' (Labels {label_filter})'
    # Use 2 columns for legend if 6 labels are present
    ncol = 2 if len(unique_labels) == 6 else 1
    plt.legend(fontsize=18, ncol=ncol)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # 深い方を下に
    
    # Set x-axis limit to 0-100%
    plt.xlim(0, 100)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    filename = f'depth_ratio_histogram{suffix}'
    png_path = os.path.join(output_dir, f'{filename}.png')
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"深さ割合ヒストグラム保存: {png_path}")
    
    # 統計データ保存
    stats_path = os.path.join(output_dir, f'depth_ratio_statistics{suffix}.txt')
    with open(stats_path, 'w') as f:
        f.write("# Depth ratio statistics\n")
        f.write(f"# Bin size: {bin_size_m} m\n")
        if label_filter is not None:
            f.write(f"# Label filter: {label_filter}\n")
        f.write("# Depth_center[m]\t")
        for label in sorted(unique_labels):
            f.write(f"Label_{label}_count\tLabel_{label}_ratio[%]\t")
        f.write("Total_count\n")
        
        for i, depth in enumerate(bin_centers):
            f.write(f"{depth:.3f}\t")
            total = 0
            for label in sorted(unique_labels):
                count = label_counts[label][i]
                ratio = label_ratios[label][i]
                f.write(f"{count}\t{ratio:.1f}\t")
                total += count
            f.write(f"{total}\n")
    
    print(f"深さ割合統計データ保存: {stats_path}")

def create_depth_ratio_histogram_time(data, bin_size_ns=10.0, output_dir='rock_statics', label_filter=None, suffix=''):
    """
    時間ごとの岩石ラベルの割合ヒストグラムを作成（時間[ns]単位）
    
    Parameters:
    -----------
    data : dict
        ラベルデータ
    bin_size_ns : float
        時間ビンサイズ [ns]
    output_dir : str
        出力ディレクトリ
    label_filter : list or None
        含めるラベル番号のリスト（None=全て）
    suffix : str
        ファイル名のサフィックス
    """
    y_coords = data['y']
    labels = data['label']
    
    # ラベルフィルタリング
    if label_filter is not None:
        mask = np.isin(labels, label_filter)
        y_coords = y_coords[mask]
        labels = labels[mask]
    
    if len(labels) == 0:
        print(f"警告: フィルタリング後のデータが空です (フィルタ: {label_filter})")
        return
    
    # 時間ビン設定
    all_times = data['y']
    time_min, time_max = all_times.min(), all_times.max()
    
    # ビン境界を設定
    bin_start = np.floor(time_min / bin_size_ns) * bin_size_ns
    bin_end = np.ceil(time_max / bin_size_ns) * bin_size_ns
    bins = np.arange(bin_start, bin_end + bin_size_ns, bin_size_ns)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # ラベル別にヒストグラム計算
    unique_labels = np.unique(labels)
    label_counts = {}
    label_ratios = {}
    
    for label in unique_labels:
        mask = labels == label
        counts, _ = np.histogram(y_coords[mask], bins=bins)
        label_counts[label] = counts
    
    # 各ビンでの割合を計算
    for label in unique_labels:
        ratios = np.zeros(len(bin_centers))
        for i in range(len(bin_centers)):
            total_count = sum(label_counts[l][i] for l in unique_labels)
            if total_count > 0:
                ratios[i] = (label_counts[label][i] / total_count) * 100  # パーセント
            else:
                ratios[i] = 0
        label_ratios[label] = ratios
    
    # プロット作成
    fig, ax1 = plt.subplots(figsize=(8, 10))
    
    # 積み上げヒストグラム（割合）
    bottom = np.zeros(len(bin_centers))
    
    for label in sorted(unique_labels):
        ax1.barh(bin_centers, label_ratios[label], 
                height=bin_size_ns * 0.8, 
                left=bottom, 
                label=f'Group {label}' if label <= 3 else f'Label {label}',
                color=get_label_color(label), alpha=0.7)
        bottom += label_ratios[label]
    
    ax1.set_xlabel('Percentage [%]', fontsize=20)
    ax1.set_ylabel('Time [ns]', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    title_text = 'Rock ratio distribution by time'
    if label_filter is not None:
        title_text += f' (Labels {label_filter})'
    # Use 2 columns for legend if 6 labels are present
    ncol = 2 if len(unique_labels) == 6 else 1
    ax1.legend(fontsize=18, ncol=ncol)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # 深い方を下に
    
    # Set x-axis limit to 0-100%
    ax1.set_xlim(0, 100)
    
    # 第二縦軸（深さ[m]）を追加
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Depth [m] in $\varepsilon_r = 4.5$', fontsize=20)
    
    # 時間軸の範囲を取得して深さに変換
    time_min, time_max = ax1.get_ylim()
    depth_min = time_to_depth(time_max)  # y軸が反転しているため
    depth_max = time_to_depth(time_min)
    ax2.set_ylim(depth_min, depth_max)
    ax2.invert_yaxis()  # 深い方を下に
    ax2.tick_params(axis='y', which='major', labelsize=16)
    
    fig.tight_layout()
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    filename = f'time_ratio_histogram{suffix}'
    png_path = os.path.join(output_dir, f'{filename}.png')
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"時間割合ヒストグラム保存: {png_path}")
    
    # 統計データ保存
    stats_path = os.path.join(output_dir, f'time_ratio_statistics{suffix}.txt')
    with open(stats_path, 'w') as f:
        f.write("# Time ratio statistics\n")
        f.write(f"# Bin size: {bin_size_ns} ns\n")
        if label_filter is not None:
            f.write(f"# Label filter: {label_filter}\n")
        f.write("# Time_center[ns]\t")
        for label in sorted(unique_labels):
            f.write(f"Label_{label}_count\tLabel_{label}_ratio[%]\t")
        f.write("Total_count\n")
        
        for i, time in enumerate(bin_centers):
            f.write(f"{time:.1f}\t")
            total = 0
            for label in sorted(unique_labels):
                count = label_counts[label][i]
                ratio = label_ratios[label][i]
                f.write(f"{count}\t{ratio:.1f}\t")
                total += count
            f.write(f"{total}\n")
    
    print(f"時間割合統計データ保存: {stats_path}")

def create_horizontal_ratio_histogram(data, bin_size_m=50.0, output_dir='rock_statics', label_filter=None, suffix=''):
    """
    水平位置ごとの岩石ラベルの割合ヒストグラムを作成
    
    Parameters:
    -----------
    data : dict
        ラベルデータ
    bin_size_m : float
        水平位置ビンサイズ [m]
    output_dir : str
        出力ディレクトリ
    label_filter : list or None
        含めるラベル番号のリスト（None=全て）
    suffix : str
        ファイル名のサフィックス
    """
    x_coords = data['x']
    labels = data['label']
    
    # ラベルフィルタリング
    if label_filter is not None:
        mask = np.isin(labels, label_filter)
        x_coords_filtered = x_coords[mask]
        labels_filtered = labels[mask]
    else:
        x_coords_filtered = x_coords
        labels_filtered = labels
    
    if len(labels_filtered) == 0:
        print(f"警告: フィルタリング後のデータが空です (フィルタ: {label_filter})")
        return
    
    # ビン設定（0スタート）
    x_max = data['x'].max()
    bin_start = 0
    bin_end = np.ceil(x_max / bin_size_m) * bin_size_m
    bins = np.arange(bin_start, bin_end + bin_size_m, bin_size_m)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # ラベル別にヒストグラム計算
    unique_labels = np.unique(labels_filtered)
    label_counts = {}
    label_ratios = {}
    
    for label in unique_labels:
        mask = labels_filtered == label
        counts, _ = np.histogram(x_coords_filtered[mask], bins=bins)
        label_counts[label] = counts
    
    # 各ビンでの割合を計算
    for label in unique_labels:
        ratios = np.zeros(len(bin_centers))
        for i in range(len(bin_centers)):
            total_count = sum(label_counts[l][i] for l in unique_labels)
            if total_count > 0:
                ratios[i] = (label_counts[label][i] / total_count) * 100  # パーセント
            else:
                ratios[i] = 0
        label_ratios[label] = ratios
    
    # プロット作成
    plt.figure(figsize=(10, 8))
    
    # 積み上げヒストグラム（割合）
    bottom = np.zeros(len(bin_centers))
    
    for label in sorted(unique_labels):
        plt.bar(bin_centers, label_ratios[label], 
               width=bin_size_m * 0.8, 
               bottom=bottom,
               label=f'Group {label}' if label <= 3 else f'Label {label}',
               color=get_label_color(label), alpha=0.7)
        bottom += label_ratios[label]
    
    plt.xlabel('Horizontal position [m]', fontsize=20)
    plt.ylabel('Percentage [%]', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    title_text = 'Rock ratio distribution by horizontal position'
    if label_filter is not None:
        title_text += f' (Labels {label_filter})'
    # Use 2 columns for legend if 6 labels are present
    ncol = 2 if len(unique_labels) == 6 else 1
    plt.legend(fontsize=18, ncol=ncol)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limit to 0-100%
    plt.ylim(0, 100)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    filename = f'horizontal_ratio_histogram{suffix}'
    png_path = os.path.join(output_dir, f'{filename}.png')
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"水平位置割合ヒストグラム保存: {png_path}")
    
    # 統計データ保存
    stats_path = os.path.join(output_dir, f'horizontal_ratio_statistics{suffix}.txt')
    with open(stats_path, 'w') as f:
        f.write("# Horizontal position ratio statistics\n")
        f.write(f"# Bin size: {bin_size_m} m\n")
        if label_filter is not None:
            f.write(f"# Label filter: {label_filter}\n")
        f.write("# Position_center[m]\t")
        for label in sorted(unique_labels):
            f.write(f"Label_{label}_count\tLabel_{label}_ratio[%]\t")
        f.write("Total_count\n")
        
        for i, pos in enumerate(bin_centers):
            f.write(f"{pos:.2f}\t")
            total = 0
            for label in sorted(unique_labels):
                count = label_counts[label][i]
                ratio = label_ratios[label][i]
                f.write(f"{count}\t{ratio:.1f}\t")
                total += count
            f.write(f"{total}\n")
    
    print(f"水平位置割合統計データ保存: {stats_path}")

def print_summary_statistics(data):
    """
    データの概要統計を表示
    
    Parameters:
    -----------
    data : dict
        ラベルデータ
    """
    print("\n=== データ概要 ===")
    print(f"総ラベル数: {len(data['label'])}")
    print(f"水平位置範囲: {data['x'].min():.2f} - {data['x'].max():.2f} m")
    print(f"時間範囲: {data['y'].min():.1f} - {data['y'].max():.1f} ns")
    print(f"深さ範囲: {time_to_depth(data['y'].min()):.3f} - {time_to_depth(data['y'].max()):.3f} m (ε_r=4.5)")
    
    print("\nラベル別個数:")
    unique_labels, counts = np.unique(data['label'], return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count}個 ({count/len(data['label'])*100:.1f}%)")

def main():
    """
    メイン処理
    """
    print("岩石ラベル統計プロットツール")
    print("=" * 50)
    
    # ファイル入力
    json_path = input("JSONファイルのパスを入力してください: ").strip()
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {json_path}")
    
    if not json_path.lower().endswith('.json'):
        raise ValueError("JSONファイルを指定してください")
    
    # 深さ計測データのファイル入力
    depth_json_path = input("深さ計測JSONファイルのパスを入力してください（空=スキップ）: ").strip()
    depth_data = None
    if depth_json_path:
        if os.path.exists(depth_json_path):
            depth_data = load_depth_measurements(depth_json_path)
            if depth_data is not None:
                print(f"深さ計測データを読み込みました: {len(depth_data['x'])}点")
            else:
                print("深さ計測データの読み込みに失敗しました")
        else:
            print(f"警告: 深さ計測ファイルが見つかりません: {depth_json_path}")
    
    # ウィンドウサイズ入力
    horizontal_bin = float(input("水平位置ビンサイズ [m] を入力してください: ").strip())
    time_bin = float(input("時間ビンサイズ [ns] を入力してください: ").strip())
    
    # データ範囲の入力
    print("\n=== 入力データ範囲 ===")
    time_range = input("深さ範囲 [ns] を入力してください（例: 0-100, Enter: 全範囲）: ").strip()
    horizontal_range = input("水平位置範囲 [m] を入力してください（例: 0-100, Enter: 全範囲）: ").strip()
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
        raise ValueError("範囲の入力形式が正しくありません。例: 0-100")

    # 出力ディレクトリ設定
    base_dir = os.path.dirname(os.path.dirname(json_path))
    filename = os.path.splitext(os.path.basename(json_path))[0]
    output_base_dir = os.path.join(base_dir, f'label_statics/{filename}_x{horizontal_bin}_t{time_bin}')
    if time_range:
        output_base_dir += f'/t{time_min}-{time_max}'
    elif horizontal_range:
        output_base_dir += f'/x{horizontal_min}-{horizontal_max}'
    elif time_range and horizontal_range:
        output_base_dir += f'/t{time_min}-{time_max}_x{horizontal_min}-{horizontal_max}'
    else:
        output_base_dir += '/full_range'
    output_time_dir = os.path.join(output_base_dir, 'depth_histograms')
    output_horizontal_dir = os.path.join(output_base_dir, 'horizontal_histograms')
    
    # データ読み込み
    print("\nデータを読み込み中...")
    data = load_label_data(json_path)

    # データ範囲フィルタリング
    if time_min is not None and time_max is not None:
        time_mask = (data['y'] >= time_min) & (data['y'] <= time_max)
        data = {key: value[time_mask] for key, value in data.items()}
    if horizontal_min is not None and horizontal_max is not None:
        horizontal_mask = (data['x'] >= horizontal_min) & (data['x'] <= horizontal_max)
        data = {key: value[horizontal_mask] for key, value in data.items()}
    
    # 概要統計表示
    print_summary_statistics(data)
    
    # 深さ計測データの概要表示
    if depth_data is not None:
        print(f"\n=== 深さ計測データ概要 ===")
        print(f"計測点数: {len(depth_data['x'])}")
        print(f"x座標範囲: {depth_data['x'].min():.2f} - {depth_data['x'].max():.2f} m")
        print(f"深さ範囲: {depth_data['depth'].min():.3f} - {depth_data['depth'].max():.3f} m")
    
    print(f"\n設定:")
    print(f"  水平位置ビンサイズ: {horizontal_bin} m")
    print(f"  時間ビンサイズ: {time_bin} ns")
    print(f"  出力ディレクトリ: {output_base_dir}")
    print(f"    - 時間方向ヒストグラム: {output_time_dir}")
    print(f"    - 水平位置ヒストグラム: {output_horizontal_dir}")
    print(f"  誘電率: 4.5")
    print(f"  深さ規格化: {'有効' if depth_data is not None else '無効'}")
    
    # 岩石のみ（ラベル1-3）のヒストグラム作成
    print("\n岩石のみ（ラベル1-3）のヒストグラムを作成中...")
    rock_labels = [1, 2, 3]
    # create_depth_histogram(data, bin_size_m=depth_bin, output_dir=output_basic_dir, 
    #                       label_filter=rock_labels, suffix='_rocks_only')
    create_horizontal_histogram(data, bin_size_m=horizontal_bin, output_dir=output_horizontal_dir, 
                               label_filter=rock_labels, suffix='_rocks_only')
    
    # 時間ヒストグラム（岩石のみ）
    print("\n時間ヒストグラム（岩石のみ）を作成中...")
    create_time_histogram(data, bin_size_ns=time_bin, output_dir=output_time_dir, 
                                label_filter=rock_labels, suffix='_rocks_only')
    # create_depth_ratio_histogram_time(data, bin_size_ns=time_bin, output_dir=output_time_dir, 
    #                                  label_filter=rock_labels, suffix='_rocks_only')
    
    # 割合ヒストグラム（岩石のみ）
    print("\n割合ヒストグラム（岩石のみ）を作成中...")
    # create_depth_ratio_histogram(data, bin_size_m=depth_bin, output_dir=output_ratio_dir, 
    #                             label_filter=rock_labels, suffix='_rocks_only')
    create_horizontal_ratio_histogram(data, bin_size_m=horizontal_bin, output_dir=output_horizontal_dir, 
                                     label_filter=rock_labels, suffix='_rocks_only')
    
    # 深さ規格化水平ヒストグラム（岩石のみ）
    # if depth_data is not None:
    #     print("\n深さ規格化水平ヒストグラム（岩石のみ）を作成中...")
    #     create_depth_normalized_horizontal_histogram(data, depth_data, bin_size_m=horizontal_bin, 
    #                                                output_dir=output_normalized_dir, label_filter=rock_labels, 
    #                                                suffix='_rocks_only')
        
    #     # 薄い層規格化水平ヒストグラム（岩石のみ）
    #     print("\n薄い層規格化水平ヒストグラム（岩石のみ）を作成中...")
    #     create_thin_layer_normalized_horizontal_histogram(data, depth_data, bin_size_m=horizontal_bin, 
    #                                                     output_dir=output_normalized_dir, label_filter=rock_labels, 
    #                                                     suffix='_rocks_only')
    
    # ラベル4-6のみのヒストグラム作成
    print("\nラベル4-6のみのヒストグラムを作成中...")
    non_rock_labels = [4, 5, 6]
    # create_depth_histogram(data, bin_size_m=depth_bin, output_dir=output_basic_dir, 
    #                       label_filter=non_rock_labels, suffix='_labels_4to6_only')
    create_horizontal_histogram(data, bin_size_m=horizontal_bin, output_dir=output_horizontal_dir, 
                               label_filter=non_rock_labels, suffix='_labels_4to6_only')
    
    # 時間ヒストグラム（ラベル4-6のみ）
    print("\n時間ヒストグラム（ラベル4-6のみ）を作成中...")
    create_time_histogram(data, bin_size_ns=time_bin, output_dir=output_time_dir, 
                               label_filter=non_rock_labels, suffix='_labels_4to6_only')
    create_depth_ratio_histogram_time(data, bin_size_ns=time_bin, output_dir=output_time_dir, 
                                     label_filter=non_rock_labels, suffix='_labels_4to6_only')
    
    # 全ラベルのヒストグラム作成
    print("\n全ラベルのヒストグラムを作成中...")
    # create_depth_histogram(data, bin_size_m=depth_bin, output_dir=output_basic_dir, 
    #                       label_filter=None, suffix='_all_labels')
    create_horizontal_histogram(data, bin_size_m=horizontal_bin, output_dir=output_horizontal_dir, 
                               label_filter=None, suffix='_all_labels')
    
    # 時間ヒストグラム（全ラベル）
    print("\n時間ヒストグラム（全ラベル）を作成中...")
    create_time_histogram(data, bin_size_ns=time_bin, output_dir=output_time_dir, 
                               label_filter=None, suffix='_all_labels')
    create_depth_ratio_histogram_time(data, bin_size_ns=time_bin, output_dir=output_time_dir, 
                                     label_filter=None, suffix='_all_labels')
    
    # 割合ヒストグラム（ラベル4-6のみ）
    print("\n割合ヒストグラム（ラベル4-6のみ）を作成中...")
    # create_depth_ratio_histogram(data, bin_size_m=depth_bin, output_dir=output_ratio_dir, 
    #                             label_filter=non_rock_labels, suffix='_labels_4to6_only')
    create_horizontal_ratio_histogram(data, bin_size_m=horizontal_bin, output_dir=output_horizontal_dir, 
                                     label_filter=non_rock_labels, suffix='_labels_4to6_only')
    
    # 割合ヒストグラム（全ラベル）
    print("\n割合ヒストグラム（全ラベル）を作成中...")
    # create_depth_ratio_histogram(data, bin_size_m=depth_bin, output_dir=output_ratio_dir, 
    #                             label_filter=None, suffix='_all_labels')
    create_horizontal_ratio_histogram(data, bin_size_m=horizontal_bin, output_dir=output_horizontal_dir, 
                                     label_filter=None, suffix='_all_labels')
    
    # 深さ規格化水平ヒストグラム（ラベル4-6のみ）
    # if depth_data is not None:
    #     print("\n深さ規格化水平ヒストグラム（ラベル4-6のみ）を作成中...")
    #     create_depth_normalized_horizontal_histogram(data, depth_data, bin_size_m=horizontal_bin, 
    #                                                output_dir=output_normalized_dir, label_filter=non_rock_labels, 
    #                                                suffix='_labels_4to6_only')
        
    #     # 薄い層規格化水平ヒストグラム（ラベル4-6のみ）
    #     print("\n薄い層規格化水平ヒストグラム（ラベル4-6のみ）を作成中...")
    #     create_thin_layer_normalized_horizontal_histogram(data, depth_data, bin_size_m=horizontal_bin, 
    #                                                     output_dir=output_normalized_dir, label_filter=non_rock_labels, 
    #                                                     suffix='_labels_4to6_only')
    
    # 深さ規格化水平ヒストグラム（全ラベル）
    # if depth_data is not None:
    #     print("\n深さ規格化水平ヒストグラム（全ラベル）を作成中...")
    #     create_depth_normalized_horizontal_histogram(data, depth_data, bin_size_m=horizontal_bin, 
    #                                                output_dir=output_normalized_dir, label_filter=None, 
    #                                                suffix='_all_labels')
        
    #     # 薄い層規格化水平ヒストグラム（全ラベル）
    #     print("\n薄い層規格化水平ヒストグラム（全ラベル）を作成中...")
    #     create_thin_layer_normalized_horizontal_histogram(data, depth_data, bin_size_m=horizontal_bin, 
    #                                                     output_dir=output_normalized_dir, label_filter=None, 
    #                                                     suffix='_all_labels')
    
    # 全体統計保存
    summary_path = os.path.join(output_base_dir, 'summary_statistics.txt')
    with open(summary_path, 'w') as f:
        f.write("# Rock label summary statistics\n")
        # f.write(f"# Settings: depth_bin={depth_bin}m, horizontal_bin={horizontal_bin}m, time_bin={time_bin}ns, epsilon_r=4.5\n")
        f.write(f"# Depth normalization: {'enabled' if depth_data is not None else 'disabled'}\n")
        if depth_data is not None:
            f.write(f"# Depth measurement file: {depth_json_path}\n")
            f.write(f"# Depth measurement points: {len(depth_data['x'])}\n")
        f.write(f"Total labels: {len(data['label'])}\n")
        f.write(f"Horizontal range: {data['x'].min():.2f} - {data['x'].max():.2f} m\n")
        f.write(f"Time range: {data['y'].min():.1f} - {data['y'].max():.1f} ns\n")
        f.write(f"Depth range: {time_to_depth(data['y'].min()):.3f} - {time_to_depth(data['y'].max()):.3f} m\n")
        f.write("\nLabel counts:\n")
        
        unique_labels, counts = np.unique(data['label'], return_counts=True)
        for label, count in zip(unique_labels, counts):
            f.write(f"Label {label}: {count} ({count/len(data['label'])*100:.1f}%)\n")
        
        # 岩石のみの統計
        rock_mask = np.isin(data['label'], [1, 2, 3])
        rock_count = np.sum(rock_mask)
        f.write(f"\nRocks only (Labels 1-3): {rock_count} ({rock_count/len(data['label'])*100:.1f}%)\n")
        
        # ラベル4-6のみの統計
        non_rock_mask = np.isin(data['label'], [4, 5, 6])
        non_rock_count = np.sum(non_rock_mask)
        f.write(f"Labels 4-6 only: {non_rock_count} ({non_rock_count/len(data['label'])*100:.1f}%)\n")
    
    print(f"\n概要統計保存: {summary_path}")
    print("\n処理完了!")

if __name__ == "__main__":
    main()