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
        1: 'red',        # red
        2: 'green',      # green
        3: 'blue',       # blue
        4: 'cyan',       # cyan
        5: 'magenta',    # magenta
        6: 'yellow'      # yellow
    }
    return color_map.get(label, 'gray')  # デフォルトはgray

def create_grid_analysis(data, window_x, window_d, label_filter=None, filter_name="all"):
    """
    グリッドベースのラベル解析を実行
    
    Parameters:
    -----------
    data : dict
        ラベルデータ
    window_x : float
        水平方向ウィンドウサイズ [m]
    window_d : float
        深さ方向ウィンドウサイズ [m]
    label_filter : list or None
        含めるラベル番号のリスト（None=全て）
    filter_name : str
        フィルタ名（ファイル名用）
    
    Returns:
    --------
    dict: グリッド解析結果
    """
    x_coords = data['x']
    y_coords = data['y']
    labels = data['label']
    
    # 深さ変換
    depths = time_to_depth(y_coords)
    
    # ラベルフィルタリング
    if label_filter is not None:
        mask = np.isin(labels, label_filter)
        x_coords = x_coords[mask]
        depths = depths[mask]
        labels = labels[mask]
    
    if len(labels) == 0:
        print(f"警告: フィルタリング後のデータが空です (フィルタ: {label_filter})")
        return None
    
    # グリッド範囲設定
    x_min, x_max = data['x'].min(), data['x'].max()
    depth_min, depth_max = time_to_depth(data['y']).min(), time_to_depth(data['y']).max()
    
    # x軸グリッドビン作成（0基準）
    x_start = 0
    x_end = np.ceil(x_max / window_x) * window_x  # 最大値を含む最小の倍数
    x_bins = np.arange(x_start, x_end + window_x, window_x)
    
    # y軸グリッドビン作成（0基準、負の値も含む）
    # 最小値と最大値を0基準のウィンドウサイズの倍数に拡張
    depth_start = np.floor(depth_min / window_d) * window_d  # 最小値を含む最大の倍数
    depth_end = np.ceil(depth_max / window_d) * window_d    # 最大値を含む最小の倍数
    depth_bins = np.arange(depth_start, depth_end + window_d, window_d)
    
    # グリッドセンター計算
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    depth_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
    
    # 各グリッドセルでのラベル集計
    grid_counts = {}
    for i, x_center in enumerate(x_centers):
        for j, depth_center in enumerate(depth_centers):
            # セル範囲内のデータを抽出
            x_mask = (x_coords >= x_bins[i]) & (x_coords < x_bins[i+1])
            d_mask = (depths >= depth_bins[j]) & (depths < depth_bins[j+1])
            cell_mask = x_mask & d_mask
            
            if np.any(cell_mask):
                cell_labels = labels[cell_mask]
                unique_labels, counts = np.unique(cell_labels, return_counts=True)
                grid_counts[(i, j)] = dict(zip(unique_labels, counts))
            else:
                grid_counts[(i, j)] = {}
    
    return {
        'x_centers': x_centers,
        'depth_centers': depth_centers,
        'x_bins': x_bins,
        'depth_bins': depth_bins,
        'grid_counts': grid_counts,
        'window_x': window_x,
        'window_d': window_d,
        'filter_name': filter_name,
        'total_points': len(labels)
    }

def calculate_optimal_figure_size(x_bins, depth_bins, window_x, window_d):
    """
    データ範囲に基づいて最適な図サイズを計算
    
    Parameters:
    -----------
    x_bins : array
        水平方向のビン
    depth_bins : array
        深さ方向のビン
    window_x : float
        水平方向ウィンドウサイズ
    window_d : float
        深さ方向ウィンドウサイズ
    
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

def plot_grid_analysis(grid_result, output_dir, suffix=""):
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
    """
    if grid_result is None:
        return
    
    x_centers = grid_result['x_centers']
    depth_centers = grid_result['depth_centers']
    x_bins = grid_result['x_bins']
    depth_bins = grid_result['depth_bins']
    grid_counts = grid_result['grid_counts']
    window_x = grid_result['window_x']
    window_d = grid_result['window_d']
    
    # 最適な図サイズを計算
    fig_width, fig_height = calculate_optimal_figure_size(x_bins, depth_bins, window_x, window_d)
    
    # CLAUDE.mdの標準フォントサイズを定義
    font_large = 20      # Titles and labels
    font_medium = 18     # Axis labels  
    font_small = 16      # Tick labels
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 各グリッドセルに円グラフを描画
    for i, x_center in enumerate(x_centers):
        for j, depth_center in enumerate(depth_centers):
            counts = grid_counts.get((i, j), {})
            
            if counts:
                # 積み上げ棒グラフの設定
                total_count = sum(counts.values())
                
                # データ範囲に基づいた基本サイズ計算
                x_range = x_bins[-1] - x_bins[0]
                depth_range = depth_bins[-1] - depth_bins[0]
                physical_aspect_ratio = x_range / depth_range
                
                # 棒グラフのパディング設定
                padding_x = window_x * 0.1  # 左右のパディング
                padding_d = window_d * 0.1  # 上下のパディング
                
                # 棒グラフの実際のサイズ
                bar_width = window_x - 2 * padding_x
                bar_height = window_d - 2 * padding_d
                
                # 棒グラフの開始位置
                bar_x = x_center - bar_width / 2
                bar_y = depth_center - bar_height / 2
                
                # ラベルをソートして一貫した表示順序を保つ
                sorted_labels = sorted(counts.keys())
                
                # 積み上げ棒グラフを描画
                current_height = 0
                for label in sorted_labels:
                    count = counts[label]
                    # 各セグメントの高さを計算（比例配分）
                    segment_height = (count / total_count) * bar_height
                    
                    # 矩形を描画
                    rect = plt.Rectangle((bar_x, bar_y + current_height),
                                       bar_width, segment_height,
                                       facecolor=get_label_color(label),
                                       edgecolor='black',
                                       linewidth=0.5,
                                       alpha=0.8)
                    ax.add_patch(rect)
                    
                    # 数値ラベルを追加（セグメントが十分大きい場合）
                    min_segment_ratio = 0.08  # 最小セグメント比率
                    if segment_height > bar_height * min_segment_ratio:
                        # 標準フォントサイズを使用
                        font_size = font_small
                        
                        # 数値を中央に配置
                        text_x = bar_x + bar_width / 2
                        text_y = bar_y + current_height + segment_height / 2
                        
                        # 背景色に応じて文字色を調整
                        text_color = 'white' if label in [1, 3, 5] else 'black'
                        ax.text(text_x, text_y, str(count), 
                               ha='center', va='center',
                               fontsize=font_size, fontweight='bold',
                               color=text_color)
                    else:
                        # 小さなセグメントの場合は棒グラフの外側に数値を表示
                        if count > 0:  # 0でない場合のみ表示
                            font_size = font_small
                            text_x = bar_x + bar_width + padding_x * 0.5
                            text_y = bar_y + current_height + segment_height / 2
                            ax.text(text_x, text_y, str(count), 
                                   ha='left', va='center',
                                   fontsize=font_size, fontweight='bold',
                                   color=get_label_color(label))
                    
                    current_height += segment_height
            else:
                # データがない場合は白い四角を描画
                rect = plt.Rectangle((x_center - window_x/2, depth_center - window_d/2),
                                   window_x, window_d, 
                                   facecolor='white', edgecolor='lightgray', linewidth=0.5)
                ax.add_patch(rect)
    
    # グリッドライン描画
    for x_bin in x_bins:
        ax.axvline(x_bin, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    for depth_bin in depth_bins:
        ax.axhline(depth_bin, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # 軸設定（アスペクト比に応じた調整）
    ax.set_xlim(x_bins[0], x_bins[-1])
    ax.set_ylim(depth_bins[0], depth_bins[-1])
    
    # 物理的な距離の比率を計算
    x_range = x_bins[-1] - x_bins[0]
    depth_range = depth_bins[-1] - depth_bins[0]
    physical_aspect_ratio = x_range / depth_range
    
    ax.set_xlabel('Horizontal position [m]', fontsize=font_medium)
    ax.set_ylabel(r'Depth [m] in $\varepsilon_r = 4.5$', fontsize=font_medium)
    
    # 軸のticks設定
    # 水平方向のticks（適切な間隔で設定）
    x_ticks = np.arange(x_bins[0], x_bins[-1], 200)
    ax.set_xticks(x_ticks)
    
    # 深さ方向のticks（適切な間隔で設定）
    depth_ticks = np.arange(depth_bins[0], depth_bins[-1], 2.0)
    ax.set_yticks(depth_ticks)
    
    ax.tick_params(axis='both', which='major', labelsize=font_small)
    ax.invert_yaxis()  # 深い方を下に
    
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
        patch = mpatches.Patch(color=get_label_color(label), label=f'Label {label}')
        legend_patches.append(patch)
    
    # 凡例のフォントサイズ（標準）
    ax.legend(handles=legend_patches, fontsize=font_medium)
    
    # タイトル（標準フォントサイズ）
    filter_text = {
        "rocks_only": "Rocks only (Labels 1-3)",
        "non_rocks_only": "Non-rocks only (Labels 4-6)",
        "all": "All labels (Labels 1-6)"
    }
    title = f"Grid Analysis - {filter_text[grid_result['filter_name']]}"
    title += f" (Window: {window_x}m x {window_d}m)"
    
    ax.set_title(title, fontsize=font_large, pad=20)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    filename = f'grid_analysis_{grid_result["filter_name"]}{suffix}'
    png_path = os.path.join(output_dir, f'{filename}.png')
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"グリッド解析プロット保存: {png_path}")

def save_grid_statistics(grid_results, output_dir, window_x, window_d):
    """
    グリッド統計をテキストファイルに保存
    
    Parameters:
    -----------
    grid_results : dict
        各フィルタのグリッド解析結果
    output_dir : str
        出力ディレクトリ
    window_x : float
        水平方向ウィンドウサイズ
    window_d : float
        深さ方向ウィンドウサイズ
    """
    stats_path = os.path.join(output_dir, 'grid_statistics.txt')
    
    with open(stats_path, 'w') as f:
        f.write("# Grid-based rock label statistics\n")
        f.write(f"# Window size: {window_x}m (horizontal) x {window_d}m (depth)\n")
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
    window_d = float(input("深さ方向ウィンドウサイズ [m] を入力してください: ").strip())
    
    # 出力ディレクトリ設定
    base_dir = os.path.dirname(os.path.dirname(json_path))
    filename = os.path.splitext(os.path.basename(json_path))[0]
    output_dir = os.path.join(base_dir, f'label_statistics_grid/{filename}_x{window_x}_d{window_d}')
    
    # データ読み込み
    print("\nデータを読み込み中...")
    data = load_label_data(json_path)
    
    print(f"総ラベル数: {len(data['label'])}")
    print(f"水平位置範囲: {data['x'].min():.2f} - {data['x'].max():.2f} m")
    depth_min = time_to_depth(data['y'].min())
    depth_max = time_to_depth(data['y'].max())
    print(f"深さ範囲: {depth_min:.3f} - {depth_max:.3f} m")
    print(f"グリッド範囲: x=0 - {np.ceil(data['x'].max() / window_x) * window_x:.0f} m, depth={np.floor(depth_min / window_d) * window_d:.2f} - {np.ceil(depth_max / window_d) * window_d:.2f} m")
    
    # 3パターンの解析実行
    print("\nグリッド解析を実行中...")
    
    # 1. 岩石のみ（ラベル1-3）
    print("岩石のみ（ラベル1-3）の解析中...")
    rocks_result = create_grid_analysis(data, window_x, window_d, [1, 2, 3], "rocks_only")
    
    # 2. 非岩石のみ（ラベル4-6）
    print("非岩石のみ（ラベル4-6）の解析中...")
    non_rocks_result = create_grid_analysis(data, window_x, window_d, [4, 5, 6], "non_rocks_only")
    
    # 3. 全ラベル（ラベル1-6）
    print("全ラベル（ラベル1-6）の解析中...")
    all_result = create_grid_analysis(data, window_x, window_d, None, "all")
    
    # プロット作成
    print("\nプロットを作成中...")
    if rocks_result:
        plot_grid_analysis(rocks_result, output_dir)
    if non_rocks_result:
        plot_grid_analysis(non_rocks_result, output_dir)
    if all_result:
        plot_grid_analysis(all_result, output_dir)
    
    # 統計データ保存
    print("\n統計データを保存中...")
    grid_results = {
        'rocks_only': rocks_result,
        'non_rocks_only': non_rocks_result,
        'all': all_result
    }
    save_grid_statistics(grid_results, output_dir, window_x, window_d)
    
    print(f"\n出力ディレクトリ: {output_dir}")
    print("処理完了!")

if __name__ == "__main__":
    main()