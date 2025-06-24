import numpy as np
import os
import time
from scipy.ndimage import maximum_filter, generate_binary_structure, gaussian_filter
from scipy import ndimage
from scipy.signal import hilbert
from tqdm import tqdm # tqdmモジュールからtqdmクラス/関数を直接インポート

# (calculate_velocity_params, load_bscan_data, perform_hough_transform,
#  find_peaks_in_accumulator, save_results 関数は変更なしなので省略します)
# --- ここから変更なしの関数群 (上記コメントの関数をここにペーストしてください) ---
def calculate_velocity_params(epsilon_r, dt_ns, dx_m):
    """
    物理パラメータから電磁波速度とピクセル単位の実効速度を計算する。
    """
    c_m_ns = 0.299792458  # 真空中の光速 (m/ns)
    v_medium_m_ns = c_m_ns / np.sqrt(epsilon_r)
    v_hough_pix_per_pix = (v_medium_m_ns * dt_ns) / (2 * dx_m)
    
    print(f"真空中の光速: {c_m_ns:.6f} m/ns")
    print(f"媒質中の電磁波速度 (v_medium): {v_medium_m_ns:.6f} m/ns")
    print(f"Hough変換用実効速度 (v_hough): {v_hough_pix_per_pix:.6f} (trace_pixel / time_sample_pixel)")
    return v_medium_m_ns, v_hough_pix_per_pix

def load_bscan_data(file_path, max_time_ns, max_dist_m, dt_ns, dx_m):
    """
    B-scanデータをtxtファイルから読み込み、指定範囲を切り出す。
    """
    print(f"B-scanデータを読み込んでいます: {file_path}")
    try:
        data = np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"エラー: ファイルの読み込みに失敗しました。 {e}")
        return None
    
    print(f"読み込み成功。データ形状: {data.shape} (時間サンプル数, トレース数)")

    num_samples_to_keep = int(max_time_ns / dt_ns)
    
    # 水平方向の制限チェック
    if max_dist_m == float('inf'):
        num_traces_to_keep = data.shape[1]  # 全トレースを使用
    else:
        num_traces_to_keep = int(max_dist_m / dx_m)

    num_samples_to_keep = min(num_samples_to_keep, data.shape[0])
    num_traces_to_keep = min(num_traces_to_keep, data.shape[1])
    
    subset_data = data[:num_samples_to_keep, :num_traces_to_keep]
    print(f"処理対象データ範囲: {subset_data.shape} (時間サンプル数, トレース数)")
    print(f"  時間方向: {num_samples_to_keep} サンプル (~{num_samples_to_keep * dt_ns:.2f} ns)")
    if max_dist_m == float('inf'):
        print(f"  水平方向: {num_traces_to_keep} トレース (~{num_traces_to_keep * dx_m:.2f} m) [制限なし]")
    else:
        print(f"  水平方向: {num_traces_to_keep} トレース (~{num_traces_to_keep * dx_m:.2f} m)")
    
    return subset_data

def envelope_based_hyperbola_edge_detection(bscan_data, v_hough_pix_per_pix, amplitude_threshold_percentile=95):
    """
    エンベロープベース双曲線エッジ検出 - 電場の正負縞模様を統合
    """
    num_samples, num_traces = bscan_data.shape
    
    # Hilbert変換によるエンベロープ計算
    print(f"エンベロープベース検出:")
    print(f"  Hilbert変換を実行中...")
    envelope_data = np.abs(hilbert(bscan_data, axis=0))
    abs_bscan_data = envelope_data
    
    if np.all(abs_bscan_data == 0):
        return np.array([]), np.array([])
    
    # 基本振幅閾値
    noise_level = np.std(abs_bscan_data[:min(50, num_samples//10), :])
    percentile_threshold = np.percentile(abs_bscan_data[abs_bscan_data > 0], amplitude_threshold_percentile)
    noise_based_threshold = noise_level * 8
    base_threshold = max(percentile_threshold, noise_based_threshold, percentile_threshold * 1.5)
    
    print(f"  基本振幅閾値: {base_threshold:.4f}")
    print(f"  元データ振幅範囲: {np.min(bscan_data):.4f} - {np.max(bscan_data):.4f}")
    print(f"  エンベロープ範囲: {np.min(envelope_data):.4f} - {np.max(envelope_data):.4f}")
    
    # 1. 基本振幅フィルタ
    amplitude_mask = abs_bscan_data >= base_threshold
    
    # 2. エンベロープ勾配ベースエッジ検出
    # エンベロープデータで時間方向と空間方向の勾配を計算
    grad_t = np.gradient(abs_bscan_data, axis=0)  # 時間方向勾配
    grad_x = np.gradient(abs_bscan_data, axis=1)  # 空間方向勾配
    grad_magnitude = np.sqrt(grad_t**2 + grad_x**2)
    
    # 勾配閾値（動的設定、効率化のため高めに設定）
    grad_threshold = np.percentile(grad_magnitude[grad_magnitude > 0], 95)  # 85→95に引き上げ
    gradient_mask = grad_magnitude > grad_threshold
    print(f"  勾配閾値: {grad_threshold:.4f}")
    
    # 3. エンベロープベース双曲線方向性フィルタ
    # エンベロープデータでSobel演算子による方向性評価
    sobel_x = ndimage.sobel(abs_bscan_data, axis=1)
    sobel_y = ndimage.sobel(abs_bscan_data, axis=0)
    
    # 双曲線特有の対称性を検出
    direction_mask = np.zeros_like(abs_bscan_data, dtype=bool)
    
    # 双曲線の対称的な勾配パターンを検出
    for i in range(2, num_samples-2):
        for j in range(2, num_traces-2):
            if amplitude_mask[i, j]:
                # 周辺の勾配パターンをチェック
                left_grad = sobel_x[i, j-1]
                right_grad = sobel_x[i, j+1]
                # 左右対称的な勾配（双曲線の特徴）
                if left_grad * right_grad < 0 and abs(left_grad + right_grad) < abs(left_grad - right_grad) * 0.5:
                    direction_mask[i, j] = True
    
    # 4. エンベロープベースマルチスケール検出
    multiscale_mask = np.zeros_like(abs_bscan_data, dtype=bool)
    
    for scale in [1, 2, 3]:  # エンベロープデータで複数スケール
        smoothed = gaussian_filter(abs_bscan_data, sigma=scale)
        scale_threshold = base_threshold * (0.9 ** scale)  # エンベロープでは閾値を高めに維持
        scale_mask = smoothed > scale_threshold
        multiscale_mask |= scale_mask
    
    # 5. 統合フィルタ
    # 複数の条件を組み合わせ
    combined_mask = amplitude_mask & (gradient_mask | direction_mask | multiscale_mask)
    
    # 6. モルフォロジー処理でノイズ除去
    # 小さな孤立点を除去
    structure = np.ones((3, 3))
    cleaned_mask = ndimage.binary_opening(combined_mask, structure=structure)
    
    # 最終エッジ点を取得
    edge_points_t, edge_points_x = np.where(cleaned_mask)
    
    print(f"  検出されたエッジ点数: {len(edge_points_t)}")
    print(f"  - 振幅フィルタ: {np.sum(amplitude_mask)} 点")
    print(f"  - 勾配フィルタ: {np.sum(gradient_mask)} 点") 
    print(f"  - 方向性フィルタ: {np.sum(direction_mask)} 点")
    print(f"  - マルチスケール: {np.sum(multiscale_mask)} 点")
    print(f"  - 統合後: {np.sum(combined_mask)} 点")
    print(f"  - クリーニング後: {len(edge_points_t)} 点")
    
    return edge_points_t, edge_points_x

def perform_hough_transform(bscan_data, v_hough_pix_per_pix, amplitude_threshold_percentile=95):
    """
    Hough変換を実行して双曲線を検出する（最適化エッジ検出版）。
    """
    num_samples, num_traces = bscan_data.shape
    print(f"Hough変換を開始します。データサイズ: {num_samples}x{num_traces}")

    # エンベロープベースエッジ検出を使用
    edge_points_t, edge_points_x = envelope_based_hyperbola_edge_detection(
        bscan_data, v_hough_pix_per_pix, amplitude_threshold_percentile
    )
    num_edge_points = len(edge_points_t)
    
    if num_edge_points == 0:
        print("エッジ点が見つかりませんでした。Hough変換をスキップします。")
        return np.zeros((num_samples, num_traces), dtype=np.int32)

    print(f"検出されたエッジ点の数: {num_edge_points}")
    
    if v_hough_pix_per_pix == 0:
        print("エラー: Hough実効速度 (v_hough_pix_per_pix) が0です。計算を中止します。")
        return np.zeros((num_samples, num_traces), dtype=np.int32)
    
    accumulator = np.zeros((num_samples, num_traces), dtype=np.int32)
    inv_v_hough_sq = (1.0 / v_hough_pix_per_pix)**2
    
    # 偽陰性重視：物理的制約を強化（誤検出を減らす）
    max_t0_samples = min(num_samples, int(500.0 / 0.3125))  # 500ns (40m深度)
    min_t0_samples = int(10.0 / 0.3125)  # 最小深度10ns（表面近くのノイズ除去）
    print(f"物理的制約: t0範囲 = {min_t0_samples}-{max_t0_samples} サンプル (10-500ns)")

    start_vote_time = time.time()
    
    # メモリ効率を考慮してエッジ点をバッチ処理
    batch_size = min(1000, num_edge_points)  # バッチサイズを調整
    num_batches = (num_edge_points + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="投票処理中"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_edge_points)
        
        # バッチ内のエッジ点を取得
        batch_t = edge_points_t[start_idx:end_idx]
        batch_x = edge_points_x[start_idx:end_idx]
        batch_size_actual = len(batch_t)
        
        # ベクトル化計算用のグリッド作成
        x0_candidates = np.arange(num_traces)
        ti_batch = batch_t[:, np.newaxis]  # (batch_size, 1)
        xi_batch = batch_x[:, np.newaxis]  # (batch_size, 1)
        x0_grid = x0_candidates[np.newaxis, :]  # (1, num_traces)
        
        # 双曲線方程式の計算 (ベクトル化)
        ti_sq = ti_batch.astype(np.float64) ** 2
        dx_sq = (xi_batch.astype(np.float64) - x0_grid.astype(np.float64)) ** 2
        t0_squared = ti_sq - dx_sq * inv_v_hough_sq
        
        # 有効な t0 を計算
        valid_mask = t0_squared >= 0
        t0_candidates = np.sqrt(np.maximum(t0_squared, 0))
        t0_indices = np.round(t0_candidates).astype(np.int32)
        
        # 範囲チェック（最小・最大深度制約を適用）
        range_mask = (t0_indices >= min_t0_samples) & (t0_indices < max_t0_samples) & valid_mask
        
        # 投票処理
        valid_t0 = t0_indices[range_mask]
        valid_x0 = np.broadcast_to(x0_grid, (batch_size_actual, num_traces))[range_mask]
        
        if len(valid_t0) > 0:
            # numpy.add.at を使用して効率的にカウント
            np.add.at(accumulator, (valid_t0, valid_x0), 1)
    
    end_vote_time = time.time()
    print(f"投票処理時間: {end_vote_time - start_vote_time:.2f}秒")
    print(f"アキュムレータの最大投票数: {np.max(accumulator)}")
    
    return accumulator

def find_peaks_in_accumulator(accumulator, min_votes_threshold=None, neighborhood_size=5):
    """
    Houghアキュムレータからピークを検出する（適応的閾値・網羅的探索版）。
    """
    max_votes = np.max(accumulator)
    print(f"アキュムレータ統計: 最大投票数={max_votes}, 平均={np.mean(accumulator):.2f}, 標準偏差={np.std(accumulator):.2f}")
    
    # 偽陰性重視：厳格な適応的閾値設定
    if min_votes_threshold is None:
        # 全データ処理に対応した適応的閾値（偽陰性重視だが実用的）
        vote_std = np.std(accumulator)
        vote_mean = np.mean(accumulator)
        adaptive_threshold = max(10, int(vote_mean + 3 * vote_std))  # 4σ→3σに緩和
        max_based_threshold = max(20, max_votes // 5)  # 最大値の1/5に緩和
        min_votes_threshold = min(adaptive_threshold, max_based_threshold)
        print(f"厳格な適応的閾値を設定: {min_votes_threshold} (統計ベース: {adaptive_threshold}, 最大値ベース: {max_based_threshold})")
    else:
        print(f"指定された最小投票数閾値: {min_votes_threshold}")
    
    # 近傍サイズの適応的調整
    adaptive_neighborhood = max(3, min(neighborhood_size, 7))  # 3-7の範囲で調整
    footprint = np.ones((adaptive_neighborhood, adaptive_neighborhood), dtype=bool)
    
    # ローカルマキシマフィルタ
    local_max = maximum_filter(accumulator, footprint=footprint, mode='constant', cval=0)
    detected_peaks_mask = (accumulator == local_max) & (accumulator >= min_votes_threshold)
    
    peak_t0_coords, peak_x0_coords = np.where(detected_peaks_mask)
    peak_scores = accumulator[peak_t0_coords, peak_x0_coords]
    
    # 偽陰性重視：厳格な物理的妥当性チェック
    valid_indices = []
    for i, (t0, x0, score) in enumerate(zip(peak_t0_coords, peak_x0_coords, peak_scores)):
        # 深度制約 (500ns = 40m相当、表面10ns以上)
        time_ns = t0 * 0.3125
        if time_ns < 10 or time_ns > 500:  # 10-500nsの範囲
            continue
        
        # 適度な最低投票数（偽陰性重視だが極端すぎない）
        if score < max(8, max_votes // 15):  # 最低8票、または最大値の1/15に緩和
            continue
            
        # 双曲線の品質チェック：周辺のアキュムレータ値との比較
        # 周辺8点の平均値と比較して、十分に突出しているかチェック
        t_min, t_max = max(0, t0-1), min(accumulator.shape[0], t0+2)
        x_min, x_max = max(0, x0-1), min(accumulator.shape[1], x0+2)
        neighborhood = accumulator[t_min:t_max, x_min:x_max]
        if len(neighborhood[neighborhood != score]) > 0:
            neighborhood_mean = np.mean(neighborhood[neighborhood != score])  # 自分以外の平均
            if score < neighborhood_mean * 2.0:  # 周辺の2倍以上に緩和（2.5倍→2倍）
                continue
            
        valid_indices.append(i)
    
    if len(valid_indices) == 0:
        print("物理的制約を満たすピークが見つかりませんでした。")
        return []
    
    # 有効なピークのみを抽出
    valid_t0 = peak_t0_coords[valid_indices]
    valid_x0 = peak_x0_coords[valid_indices]
    valid_scores = peak_scores[valid_indices]
    
    # スコア順にソート
    sorted_indices = np.argsort(valid_scores)[::-1]
    sorted_peak_t0 = valid_t0[sorted_indices]
    sorted_peak_x0 = valid_x0[sorted_indices]
    sorted_scores = valid_scores[sorted_indices]

    detected_hyperbolas = []
    for t0, x0, score in zip(sorted_peak_t0, sorted_peak_x0, sorted_scores):
        # 物理的パラメータの計算
        time_ns = t0 * 0.3125  # 時間[ns]
        # 深度計算: depth = v * t / 2, v = c / sqrt(εr)
        c_m_ns = 0.299792458  # m/ns
        v_medium = c_m_ns / np.sqrt(3.0)  # 媒質中の波速
        depth_m = v_medium * time_ns * 1e-9 / 2  # 深度[m]
        
        detected_hyperbolas.append({
            't0_pix': int(t0), 
            'x0_pix': int(x0), 
            'score': int(score),
            'depth_m': depth_m,
            'time_ns': time_ns
        })
    
    print(f"検出された双曲線（ピーク）の数: {len(detected_hyperbolas)}")
    if len(detected_hyperbolas) > 0:
        print(f"スコア範囲: {sorted_scores[-1]} - {sorted_scores[0]}")
        print(f"深度範囲: {detected_hyperbolas[-1]['depth_m']:.2f}m - {detected_hyperbolas[0]['depth_m']:.2f}m")
    
    return detected_hyperbolas

def save_results(output_dir, detected_hyperbolas, v_hough_pix_per_pix, filename="detected_hyperbolas.txt"):
    """
    検出された双曲線の情報をファイルに保存する（拡張版）。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"出力ディレクトリを作成しました: {output_dir}")
        
    output_filepath = os.path.join(output_dir, filename)
    
    with open(output_filepath, 'w') as f:
        # ヘッダーを拡張
        f.write("x0_pixel t0_pixel v_hough_pix_per_pix score time_ns depth_m\n")
        for hyp in detected_hyperbolas:
            f.write(f"{hyp['x0_pix']} {hyp['t0_pix']} {v_hough_pix_per_pix:.6f} {hyp['score']} {hyp['time_ns']:.3f} {hyp['depth_m']:.3f}\n")
            
    print(f"検出結果を保存しました: {output_filepath}")
    print(f"保存した双曲線数: {len(detected_hyperbolas)}")
    
    # 統計情報も保存
    if detected_hyperbolas:
        stats_filepath = os.path.join(output_dir, "detection_statistics.txt")
        with open(stats_filepath, 'w') as f:
            f.write("=== Hyperbola Detection Statistics ===\n")
            f.write(f"Total detected hyperbolas: {len(detected_hyperbolas)}\n")
            f.write(f"Score range: {min(h['score'] for h in detected_hyperbolas)} - {max(h['score'] for h in detected_hyperbolas)}\n")
            f.write(f"Depth range: {min(h['depth_m'] for h in detected_hyperbolas):.3f}m - {max(h['depth_m'] for h in detected_hyperbolas):.3f}m\n")
            f.write(f"Time range: {min(h['time_ns'] for h in detected_hyperbolas):.3f}ns - {max(h['time_ns'] for h in detected_hyperbolas):.3f}ns\n")
            f.write(f"Used v_hough: {v_hough_pix_per_pix:.6f} pix/pix\n")
        print(f"統計情報を保存しました: {stats_filepath}")
# --- ここまで変更なしの関数群 ---


def plot_detected_hyperbolas_on_bscan(bscan_data, detected_hyperbolas, dt_ns, dx_m, output_dir):
    """
    B-scanデータ上に検出された双曲線の頂点をプロットして保存する。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlibがインストールされていません。B-scan画像と検出結果のプロットをスキップします。")
        return

    num_samples, num_traces = bscan_data.shape
    
    plt.figure(figsize=(18, 8), tight_layout=True)
    # B-scanデータをグレースケールで表示
    # imshowのextentで物理的な軸ラベルを設定
    plt.imshow(bscan_data, aspect='auto', cmap='viridis', interpolation='nearest',
               extent=[0, num_traces * dx_m, num_samples * dt_ns, 0],
               vmin = -np.amax(np.abs(bscan_data))/15, vmax = np.amax(np.abs(bscan_data))/15
               ) # y軸は時間が下向き
    plt.colorbar(label='Amplitude')
    plt.title('B-scan Data with Detected Hyperbola Apexes')
    plt.xlabel(f'Distance (m) ({num_traces} traces)')
    plt.ylabel(f'Time (ns) ({num_samples} samples)')

    # 検出された双曲線の頂点をプロット
    if detected_hyperbolas:
        peak_x0_coords_m = [hyp['x0_pix'] * dx_m for hyp in detected_hyperbolas]
        peak_t0_coords_ns = [hyp['t0_pix'] * dt_ns for hyp in detected_hyperbolas]
        scores = [hyp['score'] for hyp in detected_hyperbolas] # スコアに応じて色やサイズを変えることも可能

        # 検出スコアに基づいてサイズを調整
        marker_sizes = [min(50, max(10, hyp['score'] * 2)) for hyp in detected_hyperbolas]
        
        # 赤い「+」マークで頂点を表示
        plt.scatter(peak_x0_coords_m, peak_t0_coords_ns, 
                    s=marker_sizes,  # スコアに応じたマーカーサイズ
                    c='red', 
                    marker='+', 
                    label=f'Detected Apexes ({len(detected_hyperbolas)})',
                    alpha=0.8)
        plt.legend()

    plt.savefig(os.path.join(output_dir, "bscan_with_hyperbolas.png"), dpi=120)
    plt.savefig(os.path.join(output_dir, "bscan_with_hyperbolas.pdf"), dpi=300)
    print(f"検出結果をプロットしたB-scan画像を保存しました")
    plt.show()
    # plt.show() # インタラクティブに表示する場合

def main():
    # --- パラメータ設定 ---
    EPSILON_R = 3.0  # 固定値
    DT_NS = 0.312500
    DX_M = 0.036

    # 処理範囲（垂直方向のみ制限、水平方向は全データ）
    MAX_TIME_NS_SUBSET = 500.0  # 40m深度相当
    MAX_DIST_M_SUBSET = float('inf')  # 水平方向は制限なし（全データ処理）
    
    # エンベロープベースエッジ検出により縞模様双曲線を統合検出（効率化）
    AMPLITUDE_THRESHOLD_PERCENTILE = 97  # エンベロープでも高い閾値で効率化
    MIN_VOTES_THRESHOLD = None  # 適応的閾値を使用（厳格化済み）
    PEAK_NEIGHBORHOOD_SIZE = 7  # 5→7に拡大（より厳格なローカルマキシマ）

    # --- 処理開始 ---
    _, v_hough_pix_per_pix = calculate_velocity_params(EPSILON_R, DT_NS, DX_M)
    if v_hough_pix_per_pix <= 1e-6:
        print("エラー: 計算されたHough実効速度が非常に小さいです。パラメータを確認してください。")
        return

    bscan_file_path = input("B-scanデータのtxtファイルパスを入力してください: ")
    if not os.path.isfile(bscan_file_path):
        print(f"エラー: ファイルが見つかりません: {bscan_file_path}")
        return

    bscan_data_subset = load_bscan_data(bscan_file_path, MAX_TIME_NS_SUBSET, MAX_DIST_M_SUBSET, DT_NS, DX_M)
    if bscan_data_subset is None or bscan_data_subset.size == 0:
        print("データが読み込めませんでした。処理を終了します。")
        return

    start_hough_time = time.time()
    accumulator = perform_hough_transform(bscan_data_subset, v_hough_pix_per_pix, AMPLITUDE_THRESHOLD_PERCENTILE)
    end_hough_time = time.time()
    print(f"Hough変換全体の処理時間: {end_hough_time - start_hough_time:.2f}秒")

    if np.all(accumulator == 0):
        print("Houghアキュムレータが全て0です。双曲線は検出されませんでした。")
        detected_hyperbolas = []
    else:
        print(f"Houghアキュムレータの最大投票数: {np.max(accumulator)}")
        detected_hyperbolas = find_peaks_in_accumulator(accumulator, MIN_VOTES_THRESHOLD, PEAK_NEIGHBORHOOD_SIZE)

    input_dir = os.path.dirname(bscan_file_path)
    output_subdir = "hyporbola_detection_Hough"
    output_dir_path = os.path.join(input_dir, output_subdir)
    
    save_results(output_dir_path, detected_hyperbolas, v_hough_pix_per_pix)
    
    # (オプション) Houghアキュムレータを画像として保存
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(18, 8), tight_layout=True)
        # extentをピクセル単位から物理単位に変更 (オプション)
        # num_samples_acc, num_traces_acc = accumulator.shape
        # plt.imshow(accumulator, aspect='auto', cmap='viridis', interpolation='nearest',
        #            extent=[0, num_traces_acc * DX_M, num_samples_acc * DT_NS, 0])
        plt.imshow(accumulator, aspect='auto', cmap='viridis', interpolation='nearest') # ピクセル単位のまま
        plt.title(f'Hough Accumulator (max votes: {np.max(accumulator)})')
        plt.xlabel('x0 (Trace Index)')
        plt.ylabel('t0 (Time Sample Index)')
        plt.colorbar(label='Votes')
        plt.savefig(os.path.join(output_dir_path, "hough_accumulator.png"), dpi=120)
        plt.savefig(os.path.join(output_dir_path, "hough_accumulator.pdf"), dpi=300)
        print(f"Houghアキュムレータの画像を保存しました)")
        plt.close() # 次のプロットのために図を閉じる
    except ImportError:
        print("Matplotlibがインストールされていません。Houghアキュムレータの画像保存をスキップします。")
    except Exception as e:
        print(f"Houghアキュムレータの画像保存中にエラーが発生しました: {e}")

    # === 新しく追加した部分: 検出結果をB-scan画像にプロット ===
    if bscan_data_subset is not None and detected_hyperbolas: # データと検出結果がある場合のみプロット
        plot_detected_hyperbolas_on_bscan(bscan_data_subset, detected_hyperbolas, DT_NS, DX_M, output_dir_path)
    elif not detected_hyperbolas:
        print("検出された双曲線がないため、B-scan画像へのプロットはスキップします。")
    # =======================================================

    print("処理が完了しました。")

if __name__ == "__main__":
    main()