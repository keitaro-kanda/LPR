import numpy as np
import os
import time
from scipy.ndimage import maximum_filter, generate_binary_structure
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
    num_traces_to_keep = int(max_dist_m / dx_m)

    num_samples_to_keep = min(num_samples_to_keep, data.shape[0])
    num_traces_to_keep = min(num_traces_to_keep, data.shape[1])
    
    subset_data = data[:num_samples_to_keep, :num_traces_to_keep]
    print(f"処理対象データ範囲: {subset_data.shape} (時間サンプル数, トレース数)")
    print(f"  時間方向: {num_samples_to_keep} サンプル (~{num_samples_to_keep * dt_ns:.2f} ns)")
    print(f"  水平方向: {num_traces_to_keep} トレース (~{num_traces_to_keep * dx_m:.2f} m)")
    
    return subset_data

def perform_hough_transform(bscan_data, v_hough_pix_per_pix, amplitude_threshold_percentile=95):
    """
    Hough変換を実行して双曲線を検出する。
    """
    num_samples, num_traces = bscan_data.shape
    print(f"Hough変換を開始します。データサイズ: {num_samples}x{num_traces}")

    abs_bscan_data = np.abs(bscan_data)
    if np.all(abs_bscan_data == 0):
        print("警告: データが全て0です。エッジ点は検出されません。")
        edge_points_t, edge_points_x = np.array([]), np.array([])
    else:
        amplitude_threshold = np.percentile(abs_bscan_data[abs_bscan_data > 0], amplitude_threshold_percentile)
        print(f"エッジ検出の振幅閾値 ({amplitude_threshold_percentile}パーセンタイル): {amplitude_threshold:.4f}")
        edge_points_t, edge_points_x = np.where(abs_bscan_data >= amplitude_threshold)
    
    num_edge_points = len(edge_points_t)
    if num_edge_points == 0:
        print("エッジ点が見つかりませんでした。Hough変換をスキップします。")
        return np.zeros((num_samples, num_traces)), []

    print(f"検出されたエッジ点の数: {num_edge_points}")
    accumulator = np.zeros((num_samples, num_traces), dtype=np.int32)

    if v_hough_pix_per_pix == 0:
        print("エラー: Hough実効速度 (v_hough_pix_per_pix) が0です。計算を中止します。")
        return accumulator, []
    
    inv_v_hough_sq = (1.0 / v_hough_pix_per_pix)**2

    start_vote_time = time.time()
    for i in tqdm(range(num_edge_points), desc="投票処理中"):
        ti = edge_points_t[i]
        xi = edge_points_x[i]
        ti_sq = float(ti * ti)

        for x0_candidate in range(num_traces):
            term_x_sq = ((float(xi) - float(x0_candidate))**2) * inv_v_hough_sq
            t0_squared = ti_sq - term_x_sq
            
            if t0_squared >= 0:
                t0_candidate = int(round(np.sqrt(t0_squared)))
                if 0 <= t0_candidate < num_samples:
                    accumulator[t0_candidate, x0_candidate] += 1
        
    print("\n投票プロセス完了。")
    end_vote_time = time.time()
    print(f"投票処理時間: {end_vote_time - start_vote_time:.2f}秒")
    
    return accumulator

def find_peaks_in_accumulator(accumulator, min_votes_threshold, neighborhood_size=5):
    """
    Houghアキュムレータからピークを検出する。
    """
    print(f"Houghアキュムレータからピークを検出中... 最小投票数閾値: {min_votes_threshold}")
    
    footprint = generate_binary_structure(2, 1)
    if neighborhood_size > 1:
         footprint = np.ones((neighborhood_size, neighborhood_size), dtype=bool)

    local_max = maximum_filter(accumulator, footprint=footprint, mode='constant', cval=0)
    detected_peaks_mask = (accumulator == local_max) & (accumulator >= min_votes_threshold)
    peak_t0_coords, peak_x0_coords = np.where(detected_peaks_mask)
    peak_scores = accumulator[peak_t0_coords, peak_x0_coords]
    
    sorted_indices = np.argsort(peak_scores)[::-1]
    sorted_peak_t0 = peak_t0_coords[sorted_indices]
    sorted_peak_x0 = peak_x0_coords[sorted_indices]
    sorted_scores = peak_scores[sorted_indices]

    detected_hyperbolas = []
    for t0, x0, score in zip(sorted_peak_t0, sorted_peak_x0, sorted_scores):
        detected_hyperbolas.append({'t0_pix': t0, 'x0_pix': x0, 'score': score})
        
    print(f"検出された双曲線（ピーク）の数: {len(detected_hyperbolas)}")
    return detected_hyperbolas

def save_results(output_dir, detected_hyperbolas, v_hough_pix_per_pix, filename="detected_hyperbolas.txt"):
    """
    検出された双曲線の情報をファイルに保存する。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"出力ディレクトリを作成しました: {output_dir}")
        
    output_filepath = os.path.join(output_dir, filename)
    
    with open(output_filepath, 'w') as f:
        f.write("x0_pixel t0_pixel v_hough_pix_per_pix score\n")
        for hyp in detected_hyperbolas:
            f.write(f"{hyp['x0_pix']} {hyp['t0_pix']} {v_hough_pix_per_pix:.6f} {hyp['score']}\n")
            
    print(f"検出結果を保存しました: {output_filepath}")
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

        # 赤い「+」マークで頂点を表示
        plt.scatter(peak_x0_coords_m, peak_t0_coords_ns, 
                    s=20,  # マーカーサイズ
                    c='red', 
                    marker='+', 
                    label=f'Detected Apexes ({len(detected_hyperbolas)})')
        plt.legend()

    plt.savefig(os.path.join(output_dir, "bscan_with_hyperbolas.png"), dpi=120)
    plt.savefig(os.path.join(output_dir, "bscan_with_hyperbolas.pdf"), dpi=300)
    print(f"検出結果をプロットしたB-scan画像を保存しました")
    plt.show()
    # plt.show() # インタラクティブに表示する場合

def main():
    # --- パラメータ設定 ---
    EPSILON_R = 3.0
    DT_NS = 0.312500
    DX_M = 0.036

    MAX_TIME_NS_SUBSET = 300.0
    MAX_DIST_M_SUBSET = 400.0
    
    AMPLITUDE_THRESHOLD_PERCENTILE = 98
    MIN_VOTES_THRESHOLD = 30 # 要調整 (Houghアキュムレータのmax_votesを参考に)
    PEAK_NEIGHBORHOOD_SIZE = 5

    # --- 処理開始 ---
    v_medium_m_ns, v_hough_pix_per_pix = calculate_velocity_params(EPSILON_R, DT_NS, DX_M)
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