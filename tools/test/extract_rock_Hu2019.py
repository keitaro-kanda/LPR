import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import emd # emd (EMD-signal) ライブラリを使用
import os
from tqdm import tqdm

# --- 1. テストデータの作成 ---
def create_synthetic_lpr_data(
    time, distance, delta_t, delta_x, # time [s], distance [m]
    rock_locations_true, noise_level_A, noise_level_B
):
    """
    合成LPRデータ（CH-2A, CH-2B）を生成する。
    岩石（回折ハイパーボラ）とノイズを含む。
    """
    time = np.arange(0, time, delta_t) # [s]
    distance = np.arange(0, distance, delta_x) # [m]
    data_A = np.zeros((time.size, distance.size))
    data_B = np.zeros((time.size, distance.size))

    # 背景ノイズ (ガウシアンランダムノイズ)
    data_A += np.random.normal(0, noise_level_A, size=(time.size, distance.size))
    data_B += np.random.normal(0, noise_level_B, size=(time.size, distance.size))

    # シンプルな水平反射 (低ディップ成分)
    reflection_time_top = 150e-9 # [s]
    echo_time_width = 10e-9 # [s] 反射の時間幅
    lambda_t = echo_time_width * 2 / 3 # 反射の時間幅を1.5波長とする
    for d in range(distance.size):
        if 0 <= int(reflection_time_top/delta_t) < time.size:
            reflection_time_width = np.arange(reflection_time_top , reflection_time_top + echo_time_width, delta_t)
            for reflection_time in reflection_time_width:
                if 0 <= int(reflection_time/delta_t) < time.size:
                    data_A[int(reflection_time/delta_t), d] += 5.0 * np.sin(2 * np.pi * (reflection_time_top - reflection_time) / lambda_t)
                    data_B[int(reflection_time/delta_t), d] += 5.0 * np.sin(2 * np.pi * (reflection_time_top - reflection_time) / lambda_t)

    # 岩石（回折ハイパーボラ）を追加
    if rock_locations_true is None:
        # デフォルトの岩石位置 (t0 [s], x0 [m], amp)
        rock_locations_true = [
            (50e-9, 30, 3.0), (35e-9, 10, 15), (130e-9, 40, 10), (250e-9, 35, 4.0)
        ]
    rock_locations_true = np.array(rock_locations_true)
    offset_range = 10 # [m]

    for t0, x0, amp in rock_locations_true:
        for x1 in np.arange(x0 - offset_range , x0 + offset_range, delta_x):
            epsiron_r = 4.5
            v = 3e8 / np.sqrt(epsiron_r) # 波速度
            time_delay_top = t0 * np.sqrt(1 + ((x1 - x0) / (v * t0 / 2))**2) # [s]
            time_delays = np.arange(time_delay_top, time_delay_top + echo_time_width, delta_t) # [s]

            if 0 <= int(time_delay_top/delta_t) < time.size and 0 <= int(x1/delta_x) < distance.size:
                for time_delay in time_delays:
                    data_A[int(time_delay/delta_t), int(x1/delta_x)] += amp * np.exp(-((x1 - x0)**2 / (2 * 1**2)))\
                                * np.sin(2 * np.pi * (time_delay_top - time_delay) / lambda_t) # ガウシアン的な減衰
                    data_B[int(time_delay/delta_t), int(x1/delta_x)] += amp * np.exp(-((x1 - x0)**2 / (2 * 1**2)))\
                                * np.sin(2 * np.pi * (time_delay_top - time_delay) / lambda_t) # ガウシアン的な減衰
            else:
                continue

    return data_A, data_B, rock_locations_true


# --- 2. データ前処理 (論文のPipelineの一部を模倣) ---
def preprocess_lpr_data(data):
    """
    LPRデータの前処理パイプラインの簡略化された実装。
    ここでは、バンドパスフィルターと背景除去を模倣する。
    """
    processed_data = gaussian_filter(data, sigma=0.5)
    return processed_data


# --- 3. f-x EMD Dip Filter ---
# コード上部にあるこの関数定義のコメントアウト（""" で囲まれている部分）を解除する
def fx_emd_dip_filter(data, p):
    """
    f-x EMDに基づくディップフィルター。
    p: 保持するIMFの数（p=1はIMF0を保持、p=2はIMF0とIMF1を保持）
    今回は「高ディップ成分を保持（高域通過フィルター）」する目的で実装する。
    """
    num_time_samples, num_distance_traces = data.shape
    filtered_data = np.zeros_like(data, dtype=float)

    # デバッグ用の出力ディレクトリを作成 (ここはそのままでOK)
    output_dir_EMD_real = os.path.join(output_dir, 'emd_debug_real')
    if not os.path.exists(output_dir_EMD_real):
        os.makedirs(output_dir_EMD_real)
    output_dir_EMD_imag = os.path.join(output_dir, 'emd_debug_imag')
    if not os.path.exists(output_dir_EMD_imag):
        os.makedirs(output_dir_EMD_imag)

    # 各時間サンプル（行）について、距離軸方向にFFTを適用
    data_fk_domain = np.fft.rfft(data, axis=1) # 距離軸 (axis=1) に対してFFT

    # 各時間周波数スライス（各行）に対してEMDを適用
    for time_idx in tqdm(range(num_time_samples), desc="Applying f-x EMD Dip Filter (spatial EMD)"):
        freq_slice_real = data_fk_domain[time_idx, :].real
        freq_slice_imag = data_fk_domain[time_idx, :].imag

        # ==== デバッグ用: IMFの可視化 ==== (この部分はそのまま有効で良いです)
        if time_idx % 50 == 0:
            try:
                imfs_real_debug = emd.sift.sift(freq_slice_real)
                imfs_imag_debug = emd.sift.sift(freq_slice_imag)

                max_imfs_to_plot = min(imfs_real_debug.shape[0], 10)
                plt.figure(figsize=(12, 2 * max_imfs_to_plot))
                plt.suptitle(f'IMF Decomposition for Time Freq Slice {time_idx} (Real Part)')
                for imf_n in range(max_imfs_to_plot):
                    plt.subplot(max_imfs_to_plot, 1, imf_n + 1)
                    plt.plot(imfs_real_debug[imf_n, :])
                    plt.title(f'IMF{imf_n} (High Dip/Frequency to Low Dip/Frequency)')
                    plt.ylabel('Amplitude')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(os.path.join(output_dir_EMD_real, f'imf_decomposition_real_{time_idx}.png'))
                plt.close()

                max_imfs_to_plot_imag = min(imfs_imag_debug.shape[0], 5)
                plt.figure(figsize=(12, 2 * max_imfs_to_plot_imag))
                plt.suptitle(f'IMF Decomposition for Time Freq Slice {time_idx} (Imag Part)')
                for imf_n in range(max_imfs_to_plot_imag):
                    plt.subplot(max_imfs_to_plot_imag, 1, imf_n + 1)
                    plt.plot(imfs_imag_debug[imf_n, :])
                    plt.title(f'IMF{imf_n} (High Dip/Frequency to Low Dip/Frequency)')
                    plt.ylabel('Amplitude')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(os.path.join(output_dir_EMD_imag, f'imf_decomposition_imag_{time_idx}.png'))
                plt.close()

            except Exception as e:
                print(f"Error plotting IMFs at time_idx {time_idx}: {e}")

        # 実数部に対するEMD (空間周波数成分を分離)
        imfs_real = emd.sift.sift(freq_slice_real)
        # 虚数部に対するEMD (空間周波数成分を分離)
        imfs_imag = emd.sift.sift(freq_slice_imag)

        # フィルター処理された信号の再構築: p個のIMFを保持
        # ここでのIMFは空間的な波数成分に対応し、IMF0, IMF1...が高ディップ成分
        
        reconstructed_real = np.zeros_like(freq_slice_real, dtype=float)
        # IMFがp個以上存在する場合にのみ、p個のIMFを合計して保持
        if imfs_real.shape[0] >= p:
            # imfs_real[0:p, :] は、IMF0からIMF(p-1)までを指す (Pythonの0-indexed)
            # 高ディップ成分を保持したいので、インデックスの小さいIMFを合計
            summed_imfs_real = np.sum(imfs_real[0:p, :], axis=0) # p個のIMFを保持
            
            # 長さの不一致対策（念のため）
            min_len_real = min(len(freq_slice_real), len(summed_imfs_real))
            reconstructed_real[:min_len_real] = summed_imfs_real[:min_len_real]

        reconstructed_imag = np.zeros_like(freq_slice_imag, dtype=float)
        if imfs_imag.shape[0] >= p:
            summed_imfs_imag = np.sum(imfs_imag[0:p, :], axis=0) # p個のIMFを保持
            
            # 長さの不一致対策（念のため）
            min_len_imag = min(len(freq_slice_imag), len(summed_imfs_imag))
            reconstructed_imag[:min_len_imag] = summed_imfs_imag[:min_len_imag]

        # フィルター処理された時間周波数スライスの合成
        if reconstructed_real.shape != reconstructed_imag.shape:
             print(f"Warning: reconstructed_real shape {reconstructed_real.shape} and reconstructed_imag shape {reconstructed_imag.shape} are different at time_idx {time_idx}.")
             min_len = min(reconstructed_real.shape[0], reconstructed_imag.shape[0])
             filtered_freq_spectrum_slice = reconstructed_real[:min_len] + 1j * reconstructed_imag[:min_len]
        else:
             filtered_freq_spectrum_slice = reconstructed_real + 1j * reconstructed_imag # ここも修正が必要でした
        
        # 空間軸の逆フーリエ変換（f-kドメインからf-xドメインへ）
        filtered_trace_spatial = np.fft.irfft(filtered_freq_spectrum_slice, n=num_distance_traces)
        
        filtered_data[time_idx, :] = filtered_trace_spatial

    return filtered_data


# --- 4. Local Similarityの計算 ---

def calculate_local_similarity_patch(patch1, patch2):
    """
    2つのローカルパッチ間の類似性（正規化相互相関）を計算する。
    論文のAppendix Bに記載の数式群の簡略化した代替として、正規化相互相関を用いる。
    """
    vec1 = patch1.flatten()
    vec2 = patch2.flatten()

    vec1_centered = vec1 - np.mean(vec1)
    vec2_centered = vec2 - np.mean(vec2)

    norm1 = np.linalg.norm(vec1_centered)
    norm2 = np.linalg.norm(vec2_centered)

    if norm1 == 0 or norm2 == 0:
        return 0.0 # いずれかのパッチが定数の場合、類似性は0とする

    similarity = np.dot(vec1_centered, vec2_centered) / (norm1 * norm2)
    return similarity

def calculate_local_similarity_spectrum(data_A, data_B, window_size_local=(5, 5)):
    """
    2つのLPRデータセット間の局所類似性スペクトルを計算する。
    """
    num_time_samples, num_distance_traces = data_A.shape
    similarity_spectrum = np.zeros_like(data_A, dtype=float)

    h_win, w_win = window_size_local
    half_h_win, half_w_win = h_win // 2, w_win // 2

    for i in tqdm(range(num_time_samples), desc="Calculating Local Similarity Spectrum"):
        for j in range(num_distance_traces):
            # ローカル窓を切り出す
            t_start = max(0, i - half_h_win)
            t_end = min(num_time_samples, i + half_h_win + 1)
            d_start = max(0, j - half_w_win)
            d_end = min(num_distance_traces, j + half_w_win + 1)

            patch_A = data_A[t_start:t_end, d_start:d_end]
            patch_B = data_B[t_start:t_end, d_start:d_end]

            # パッチサイズがウィンドウサイズと異なる場合（境界処理）、最小サイズに合わせる
            min_rows = min(patch_A.shape[0], patch_B.shape[0])
            min_cols = min(patch_A.shape[1], patch_B.shape[1])
            patch_A = patch_A[:min_rows, :min_cols]
            patch_B = patch_B[:min_rows, :min_cols]

            if patch_A.size > 0 and patch_B.size > 0:
                similarity_spectrum[i, j] = calculate_local_similarity_patch(patch_A, patch_B)
            else:
                similarity_spectrum[i, j] = 0.0 # 空のパッチの場合

    return similarity_spectrum

# --- 5. スムーズネス促進 (補完アルゴリズム) ---
def apply_smoothness_promotion(spectrum, sigma=1.0):
    """
    類似性スペクトルにガウシアンフィルターを適用して平滑化する。
    論文のS_mの役割を果たすと解釈。
    sigma: ガウス分布の標準偏差（平滑化の度合い）
    """
    return gaussian_filter(spectrum, sigma=sigma)

# --- 6. ソフト閾値関数 (補完アルゴリズム) ---
def apply_soft_thresholding(spectrum, threshold_value):
    """
    類似性スペクトルにソフト閾値関数を適用する。
    """
    thresholded_spectrum = np.copy(spectrum)
    thresholded_spectrum[thresholded_spectrum <= threshold_value] = 0
    return thresholded_spectrum

# --- 7. メイン処理フロー ---
if __name__ == "__main__":
    # 出力フォルダの設定
    output_dir = "/Volumes/SSD_Kanda_SAMSUNG/LPR/rock_extraction_Hu2019"

    # パラメータ設定
    TIME_MAX = 300e-9 # 300ns
    X_MAX = 50.0 # 50m
    DELTA_T = 3e-10 # 3ns
    DELTA_X = 0.05 # 5cm
    NOISE_LEVEL_A = 0.1 # CH-2Aのノイズレベル
    NOISE_LEVEL_B = 0.03 # CH-2Bのノイズレ
    IMF_RETAIN_COUNT = 2 # p=2, IMF1とIMF2を保持
    LOCAL_SIM_WINDOW_SIZE = (7, 7) # ローカル類似性計算のための窓サイズ
    SMOOTHNESS_SIGMA = 1.5 # スムーズネス促進のためのガウシアンフィルターのsigma
    SOFT_THRESHOLD_VALUE = 0.5 # ソフト閾値

    output_dir = os.path.join(output_dir, f'p_{IMF_RETAIN_COUNT}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. テストデータの作成
    print("1. 合成LPRデータの作成...")
    data_A, data_B, true_rock_locs = create_synthetic_lpr_data(
        time=TIME_MAX,
        distance=X_MAX,
        delta_t=DELTA_T,
        delta_x=DELTA_X,
        rock_locations_true=None, # Noneを指定するとデフォルトの岩石位置を使用
        noise_level_A=NOISE_LEVEL_A, # CH-2Aのノイズレベル
        noise_level_B=NOISE_LEVEL_B  # CH-2Bのノイズレベル
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(data_A, aspect='auto', cmap='gray', vmin=-1, vmax=1,
                extent=[0, X_MAX, TIME_MAX * 1e9, 0]) # 時間をns単位で表示
    plt.xlabel('Distance (m)')
    plt.ylabel('Time (ns)')
    plt.title('Synthetic CH-2A Data (Raw)')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(data_B, aspect='auto', cmap='gray', vmin=-1, vmax=1,
               extent=[0, X_MAX, TIME_MAX * 1e9, 0]) # 時間をns単位で表示
    plt.xlabel('Distance (m)')
    plt.ylabel('Time (ns)')
    plt.title('Synthetic CH-2B Data (Raw)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_synthetic_data.png'))
    plt.show()

    # 2. データ前処理
    print("2. データ前処理...")
    processed_data_A = preprocess_lpr_data(data_A)
    processed_data_B = preprocess_lpr_data(data_B)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(processed_data_A, aspect='auto', cmap='gray', vmin=-0.5, vmax=0.5,
                extent=[0, X_MAX, TIME_MAX * 1e9, 0]) # 時間をns単位で表示
    plt.xlabel('Distance (m)')
    plt.ylabel('Time (ns)')
    plt.title('CH-2A Data after Preprocessing')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(processed_data_B, aspect='auto', cmap='gray', vmin=-0.5, vmax=0.5,
               extent=[0, X_MAX, TIME_MAX * 1e9, 0]) # 時間をns単位で表示
    plt.xlabel('Distance (m)')
    plt.ylabel('Time (ns)')
    plt.title('CH-2B Data after Preprocessing')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_processed_data.png'))
    plt.show()

    # 3. f-x EMD Dip Filterの適用
    print(f"3. f-x EMD Dip Filterの適用 (IMF {IMF_RETAIN_COUNT}個保持)...")
    filtered_data_A = fx_emd_dip_filter(processed_data_A, IMF_RETAIN_COUNT)
    filtered_data_B = fx_emd_dip_filter(processed_data_B, IMF_RETAIN_COUNT)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(filtered_data_A, aspect='auto', cmap='gray', vmin=-0.5, vmax=0.5,
                extent=[0, X_MAX, TIME_MAX * 1e9, 0]) # 時間をns単位で表示
    plt.xlabel('Distance (m)')
    plt.ylabel('Time (ns)')
    plt.title(f'CH-2A Data after f-x EMD Dip Filter (p={IMF_RETAIN_COUNT})')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(filtered_data_B, aspect='auto', cmap='gray', vmin=-0.5, vmax=0.5,
               extent=[0, X_MAX, TIME_MAX * 1e9, 0]) # 時間をns単位で表示
    plt.xlabel('Distance (m)')
    plt.ylabel('Time (ns)')
    plt.title(f'CH-2B Data after f-x EMD Dip Filter (p={IMF_RETAIN_COUNT})')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_filtered_data.png'))
    plt.show()

    # 4. Local Similarity Spectrumの計算
    print("4. Local Similarity Spectrumの計算...")
    similarity_spectrum = calculate_local_similarity_spectrum(filtered_data_A, filtered_data_B, window_size_local=LOCAL_SIM_WINDOW_SIZE)

    plt.figure(figsize=(6, 4))
    plt.imshow(similarity_spectrum, aspect='auto', cmap='turbo', vmin=0, vmax=1,
               extent=[0, X_MAX, TIME_MAX * 1e9, 0]) # 時間をns単位で表示
    plt.title(f'Local Similarity Spectrum (Window: {LOCAL_SIM_WINDOW_SIZE})')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_similarity_spectrum.png'))
    plt.show()

    # 5. スムーズネス促進 (ガウシアンフィルター)
    print(f"5. スムーズネス促進の適用 (sigma={SMOOTHNESS_SIGMA})...")
    smoothed_spectrum = apply_smoothness_promotion(similarity_spectrum, sigma=SMOOTHNESS_SIGMA)

    plt.figure(figsize=(6, 4))
    plt.imshow(smoothed_spectrum, aspect='auto', cmap='turbo', vmin=0, vmax=1,
               extent=[0, X_MAX, TIME_MAX * 1e9, 0]) # 時間をns単位で表示
    plt.title(f'Smoothed Local Similarity Spectrum (Sigma: {SMOOTHNESS_SIGMA})')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_smoothed_spectrum.png'))
    plt.show()

    # 6. ソフト閾値関数の適用
    print(f"6. ソフト閾値関数の適用 (Threshold: {SOFT_THRESHOLD_VALUE})...")
    final_spectrum = apply_soft_thresholding(smoothed_spectrum, threshold_value=SOFT_THRESHOLD_VALUE)

    plt.figure(figsize=(6, 4))
    plt.imshow(final_spectrum, aspect='auto', cmap='turbo', vmin=0, vmax=1)
    plt.title(f'Final Local Similarity Spectrum (Threshold: {SOFT_THRESHOLD_VALUE})')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_final_spectrum.png'))
    plt.show()
    
    # 7. Local Maximumの抽出 (岩石位置特定 - 論文の最後のステップ)
    # 論文では「Mute the reflection area. Pick the local maximum and record the corresponding space coordinates.」
    # ここでは、簡略化のため、単純に局所最大値を検出する。
    # 実際には、反射領域のミュート処理が別途必要。
    from scipy.signal import argrelextrema

    print("7. 局所最大値の抽出 (岩石位置特定)...")
    # 局所最大値を検出するための最小距離を設定
    # min_distance は隣接する最大値間の最小距離。これを大きくすると検出される岩石の数が減る。
    # 適切な min_distance はデータと岩石のサイズによる
    min_distance_for_rock = 10
    
    rock_locations_detected = []
    # 各列（距離）でピークを検出
    for col_idx in range(final_spectrum.shape[1]):
        # ピーク検出は1Dで処理し、その後2Dの座標に変換
        # ピークの条件として、隣接するmin_distance内の最大値で、かつ閾値以上の値を持つものを抽出
        peaks_in_col = argrelextrema(final_spectrum[:, col_idx], np.greater, order=min_distance_for_rock)[0]
        
        for row_idx in peaks_in_col:
            if final_spectrum[row_idx, col_idx] > 0: # 閾値処理後も値が残っている場合
                 rock_locations_detected.append((row_idx, col_idx, final_spectrum[row_idx, col_idx]))

    print(f"検出された岩石の数: {len(rock_locations_detected)}")

    plt.figure(figsize=(6, 4))
    plt.imshow(final_spectrum, aspect='auto', cmap='turbo', vmin=0, vmax=1)
    plt.title('Detected Rock Locations on Final Similarity Spectrum')
    plt.colorbar()
    
    # 検出された岩石位置をプロット
    for t, d, val in rock_locations_detected:
        plt.plot(d, t, 'rx', markersize=8, markeredgewidth=2) # 赤いXで検出位置をプロット
    
    # 元の岩石位置をプロット (比較用)
    for t_true, d_true, _ in true_rock_locs:
        plt.plot(d_true, t_true, 'wo', markersize=10, fillstyle='none', markeredgewidth=2) # 白い丸で真の岩石位置をプロット

    plt.xlabel('Distance (Traces)')
    plt.ylabel('Time (Samples)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_detected_rocks.png'))
    plt.show()

    print("\n--- 処理完了 ---")