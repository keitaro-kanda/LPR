import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PyHHT.emd import EMD # EMDの実装にPyHHTを使用

# --- 1. テストデータの作成 ---
def create_synthetic_lpr_data(
    num_time_samples=150, num_distance_traces=90, 
    rock_locations_true=None, noise_level_A=0.1, noise_level_B=0.03
):
    """
    合成LPRデータ（CH-2A, CH-2B）を生成する。
    岩石（回折ハイパーボラ）とノイズを含む。
    """
    time = np.arange(num_time_samples)
    distance = np.arange(num_distance_traces)
    data_A = np.zeros((num_time_samples, num_distance_traces))
    data_B = np.zeros((num_time_samples, num_distance_traces))

    # 背景ノイズ (ガウシアンランダムノイズ)
    data_A += np.random.normal(0, noise_level_A, size=(num_time_samples, num_distance_traces))
    data_B += np.random.normal(0, noise_level_B, size=(num_time_samples, num_distance_traces))

    # シンプルな水平反射 (低ディップ成分)
    # 反射位置を時間50nsあたりに設定
    reflection_time = 50
    for d in range(num_distance_traces):
        if 0 <= reflection_time < num_time_samples:
            data_A[reflection_time, d] += 5.0
            data_B[reflection_time, d] += 5.0

    # 岩石（回折ハイパーボラ）を追加
    if rock_locations_true is None:
        # デフォルトの岩石位置 (時間, 距離, 強度)
        rock_locations_true = [
            (20, 10, 1.0), (35, 30, 1.2), (60, 50, 1.5), (80, 70, 1.0)
        ]

    for t_rock, d_rock, amp in rock_locations_true:
        for d_offset in range(num_distance_traces):
            # ハイパーボラの式: t = sqrt(t0^2 + (x/v)^2)
            # ここでは簡略化のため、頂点からの距離に比例して時間が遅延するモデル
            # LPRの波速度は真空中の光速 c / sqrt(誘電率)。ここでは仮の値を使用。
            velocity_factor = 0.5 # 仮の速度係数、小さいほどハイパーボラが急になる
            time_delay = int(np.sqrt((d_offset - d_rock)**2 * velocity_factor + t_rock**2))
            
            if 0 <= time_delay < num_time_samples:
                data_A[time_delay, d_offset] += amp * np.exp(-((d_offset - d_rock)**2 / (2 * 5**2))) # ガウシアン的な減衰
                data_B[time_delay, d_offset] += amp * np.exp(-((d_offset - d_rock)**2 / (2 * 5**2)))

    return data_A, data_B, rock_locations_true

# --- 2. データ前処理 (論文のPipelineの一部を模倣) ---
def preprocess_lpr_data(data):
    """
    LPRデータの前処理パイプラインの簡略化された実装。
    ここでは、バンドパスフィルターと背景除去を模倣する。
    """
    # 実際のLPRデータ処理は複雑なため、ここでは簡単なノイズ除去と平滑化のみを適用
    # 例: ガウシアンフィルターで軽く平滑化
    processed_data = gaussian_filter(data, sigma=0.5) 
    
    # 背景除去 (ここでは簡単な平均値除去を模倣)
    # processed_data = processed_data - np.mean(processed_data, axis=1, keepdims=True)
    return processed_data

# --- 3. f-x EMD Dip Filter ---
def fx_emd_dip_filter(data, p=2):
    """
    f-x EMDに基づく低域通過ディップフィルター。
    各トレースの周波数スライスにEMDを適用し、IMF1とIMF2を除去する。
    p: 除去するIMFの数（p=2はIMF1とIMF2を除去）
    """
    num_time_samples, num_distance_traces = data.shape
    filtered_data = np.zeros_like(data, dtype=float) # float型で初期化

    emd = EMD()

    for i in range(num_distance_traces):
        trace = data[:, i]

        # 1Dフーリエ変換（時間方向）
        # RFFTは実数入力のための最適化されたFFT
        freq_spectrum = np.fft.rfft(trace) 
        
        # 各周波数スライス（ここでは周波数成分）を実数部と虚数部に分離
        # PyHHTのEMDは実数入力を前提とする
        real_part = freq_spectrum.real
        imag_part = freq_spectrum.imag

        # 実数部に対するEMD
        imfs_real = emd.sift(real_part)
        
        # 虚数部に対するEMD
        imfs_imag = emd.sift(imag_part)

        # フィルター処理された信号の再構築: IMF1とIMF2を除去 (p=2)
        # imfs_realとimfs_imagはリストであり、各要素がIMFのndarray
        # 最初のp個のIMFを除去し、残りを合計
        
        filtered_real = np.zeros_like(real_part)
        if len(imfs_real) > p:
            # imfs_real[p:] はp番目以降のIMF（インデックスは0から始まる）
            for imf_comp in imfs_real[p:]:
                filtered_real += imf_comp
        # else: 全てのIMFがp以下の場合、全て除去されるため filtered_real は0のまま

        filtered_imag = np.zeros_like(imag_part)
        if len(imfs_imag) > p:
            for imf_comp in imfs_imag[p:]:
                filtered_imag += imf_comp
        # else: 全てのIMFがp以下の場合、全て除去されるため filtered_imag は0のまま

        # フィルター処理された周波数スライスの合成
        filtered_freq_spectrum = filtered_real + 1j * filtered_imag
        
        # 1D逆フーリエ変換（時間方向）
        # irfftは実数出力のための最適化されたIFFT
        filtered_trace = np.fft.irfft(filtered_freq_spectrum, n=num_time_samples) 
        
        filtered_data[:, i] = filtered_trace

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

    for i in range(num_time_samples):
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
    # パラメータ設定
    NUM_TIME_SAMPLES = 150
    NUM_DISTANCE_TRACES = 90
    IMF_REMOVE_COUNT = 2 # p=2, IMF1とIMF2を除去
    LOCAL_SIM_WINDOW_SIZE = (7, 7) # ローカル類似性計算のための窓サイズ
    SMOOTHNESS_SIGMA = 1.5 # スムーズネス促進のためのガウシアンフィルターのsigma
    SOFT_THRESHOLD_VALUE = 0.5 # ソフト閾値

    # 1. テストデータの作成
    print("1. 合成LPRデータの作成...")
    data_A, data_B, true_rock_locs = create_synthetic_lpr_data(
        num_time_samples=NUM_TIME_SAMPLES,
        num_distance_traces=NUM_DISTANCE_TRACES
    )

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(data_A, aspect='auto', cmap='gray', vmin=-1, vmax=1)
    plt.title('Synthetic CH-2A Data (Raw)')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(data_B, aspect='auto', cmap='gray', vmin=-1, vmax=1)
    plt.title('Synthetic CH-2B Data (Raw)')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # 2. データ前処理
    print("2. データ前処理...")
    processed_data_A = preprocess_lpr_data(data_A)
    processed_data_B = preprocess_lpr_data(data_B)

    # 3. f-x EMD Dip Filterの適用
    print(f"3. f-x EMD Dip Filterの適用 (IMF {IMF_REMOVE_COUNT}個除去)...")
    filtered_data_A = fx_emd_dip_filter(processed_data_A, p=IMF_REMOVE_COUNT)
    filtered_data_B = fx_emd_dip_filter(processed_data_B, p=IMF_REMOVE_COUNT)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(filtered_data_A, aspect='auto', cmap='gray', vmin=-0.5, vmax=0.5)
    plt.title(f'CH-2A Data after f-x EMD Dip Filter (p={IMF_REMOVE_COUNT})')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(filtered_data_B, aspect='auto', cmap='gray', vmin=-0.5, vmax=0.5)
    plt.title(f'CH-2B Data after f-x EMD Dip Filter (p={IMF_REMOVE_COUNT})')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # 4. Local Similarity Spectrumの計算
    print("4. Local Similarity Spectrumの計算...")
    similarity_spectrum = calculate_local_similarity_spectrum(filtered_data_A, filtered_data_B, window_size_local=LOCAL_SIM_WINDOW_SIZE)

    plt.figure(figsize=(6, 5))
    plt.imshow(similarity_spectrum, aspect='auto', cmap='jet', vmin=0, vmax=1)
    plt.title(f'Local Similarity Spectrum (Window: {LOCAL_SIM_WINDOW_SIZE})')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # 5. スムーズネス促進 (ガウシアンフィルター)
    print(f"5. スムーズネス促進の適用 (sigma={SMOOTHNESS_SIGMA})...")
    smoothed_spectrum = apply_smoothness_promotion(similarity_spectrum, sigma=SMOOTHNESS_SIGMA)

    plt.figure(figsize=(6, 5))
    plt.imshow(smoothed_spectrum, aspect='auto', cmap='jet', vmin=0, vmax=1)
    plt.title(f'Smoothed Local Similarity Spectrum (Sigma: {SMOOTHNESS_SIGMA})')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # 6. ソフト閾値関数の適用
    print(f"6. ソフト閾値関数の適用 (Threshold: {SOFT_THRESHOLD_VALUE})...")
    final_spectrum = apply_soft_thresholding(smoothed_spectrum, threshold_value=SOFT_THRESHOLD_VALUE)

    plt.figure(figsize=(6, 5))
    plt.imshow(final_spectrum, aspect='auto', cmap='jet', vmin=0, vmax=1)
    plt.title(f'Final Local Similarity Spectrum (Threshold: {SOFT_THRESHOLD_VALUE})')
    plt.colorbar()
    plt.tight_layout()
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

    plt.figure(figsize=(8, 6))
    plt.imshow(final_spectrum, aspect='auto', cmap='jet', vmin=0, vmax=1)
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
    plt.show()

    print("\n--- 処理完了 ---")