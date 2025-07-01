import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import emd # emd (EMD-signal) ライブラリを使用
import os
from tqdm import tqdm


# --- Ricker波形生成ヘルパー関数 ---
def ricker_wavelet(t, central_frequency):
    """
    Ricker波形を生成する。
    t: 時間配列 (s)
    central_frequency: 中心周波数 (Hz)
    """
    # Ricker波形の通常は時間ゼロでピークが来るように定義されるが、
    # 地中レーダーでは信号が伝播して戻ってくる時間を考慮するため、
    # その時間スケールを調整する
    
    # ピークが0にくる時間軸
    # Ricker波形の一般的な持続時間はおよそ 2 / central_frequency
    # したがって、tの範囲を-1/fc から 1/fc くらいで定義することが多い
    # ここでは、tをシフトしてピークを任意の位置に持っていく
    
    # 実際には、tは反射波形が生成される各点での時間であり、
    # 波形自体の相対時間を表す必要がある。
    # 通常、Ricker波形の引数 t はピークからの相対時間 t' = t - t_peak
    # ここでは、t_peak を 0 として、波形の形状のみを定義。
    
    # ω = 2 * pi * central_frequency
    # wavelet = (1 - 2 * (pi * central_frequency * t)**2) * np.exp(-(pi * central_frequency * t)**2)
    # 上記は時間t=0でピークが来るRicker波形。
    # GPRの波形としては、通常、正負のピークを持つ波形が使われる。
    # これは二階微分ガウシアンに対応する。

    # ここでは一般的なRicker波形（ピークが1つ）を生成
    # よりGPRに近いのは二階微分ガウシアン（ピークが2つ）
    # GPRの文脈では、"Ricker wavelet" という言葉が二階微分ガウシアンを指すことが多い。
    # ここでは二階微分ガウシアンを実装する。
    # 詳細は下記参照: https://en.wikipedia.org/wiki/Ricker_wavelet

    f_c = central_frequency
    t_prime = t # 時間シフトは適用側で行う

    arg = (np.pi * f_c * t_prime)**2
    # GPRで一般的な二階微分ガウシアン
    wavelet = (1.0 - 2.0 * arg) * np.exp(-arg)
    
    # 必要に応じて、波形の中心を時間ゼロからシフトする（ピークを少し遅らせるなど）
    # 例: wavelet = (1.0 - 2.0 * arg) * np.exp(-arg) * np.sin(2 * np.pi * f_c * t_prime) # これはRickerではないが、GPRで使うこともある
    
    return wavelet


# --- 1. テストデータの作成 ---
def create_synthetic_lpr_data(
    time_max, distance_max, delta_t, delta_x, # time [s], distance [m]
    rock_locations_true, noise_level_A, noise_level_B
):
    """
    合成LPRデータ（CH-2A, CH-2B）を生成する。
    岩石（回折ハイパーボラ）、水平反射、傾斜反射、およびノイズを含む。
    Ricker波形を使用。
    """
    time = np.arange(0, time_max, delta_t) # [s]
    distance = np.arange(0, distance_max, delta_x) # [m]
    data_A = np.zeros((time.size, distance.size))
    data_B = np.zeros((time.size, distance.size))

    # Ricker波形パラメータ
    CENTRAL_FREQUENCY = 500e6 # 500 MHz
    
    # Ricker波形が持つ時間幅の目安（中心周波数の約2倍の逆数）
    # これは波形を生成する際の「相対時間t_ricker」の範囲を設定するのに使う
    # 例えば、-1/f_c から 1/f_c
    ricker_duration = 5.0 / CENTRAL_FREQUENCY # 秒
    ricker_time_array = np.arange(-ricker_duration / 2, ricker_duration / 2, delta_t)
    ricker_waveform = ricker_wavelet(ricker_time_array, CENTRAL_FREQUENCY)

    # 背景ノイズ (ガウシアンランダムノイズ)
    data_A += np.random.normal(0, noise_level_A, size=(time.size, distance.size))
    data_B += np.random.normal(0, noise_level_B, size=(time.size, distance.size))

    # 共通の誘電率
    epsiron_r = 4.5
    v = 3e8 / np.sqrt(epsiron_r) # 波速度

    # シンプルな水平反射 (低ディップ成分)
    reflection_time_top = 50e-9 # [s]
    # 反射波形を重ねる
    for d_idx in range(distance.size):
        # 信号のピークがreflection_time_topに来るように、Ricker波形をシフトして加算
        start_time_idx = int((reflection_time_top - ricker_duration / 2) / delta_t)
        end_time_idx = int((reflection_time_top + ricker_duration / 2) / delta_t)

        for i, ricker_val in enumerate(ricker_waveform):
            current_t_idx = start_time_idx + i
            if 0 <= current_t_idx < time.size:
                data_A[current_t_idx, d_idx] += 1.5 * ricker_val
                data_B[current_t_idx, d_idx] += 1.5 * ricker_val


    # 傾斜した反射成分 (中程度のディップ成分)
    inclined_reflection_params = [
        (20e-9, 0, 1.5e-8, 1.5), # (開始時間 [s], 開始距離 [m], 傾斜 [s/m], 振幅)
        (70e-9, distance_max, -1.0e-8, 1.5) # 逆方向の傾斜 (distance_maxから始まる)
    ]
    
    for start_time, start_dist, slope_time_per_meter, amp in inclined_reflection_params:
        for d_idx, d_val in enumerate(distance):
            # 距離に応じた時間遅延を計算
            time_at_dist = start_time + (d_val - start_dist) * slope_time_per_meter
            
            # 反射波形を重ねる
            start_time_idx = int((time_at_dist - ricker_duration / 2) / delta_t)
            end_time_idx = int((time_at_dist + ricker_duration / 2) / delta_t)

            for i, ricker_val in enumerate(ricker_waveform):
                current_t_idx = start_time_idx + i
                if 0 <= current_t_idx < time.size:
                    data_A[current_t_idx, d_idx] += amp * ricker_val
                    data_B[current_t_idx, d_idx] += amp * ricker_val

    # 岩石（回折ハイパーボラ）を追加
    if rock_locations_true is None:
        rock_locations_true = [
            (20e-9, 3.0, 1.5), (30e-9, 7.5, 1.5),
            (70e-9, 5.0, 1.5), (65e-9, 6.5, 1.5)
        ]
    rock_locations_true = np.array(rock_locations_true)

    for t0, x0, amp in rock_locations_true:
        for x1_idx, x1 in enumerate(distance):
            # ハイパーボラの式を適用 (GPRでは往復のパスが考慮される)
            time_delay_at_x1 = np.sqrt(t0**2 + ((x1 - x0) / (v / 2))**2) 
            
            # 回折波形を重ねる
            start_time_idx = int((time_delay_at_x1 - ricker_duration / 2) / delta_t)
            end_time_idx = int((time_delay_at_x1 + ricker_duration / 2) / delta_t)

            for i, ricker_val in enumerate(ricker_waveform):
                current_t_idx = start_time_idx + i
                if 0 <= current_t_idx < time.size and 0 <= x1_idx < distance.size:
                    # Ricker波形にもガウシアン的な横方向の減衰を適用
                    attenuation = np.exp(-((x1 - x0)**2 / (2 * 1**2))) # 1mの広がり
                    data_A[current_t_idx, x1_idx] += amp * ricker_val * attenuation
                    data_B[current_t_idx, x1_idx] += amp * ricker_val * attenuation

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

    # 新しい波数スペクトル用の出力ディレクトリ
    output_dir_wavenumber_spectrum = os.path.join(output_dir, 'wavenumber_spectrum')
    if not os.path.exists(output_dir_wavenumber_spectrum):
        os.makedirs(output_dir_wavenumber_spectrum)


    # 各時間サンプル（行）について、距離軸方向にFFTを適用
    data_fk_domain = np.fft.rfft(data, axis=1) # 距離軸 (axis=1) に対してFFT

    # 各時間周波数スライス（各行）に対してEMDを適用
    for time_idx in tqdm(range(num_time_samples), desc="Applying f-x EMD Dip Filter (spatial EMD)"):
        freq_slice_real = data_fk_domain[time_idx, :].real
        freq_slice_imag = data_fk_domain[time_idx, :].imag

        # 実数部に対するEMD (空間周波数成分を分離)
        imfs_real = emd.sift.sift(freq_slice_real)
        # 虚数部に対するEMD (空間周波数成分を分離)
        imfs_imag = emd.sift.sift(freq_slice_imag)

        if time_idx == 0:
            print("IMF shapes (Real):", imfs_real.shape)
            print("IMF shapes (Imag):", imfs_imag.shape)
        # フィルター処理された信号の再構築: p個のIMFを保持
        # ここでのIMFは空間的な波数成分に対応し、IMF0, IMF1...が高ディップ成分

        # ==== デバッグ用: IMFの可視化 ==== (この部分はそのまま有効で良いです)
        if time_idx % 100 == 0:
            try:
                max_imfs_to_plot = min(imfs_real.shape[0], 10)
                plt.figure(figsize=(12, 2 * max_imfs_to_plot))
                plt.suptitle(f'IMF Decomposition for Time Freq Slice {time_idx} (Real Part)')
                for imf_n in range(max_imfs_to_plot):
                    plt.subplot(max_imfs_to_plot, 1, imf_n + 1)
                    plt.plot(imfs_real[imf_n * 10, :])
                    plt.title(f'IMF{imf_n * 10} (High Dip/Frequency to Low Dip/Frequency)')
                    plt.ylabel('Amplitude')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(os.path.join(output_dir_EMD_real, f'imf_decomposition_real_{time_idx}.png'))
                plt.close()

                max_imfs_to_plot_imag = min(imfs_imag.shape[0], 10)
                plt.figure(figsize=(12, 2 * max_imfs_to_plot_imag))
                plt.suptitle(f'IMF Decomposition for Time Freq Slice {time_idx} (Imag Part)')
                for imf_n in range(max_imfs_to_plot_imag):
                    plt.subplot(max_imfs_to_plot_imag, 1, imf_n + 1)
                    plt.plot(imfs_imag[imf_n * 10, :])
                    plt.title(f'IMF{imf_n * 10} (High Dip/Frequency to Low Dip/Frequency)')
                    plt.ylabel('Amplitude')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(os.path.join(output_dir_EMD_imag, f'imf_decomposition_imag_{time_idx}.png'))
                plt.close()

                # ==== 各IMFの波数スペクトルをプロット ====
                plt.figure(figsize=(12, 2 * max_imfs_to_plot))
                plt.suptitle(f'Wavenumber Spectrum of IMFs for Time Freq Slice {time_idx} (Real Part)')
                
                # 波数軸の計算 (rfftfreqは正の波数のみを返す)
                # kxは空間軸の波数
                # len(freq_slice_real) は、FFTによって生成された波数スライスの実際の長さに対応
                kx = np.fft.rfftfreq(len(freq_slice_real), d=DELTA_X) 

                for imf_n in range(max_imfs_to_plot):
                    plt.subplot(max_imfs_to_plot, 1, imf_n + 1)
                    
                    # IMFと波数軸の長さを比較し、短い方に合わせてプロット範囲を調整
                    current_imf = imfs_real[imf_n, :]
                    min_len = min(len(kx), len(current_imf))
                    
                    # IMF自体は波数ドメインの成分なので、その振幅をプロット
                    plt.plot(kx[:min_len], np.abs(current_imf[:min_len])) # ここを修正
                    plt.title(f'IMF{imf_n} Wavenumber Spectrum')
                    plt.xlabel('Wavenumber (cycles/m)')
                    plt.ylabel('Amplitude')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(os.path.join(output_dir_wavenumber_spectrum, f'imf_wavenumber_spectrum_real_{time_idx}.png'))
                plt.close()

            except Exception as e:
                print(f"Error plotting IMFs at time_idx {time_idx}: {e}")


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
    TIME_MAX = 100e-9 # 100ns
    X_MAX = 10.0 # 10m
    DELTA_T = 2e-10 # 2ns
    DELTA_X = 0.02 # 2cm
    NOISE_LEVEL_A = 0.1 # CH-2Aのノイズレベル
    NOISE_LEVEL_B = 0.03 # CH-2Bのノイズレベル
    IMF_RETAIN_COUNT = int(input("IMFの保持数を入力してください (例: 2): ").strip()) # p=2, IMF1とIMF2を保持
    LOCAL_SIM_WINDOW_SIZE = (7, 7) # ローカル類似性計算のための窓サイズ
    SMOOTHNESS_SIGMA = 1.5 # スムーズネス促進のためのガウシアンフィルターのsigma
    SOFT_THRESHOLD_VALUE = 0.5 # ソフト閾値

    output_dir = os.path.join(output_dir, f'p_{IMF_RETAIN_COUNT}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. テストデータの作成
    print("1. 合成LPRデータの作成...")
    data_A, data_B, true_rock_locs = create_synthetic_lpr_data(
        time_max=TIME_MAX,
        distance_max=X_MAX,
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

    # テストデータのf-kドメイン変換
    fk_data_A = np.fft.fft2(data_A) # 時間と距離の両方に対してFFT
    fk_data_B = np.fft.fft2(data_B)
    fk_data_A = np.fft.fftshift(fk_data_A) # ゼロ周波数を中心に移動
    fk_data_B = np.fft.fftshift(fk_data_B)
    fk_data_A_power = 20 * np.log(np.abs(fk_data_A)) # ゼロ除算を避けるために小さな値を加える
    fk_data_B_power = 20 * np.log(np.abs(fk_data_B))
    # fkドメインの周波数軸と距離軸を計算
    time = np.arange(0, TIME_MAX, DELTA_T)
    freq_axis = np.fft.fftfreq(len(time), d=DELTA_T) # 時間軸の周波数
    freq_axis = np.fft.fftshift(freq_axis) # ゼロ周波数を中心に移動
    x = np.arange(0, X_MAX, DELTA_X)
    wavenumber_axis = np.fft.fftfreq(len(x), d=DELTA_X) # 距離軸の波数
    wavenumber_axis = np.fft.fftshift(wavenumber_axis) #
    # fkドメインの可視化
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(fk_data_A_power, aspect='auto', cmap='viridis',
               extent=[wavenumber_axis.min(), wavenumber_axis.max(), freq_axis.min(), freq_axis.max()])
    plt.xlabel('Wavenumber (/m)')
    plt.ylabel('Frequency (Hz)')
    plt.title('CH-2A Data in f-k Domain')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(fk_data_B_power, aspect='auto', cmap='viridis',
               extent=[wavenumber_axis.min(), wavenumber_axis.max(), freq_axis.min(), freq_axis.max()])
    plt.xlabel('Wavenumber (/m)')
    plt.ylabel('Frequency (Hz)')
    plt.title('CH-2B Data in f-k Domain')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_fk_data.png'))
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

    # f-kドメインでのフィルタリング結果の可視化
    fk_filtered_data_A = np.fft.fft2(filtered_data_A) # 時間と距離の両方に対してFFT
    fk_filtered_data_B = np.fft.fft2(filtered_data_B)
    fk_filtered_data_A = np.fft.fftshift(fk_filtered_data_A) # ゼロ周波数を中心に移動
    fk_filtered_data_B = np.fft.fftshift(fk_filtered_data_B)
    fk_filtered_data_A_power = 20 * np.log(np.abs(fk_filtered_data_A)) # ゼロ除算を避けるために小さな値を加える
    fk_filtered_data_B_power = 20 * np.log(np.abs(fk_filtered_data_B))
    # fkドメインの周波数軸と距離軸を計算
    time = np.arange(0, TIME_MAX, DELTA_T)
    freq_axis = np.fft.fftfreq(len(time), d=DELTA_T) # 時間軸の周波数
    freq_axis = np.fft.fftshift(freq_axis) # ゼロ周波数を中心に移動
    x = np.arange(0, X_MAX, DELTA_X)
    wavenumber_axis = np.fft.fftfreq(len(x), d=DELTA_X) # 距離軸の波数
    wavenumber_axis = np.fft.fftshift(wavenumber_axis) #
    # fkドメインの可視化
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(fk_filtered_data_A_power, aspect='auto', cmap='viridis',
               extent=[wavenumber_axis.min(), wavenumber_axis.max(), freq_axis.min(), freq_axis.max()])
    plt.xlabel('Wavenumber (/m)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'CH-2A Data in f-k Domain (Filtered, p={IMF_RETAIN_COUNT})')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(fk_filtered_data_B_power, aspect='auto', cmap='viridis',
               extent=[wavenumber_axis.min(), wavenumber_axis.max(), freq_axis.min(), freq_axis.max()])
    plt.xlabel('Wavenumber (/m)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'CH-2B Data in f-k Domain (Filtered, p={ IMF_RETAIN_COUNT})')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_fk_filtered_data.png'))
    plt.show()
    """
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
        plt.plot(d * DELTA_X, t * DELTA_T * 1e9, 'rx', markersize=8, markeredgewidth=2) # 赤いXで検出位置をプロット
    
    # 元の岩石位置をプロット (比較用)
    for t_true, d_true, _ in true_rock_locs:
        plt.plot(d_true, t_true * 1e9, 'wo', markersize=10, fillstyle='none', markeredgewidth=2) # 白い丸で真の岩石位置をプロット

    plt.xlabel('Distance (m)')
    plt.ylabel('Time (ns)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_detected_rocks.png'))
    plt.show()
    """
    print("\n--- 処理完了 ---")