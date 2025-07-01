import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
from tqdm import tqdm

def fk_transform_full(data, dt, dx):
    """
    全体データのf-k変換を実行
    
    Parameters:
    -----------
    data : np.ndarray
        入力B-scanデータ (time x traces)
    dt : float
        時間サンプリング間隔 [s]
    dx : float
        トレース間隔 [m]
    
    Returns:
    --------
    KK_shifted : np.ndarray
        f-k変換結果 (周波数中心シフト済み)
    f_MHz : np.ndarray
        周波数軸 [MHz]
    K : np.ndarray
        波数軸 [1/m]
    """
    
    # Handle NaN values
    data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize data
    data_max = np.nanmax(np.abs(data_clean))
    if data_max == 0 or np.isnan(data_max):
        data_max = 1.0
    data_norm = data_clean / data_max
    
    # Create frequency and wavenumber vectors
    N_time = data.shape[0]
    N_traces = data.shape[1]
    
    f = np.fft.fftfreq(N_time, dt)
    f = np.fft.fftshift(f)
    f_MHz = f * 1e-6  # Convert to MHz
    
    K = np.fft.fftfreq(N_traces, dx)
    K = np.fft.fftshift(K)
    
    # 2D FFT
    KK = np.fft.fft2(data_norm)
    KK_shifted = np.fft.fftshift(KK)
    
    return KK_shifted, f_MHz, K

def fk_transform_windowed(data, dt, dx, window_time, window_trace):
    """
    窓関数を使用した区切りf-k変換を実行
    
    Parameters:
    -----------
    data : np.ndarray
        入力B-scanデータ (time x traces)
    dt : float
        時間サンプリング間隔 [s]
    dx : float
        トレース間隔 [m]
    window_time : int
        時間方向の窓サイズ [サンプル数]
    window_trace : int
        トレース方向の窓サイズ [トレース数]
    
    Returns:
    --------
    results : list
        各窓の変換結果 [{'data': KK_shifted, 'f_MHz': f_MHz, 'K': K, 'pos': (t_start, x_start)}]
    """
    print(f'区切りf-k変換を実行中... (窓サイズ: {window_time} x {window_trace})')
    
    N_time, N_traces = data.shape
    results = []
    
    # 窓の数を計算
    n_windows_time = N_time // window_time
    n_windows_trace = N_traces // window_trace
    
    print(f'時間方向窓数: {n_windows_time}, トレース方向窓数: {n_windows_trace}')
    print(f'総窓数: {n_windows_time * n_windows_trace}')
    
    # 各窓での処理
    for i in tqdm(range(n_windows_time), desc="時間窓"):
        for j in range(n_windows_trace):
            # 窓の範囲を計算
            t_start = i * window_time
            t_end = min(t_start + window_time, N_time)
            x_start = j * window_trace
            x_end = min(x_start + window_trace, N_traces)
            
            # 窓データを抽出
            window_data = data[t_start:t_end, x_start:x_end]
            
            # 窓が小さすぎる場合はスキップ
            if window_data.shape[0] < 10 or window_data.shape[1] < 10:
                continue
            
            # Check for NaN values in window data
            if np.any(np.isnan(window_data)):
                print(f"窓 ({i}, {j}) にNaN値が含まれています - 処理を続行")
            
            # Check if window data has any valid values
            if np.all(np.isnan(window_data)) or np.nanmax(np.abs(window_data)) == 0:
                print(f"窓 ({i}, {j}) に有効なデータがありません - スキップ")
                continue
            
            # この窓のf-k変換
            try:
                KK_shifted, f_MHz, K = fk_transform_full(window_data, dt, dx)
                
                results.append({
                    'data': KK_shifted,
                    'f_MHz': f_MHz,
                    'K': K,
                    'pos': (t_start, x_start),
                    'shape': window_data.shape,
                    'window_idx': (i, j),
                    'window_data': window_data  # B-scanデータも保存
                })
            except Exception as e:
                print(f"窓 ({i}, {j}) でエラー: {e}")
                continue
    
    return results

def apply_fk_filter(KK_shifted, f_MHz, K, filter_type, **filter_params):
    """
    f-kフィルターを適用する関数
    
    Parameters:
    -----------
    KK_shifted : np.ndarray
        f-k変換データ
    f_MHz : np.ndarray
        周波数軸 [MHz]
    K : np.ndarray
        波数軸 [1/m]
    filter_type : str
        フィルタータイプ ('lowpass', 'highpass', 'bandpass', 'triangular' etc.)
    **filter_params : dict
        フィルターパラメータ
    
    Returns:
    --------
    KK_filtered : np.ndarray
        フィルター適用後のf-kデータ
    """
    if filter_type == 'triangular':
        # 三角形フィルター: (f, k) = (250, 0), (750, 4), (750, -4)を頂点とする領域を保持
        return apply_triangular_filter(KK_shifted, f_MHz, K)
    elif filter_type == 'rectangular':
        # 四角形フィルター: (f, k) = (250, -4), (750, -4), (750, 4), (250, 4)を頂点とする領域を保持
        return apply_rectangular_filter(KK_shifted, f_MHz, K)
    else:
        # その他のフィルタータイプ（未実装）
        print(f"フィルタータイプ '{filter_type}' は未実装です。元のデータを返します。")
        return KK_shifted

def apply_triangular_filter(KK_shifted, f_MHz, K):
    """
    三角形領域フィルターを適用（f=0を軸とする対称領域）
    正の周波数領域: (f, k) = (250, 0), (750, 4), (750, -4) [MHz, 1/m]
    負の周波数領域: (f, k) = (-250, 0), (-750, 4), (-750, -4) [MHz, 1/m]
    
    Parameters:
    -----------
    KK_shifted : np.ndarray
        f-k変換データ
    f_MHz : np.ndarray
        周波数軸 [MHz]
    K : np.ndarray
        波数軸 [1/m]
    
    Returns:
    --------
    KK_filtered : np.ndarray
        フィルター適用後のf-kデータ
    """
    # 正の周波数領域の三角形頂点 [MHz, 1/m]
    vertices_pos = np.array([
        [250, 0],    # 頂点A
        [750, 4],    # 頂点B
        [750, -4]    # 頂点C
    ])
    
    # 負の周波数領域の三角形頂点（f=0を軸とする対称）
    vertices_neg = np.array([
        [-250, 0],   # 頂点A'
        [-750, 4],   # 頂点B'
        [-750, -4]   # 頂点C'
    ])
    
    print(f"三角形フィルター適用（対称領域）:")
    print(f"  正の周波数領域:")
    print(f"    A: (f={vertices_pos[0,0]} MHz, k={vertices_pos[0,1]} 1/m)")
    print(f"    B: (f={vertices_pos[1,0]} MHz, k={vertices_pos[1,1]} 1/m)")
    print(f"    C: (f={vertices_pos[2,0]} MHz, k={vertices_pos[2,1]} 1/m)")
    print(f"  負の周波数領域:")
    print(f"    A': (f={vertices_neg[0,0]} MHz, k={vertices_neg[0,1]} 1/m)")
    print(f"    B': (f={vertices_neg[1,0]} MHz, k={vertices_neg[1,1]} 1/m)")
    print(f"    C': (f={vertices_neg[2,0]} MHz, k={vertices_neg[2,1]} 1/m)")
    
    # 両方の三角形領域のマスクを作成
    mask_pos = create_triangular_mask(f_MHz, K, vertices_pos)
    mask_neg = create_triangular_mask(f_MHz, K, vertices_neg)
    
    # 統合マスク（どちらかの三角形内であればTrue）
    mask = mask_pos | mask_neg
    
    # マスクを適用
    KK_filtered = KK_shifted.copy()
    KK_filtered[~mask] = 0  # 三角形外の領域をゼロにする
    
    # フィルター適用結果の統計
    total_points = KK_shifted.size
    preserved_points = np.sum(mask)
    preserved_ratio = preserved_points / total_points * 100
    
    print(f"  フィルター適用結果:")
    print(f"    保持された点数: {preserved_points} / {total_points} ({preserved_ratio:.2f}%)")
    
    return KK_filtered

def apply_rectangular_filter(KK_shifted, f_MHz, K):
    """
    四角形領域フィルターを適用（f=0を軸とする対称領域）
    正の周波数領域: (f, k) = (250, -4), (750, -4), (750, 4), (250, 4) [MHz, 1/m]
    負の周波数領域: (f, k) = (-250, -4), (-750, -4), (-750, 4), (-250, 4) [MHz, 1/m]
    
    Parameters:
    -----------
    KK_shifted : np.ndarray
        f-k変換データ
    f_MHz : np.ndarray
        周波数軸 [MHz]
    K : np.ndarray
        波数軸 [1/m]
    
    Returns:
    --------
    KK_filtered : np.ndarray
        フィルター適用後のf-kデータ
    """
    # 正の周波数領域の四角形頂点 [MHz, 1/m]
    vertices_pos = np.array([
        [250, -4],   # 頂点A (左下)
        [750, -4],   # 頂点B (右下)
        [750, 4],    # 頂点C (右上)
        [250, 4]     # 頂点D (左上)
    ])
    
    # 負の周波数領域の四角形頂点（f=0を軸とする対称）
    vertices_neg = np.array([
        [-250, -4],  # 頂点A' (右下)
        [-750, -4],  # 頂点B' (左下)
        [-750, 4],   # 頂点C' (左上)
        [-250, 4]    # 頂点D' (右上)
    ])
    
    print(f"四角形フィルター適用（対称領域）:")
    print(f"  正の周波数領域:")
    print(f"    A: (f={vertices_pos[0,0]} MHz, k={vertices_pos[0,1]} 1/m)")
    print(f"    B: (f={vertices_pos[1,0]} MHz, k={vertices_pos[1,1]} 1/m)")
    print(f"    C: (f={vertices_pos[2,0]} MHz, k={vertices_pos[2,1]} 1/m)")
    print(f"    D: (f={vertices_pos[3,0]} MHz, k={vertices_pos[3,1]} 1/m)")
    print(f"  負の周波数領域:")
    print(f"    A': (f={vertices_neg[0,0]} MHz, k={vertices_neg[0,1]} 1/m)")
    print(f"    B': (f={vertices_neg[1,0]} MHz, k={vertices_neg[1,1]} 1/m)")
    print(f"    C': (f={vertices_neg[2,0]} MHz, k={vertices_neg[2,1]} 1/m)")
    print(f"    D': (f={vertices_neg[3,0]} MHz, k={vertices_neg[3,1]} 1/m)")
    
    # 両方の四角形領域のマスクを作成
    mask_pos = create_rectangular_mask(f_MHz, K, vertices_pos)
    mask_neg = create_rectangular_mask(f_MHz, K, vertices_neg)
    
    # 統合マスク（どちらかの四角形内であればTrue）
    mask = mask_pos | mask_neg
    
    # マスクを適用
    KK_filtered = KK_shifted.copy()
    KK_filtered[~mask] = 0  # 四角形外の領域をゼロにする
    
    # フィルター適用結果の統計
    total_points = KK_shifted.size
    preserved_points = np.sum(mask)
    preserved_ratio = preserved_points / total_points * 100
    
    print(f"  フィルター適用結果:")
    print(f"    保持された点数: {preserved_points} / {total_points} ({preserved_ratio:.2f}%)")
    
    return KK_filtered

def create_rectangular_mask(f_MHz, K, vertices):
    """
    四角形領域のマスクを作成
    
    Parameters:
    -----------
    f_MHz : np.ndarray
        周波数軸 [MHz]
    K : np.ndarray
        波数軸 [1/m]
    vertices : np.ndarray
        四角形の頂点 [[f1, k1], [f2, k2], [f3, k3], [f4, k4]]
    
    Returns:
    --------
    mask : np.ndarray
        四角形内部でTrueとなるマスク
    """
    # メッシュグリッドを作成
    K_grid, F_grid = np.meshgrid(K, f_MHz)
    
    # 各点が四角形内部にあるかを判定
    mask = point_in_rectangle(F_grid, K_grid, vertices)
    
    return mask

def point_in_rectangle(f_points, k_points, vertices):
    """
    点が四角形内部にあるかを判定
    四角形が軸に平行な長方形の場合の最適化された判定
    
    Parameters:
    -----------
    f_points : np.ndarray
        周波数座標
    k_points : np.ndarray
        波数座標
    vertices : np.ndarray
        四角形の頂点 [[f1, k1], [f2, k2], [f3, k3], [f4, k4]]
    
    Returns:
    --------
    inside : np.ndarray
        四角形内部でTrueとなるブール配列
    """
    # 軸に平行な長方形の場合、min/maxで簡単に判定可能
    f_min = np.min(vertices[:, 0])  # 最小周波数
    f_max = np.max(vertices[:, 0])  # 最大周波数
    k_min = np.min(vertices[:, 1])  # 最小波数
    k_max = np.max(vertices[:, 1])  # 最大波数
    
    # 四角形内部判定
    inside = (f_points >= f_min) & (f_points <= f_max) & \
             (k_points >= k_min) & (k_points <= k_max)
    
    return inside

def create_triangular_mask(f_MHz, K, vertices):
    """
    三角形領域のマスクを作成
    
    Parameters:
    -----------
    f_MHz : np.ndarray
        周波数軸 [MHz]
    K : np.ndarray
        波数軸 [1/m]
    vertices : np.ndarray
        三角形の頂点 [[f1, k1], [f2, k2], [f3, k3]]
    
    Returns:
    --------
    mask : np.ndarray
        三角形内部でTrueとなるマスク
    """
    # メッシュグリッドを作成
    K_grid, F_grid = np.meshgrid(K, f_MHz)
    
    # 各点が三角形内部にあるかを判定
    mask = point_in_triangle(F_grid, K_grid, vertices)
    
    return mask

def point_in_triangle(f_points, k_points, vertices):
    """
    点が三角形内部にあるかを重心座標を使って判定
    
    Parameters:
    -----------
    f_points : np.ndarray
        周波数座標
    k_points : np.ndarray
        波数座標
    vertices : np.ndarray
        三角形の頂点 [[f1, k1], [f2, k2], [f3, k3]]
    
    Returns:
    --------
    inside : np.ndarray
        三角形内部でTrueとなるブール配列
    """
    # 三角形の頂点
    A = vertices[0]  # [f1, k1]
    B = vertices[1]  # [f2, k2] 
    C = vertices[2]  # [f3, k3]
    
    # 重心座標での判定
    # P = uA + vB + wC, u + v + w = 1
    # P が三角形内部 <=> u >= 0, v >= 0, w >= 0
    
    # ベクトル計算
    v0 = C - A  # AC
    v1 = B - A  # AB
    v2_f = f_points - A[0]  # Pf - Af
    v2_k = k_points - A[1]  # Pk - Ak
    
    # 内積計算
    dot00 = np.dot(v0, v0)  # AC・AC
    dot01 = np.dot(v0, v1)  # AC・AB
    dot11 = np.dot(v1, v1)  # AB・AB
    dot02 = v0[0] * v2_f + v0[1] * v2_k  # AC・AP
    dot12 = v1[0] * v2_f + v1[1] * v2_k  # AB・AP
    
    # 重心座標を計算
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    # 三角形内部判定
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    
    return inside

def inverse_fk_transform(KK_filtered, data_shape):
    """
    f-k逆変換を実行してB-scanデータに戻す
    
    Parameters:
    -----------
    KK_filtered : np.ndarray
        フィルター適用後のf-kデータ
    data_shape : tuple
        元データの形状 (time, traces)
    
    Returns:
    --------
    filtered_bscan : np.ndarray
        フィルター適用後のB-scanデータ
    """
    # f-k逆変換
    KK_ishifted = np.fft.ifftshift(KK_filtered)
    filtered_bscan = np.fft.ifft2(KK_ishifted)
    
    # 実部のみ取得（虚部は数値誤差）
    filtered_bscan = np.real(filtered_bscan)
    
    # 元の形状にリサイズ（必要に応じて）
    if filtered_bscan.shape != data_shape:
        filtered_bscan = filtered_bscan[:data_shape[0], :data_shape[1]]
    
    return filtered_bscan

def plot_fk_filter_comparison(original_data, filtered_data, KK_original, KK_filtered, 
                            f_MHz, K, output_path, filter_info=""):
    """
    フィルター適用前後の比較プロット
    
    Parameters:
    -----------
    original_data : np.ndarray
        元のB-scanデータ
    filtered_data : np.ndarray
        フィルター適用後のB-scanデータ
    KK_original : np.ndarray
        元のf-kデータ
    KK_filtered : np.ndarray
        フィルター適用後のf-kデータ
    f_MHz : np.ndarray
        周波数軸 [MHz]
    K : np.ndarray
        波数軸 [1/m]
    output_path : str
        出力ファイルパス
    filter_info : str
        フィルター情報（タイトル用）
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # B-scanデータの共通カラーバー範囲
    data_max = max(np.nanmax(np.abs(original_data)), np.nanmax(np.abs(filtered_data)))
    vmin_bscan = -data_max / 10
    vmax_bscan = data_max / 10
    
    # f-kデータの対数スケール計算
    KK_orig_log = 20 * np.log10(np.abs(KK_original) + 1e-10)
    KK_filt_log = 20 * np.log10(np.abs(KK_filtered) + 1e-10)
    
    # 左上: 元のB-scan
    im1 = axes[0,0].imshow(original_data, aspect='auto', cmap='viridis',
                          vmin=vmin_bscan, vmax=vmax_bscan)
    axes[0,0].set_title('Original B-scan', fontsize=16)
    axes[0,0].set_xlabel('Trace', fontsize=14)
    axes[0,0].set_ylabel('Time [samples]', fontsize=14)
    
    # 右上: フィルター適用後のB-scan
    im2 = axes[0,1].imshow(filtered_data, aspect='auto', cmap='viridis',
                          vmin=-np.amax(np.abs(filtered_data))/10, vmax=np.amax(np.abs(filtered_data))/10)
    axes[0,1].set_title(f'Filtered B-scan {filter_info}', fontsize=16)
    axes[0,1].set_xlabel('Trace', fontsize=14)
    axes[0,1].set_ylabel('Time [samples]', fontsize=14)
    
    # 左下: 元のf-k
    im3 = axes[1,0].imshow(KK_orig_log, aspect='auto', cmap='turbo', origin='lower',
                          extent=(K.min(), K.max(), f_MHz.min(), f_MHz.max()),
                          vmin=np.percentile(KK_orig_log, 5),
                          vmax=np.percentile(KK_orig_log, 99))
    axes[1,0].set_title('Original F-K Transform', fontsize=16)
    axes[1,0].set_xlabel('Wavenumber [1/m]', fontsize=14)
    axes[1,0].set_ylabel('Frequency [MHz]', fontsize=14)
    
    # 右下: フィルター適用後のf-k
    im4 = axes[1,1].imshow(KK_filt_log, aspect='auto', cmap='turbo', origin='lower',
                          extent=(K.min(), K.max(), f_MHz.min(), f_MHz.max()),
                          vmin=np.percentile(KK_filt_log, 5),
                          vmax=np.percentile(KK_filt_log, 99))
    axes[1,1].set_title(f'Filtered F-K Transform {filter_info}', fontsize=16)
    axes[1,1].set_xlabel('Wavenumber [1/m]', fontsize=14)
    axes[1,1].set_ylabel('Frequency [MHz]', fontsize=14)
    
    # カラーバー追加
    for i, im in enumerate([im1, im2]):
        divider = axgrid1.make_axes_locatable(axes[0,i])
        cax = divider.append_axes('right', size='3%', pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Amplitude', fontsize=12)
    
    for i, im in enumerate([im3, im4]):
        divider = axgrid1.make_axes_locatable(axes[1,i])
        cax = divider.append_axes('right', size='3%', pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Amplitude [dB]', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 入力項目
    data_path = input("データファイルのパスを入力してください: ").strip()
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"ファイル {data_path} が存在しません。")
    
    # 計算モード選択
    print("\n計算モードを選択してください:")
    print("1. 全体f-kフィルタリング")
    print("2. 区切りf-kフィルタリング")
    print("3. 両方実行")
    
    mode = input("モード番号を入力 (1/2/3): ").strip()
    if mode not in ['1', '2', '3']:
        raise ValueError("無効なモードです。1, 2, 3のいずれかを選択してください。")
    
    # パラメータ設定
    dt = 0.312500e-9  # [s] Sample interval
    dx = 3.6e-2       # [m] Trace interval
    
    # 窓サイズ設定（区切りf-k変換の場合）
    if mode in ['2', '3']:
        print("\n窓サイズを設定してください:")
        
        # 時間方向窓サイズ入力
        while True:
            try:
                window_time_ns = float(input("時間方向の窓サイズ [ns]: ").strip())
                if window_time_ns <= 0:
                    print("正の値を入力してください。")
                    continue
                # nsをサンプル数に変換
                window_time = int(window_time_ns * 1e-9 / dt)
                if window_time < 5:
                    print(f"窓サイズが小さすぎます（{window_time}サンプル）。より大きな値を入力してください。")
                    continue
                break
            except ValueError:
                print("有効な数値を入力してください。")
        
        # 空間方向窓サイズ入力
        while True:
            try:
                window_trace_m = float(input("空間方向の窓サイズ [m]: ").strip())
                if window_trace_m <= 0:
                    print("正の値を入力してください。")
                    continue
                # mをトレース数に変換
                window_trace = int(window_trace_m / dx)
                if window_trace < 5:
                    print(f"窓サイズが小さすぎます（{window_trace}トレース）。より大きな値を入力してください。")
                    continue
                break
            except ValueError:
                print("有効な数値を入力してください。")
        
        print(f"設定された窓サイズ: {window_time}サンプル ({window_time_ns}ns) × {window_trace}トレース ({window_trace_m}m)")
    else:
        # 全体変換の場合はデフォルト値（使用されない）
        window_time = 30
        window_trace = 20
    
    # フィルタータイプ選択
    print("\nf-kフィルタータイプを選択してください:")
    print("1. 三角形フィルター")
    print("2. 四角形フィルター")
    
    filter_choice = input("フィルター番号を入力 (1/2): ").strip()
    filter_types = {'1': 'triangular', '2': 'rectangular'}
    if filter_choice not in filter_types:
        raise ValueError("無効なフィルタータイプです。1または2を選択してください。")
    
    filter_type = filter_types[filter_choice]
    filter_params = {}  # TODO: フィルターパラメータの入力処理を追加
    
    # フィルターの詳細情報を表示
    if filter_type == 'triangular':
        print("\n三角形フィルターが選択されました（f=0を軸とする対称領域）:")
        print("  正の周波数領域:")
        print("    A: (f=250 MHz, k=0 1/m)")
        print("    B: (f=750 MHz, k=4 1/m)")
        print("    C: (f=750 MHz, k=-4 1/m)")
        print("  負の周波数領域:")
        print("    A': (f=-250 MHz, k=0 1/m)")
        print("    B': (f=-750 MHz, k=4 1/m)")
        print("    C': (f=-750 MHz, k=-4 1/m)")
        print("  これらの対称な三角形領域内のf-k成分のみが保持されます。")
    elif filter_type == 'rectangular':
        print("\n四角形フィルターが選択されました（f=0を軸とする対称領域）:")
        print("  正の周波数領域:")
        print("    A: (f=250 MHz, k=-4 1/m)")
        print("    B: (f=750 MHz, k=-4 1/m)")
        print("    C: (f=750 MHz, k=4 1/m)")
        print("    D: (f=250 MHz, k=4 1/m)")
        print("  負の周波数領域:")
        print("    A': (f=-250 MHz, k=-4 1/m)")
        print("    B': (f=-750 MHz, k=-4 1/m)")
        print("    C': (f=-750 MHz, k=4 1/m)")
        print("    D': (f=-250 MHz, k=4 1/m)")
        print("  これらの対称な四角形領域内のf-k成分のみが保持されます。")
    
    # 出力ディレクトリ作成
    output_dir = os.path.join(os.path.dirname(data_path), 'fk_filtering')
    os.makedirs(output_dir, exist_ok=True)
    
    # データ読み込み
    print('データを読み込み中...')
    Bscan_data = np.loadtxt(data_path, delimiter=' ')
    print(f'データ形状: {Bscan_data.shape}')
    
    # NaN値の統計を表示
    nan_count = np.sum(np.isnan(Bscan_data))
    total_count = Bscan_data.size
    if nan_count > 0:
        print(f'NaN値の数: {nan_count} / {total_count} ({nan_count/total_count*100:.2f}%)')
    else:
        print('NaN値は検出されませんでした')
    
    # モード1または3: 全体f-kフィルタリング
    if mode in ['1', '3']:
        print("\n=== 全体f-kフィルタリング ===")
        
        # f-k変換
        KK_shifted, f_MHz, K = fk_transform_full(Bscan_data, dt, dx)
        
        # フィルター適用
        KK_filtered = apply_fk_filter(KK_shifted, f_MHz, K, filter_type, **filter_params)
        
        # 逆変換
        filtered_bscan = inverse_fk_transform(KK_filtered, Bscan_data.shape)
        
        # 比較プロット・保存
        output_path = os.path.join(output_dir, f'full_fk_filtering_{filter_type}.png')
        plot_fk_filter_comparison(Bscan_data, filtered_bscan, KK_shifted, KK_filtered,
                                f_MHz, K, output_path, f"({filter_type})")
        print(f"全体f-kフィルタリング結果を保存: {output_path}")
        
        # フィルター適用後のデータも保存
        np.savetxt(os.path.join(output_dir, f'full_filtered_bscan_{filter_type}.txt'), 
                   filtered_bscan, delimiter=' ')
    
    # モード2または3: 区切りf-kフィルタリング
    if mode in ['2', '3']:
        print(f"\n=== 区切りf-kフィルタリング ===")
        
        # 区切りf-k変換を実行
        windowed_results = fk_transform_windowed(Bscan_data, dt, dx, 
                                               window_time, window_trace)
        
        print(f"処理完了: {len(windowed_results)} 個の窓")
        
        # 出力ディレクトリ名を窓サイズとフィルタータイプで決定
        window_time_ns_display = window_time * dt * 1e9  # サンプル → ns
        window_trace_m_display = window_trace * dx  # トレース → m
        windowed_dir = os.path.join(output_dir, 
                                   f'windowed_results_x{window_trace_m_display:.1f}m_t{window_time_ns_display:.1f}ns_{filter_type}')
        os.makedirs(windowed_dir, exist_ok=True)
        
        print("各窓にフィルターを適用中...")
        filtered_results = []
        
        for idx, result in enumerate(tqdm(windowed_results, desc="フィルタリング中")):
            try:
                # フィルター適用
                KK_filtered = apply_fk_filter(result['data'], result['f_MHz'], result['K'], 
                                            filter_type, **filter_params)
                
                # 逆変換
                filtered_window = inverse_fk_transform(KK_filtered, result['window_data'].shape)
                
                # 結果を保存
                filtered_results.append({
                    'original_data': result['window_data'],
                    'filtered_data': filtered_window,
                    'KK_original': result['data'],
                    'KK_filtered': KK_filtered,
                    'f_MHz': result['f_MHz'],
                    'K': result['K'],
                    'pos': result['pos'],
                    'window_idx': result['window_idx']
                })
                
            except Exception as e:
                print(f"窓 {result['window_idx']} でフィルタリングエラー: {e}")
                continue
        
        print("各窓の結果を保存中...")
        for idx, result in enumerate(tqdm(filtered_results, desc="保存中")):
            t_start_sample, x_start_sample = result['pos']
            
            # 物理単位での開始位置を計算
            t_start_ns = t_start_sample * dt * 1e9  # サンプル → ns
            x_start_m = x_start_sample * dx  # トレース → m
            
            # ファイル名作成（物理単位使用）
            filename = f'window_x{x_start_m:.1f}m_t{t_start_ns:.1f}ns_filtered_{filter_type}.png'
            output_path = os.path.join(windowed_dir, filename)
            
            # 比較プロットで保存
            plot_fk_filter_comparison(
                result['original_data'], result['filtered_data'],
                result['KK_original'], result['KK_filtered'],
                result['f_MHz'], result['K'], output_path, 
                f"({filter_type})"
            )
            
            # フィルター適用後のB-scanデータも保存
            txt_filename = f'window_x{x_start_m:.1f}m_t{t_start_ns:.1f}ns_filtered_{filter_type}.txt'
            txt_path = os.path.join(windowed_dir, txt_filename)
            np.savetxt(txt_path, result['filtered_data'], delimiter=' ')
        
        print(f"区切りf-kフィルタリング結果を保存: {windowed_dir}")
        print(f"総ファイル数: {len(filtered_results)}")
    
    print(f"\n処理完了! 結果は {output_dir} に保存されました。")

if __name__ == "__main__":
    main()