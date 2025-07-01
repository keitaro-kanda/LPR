import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
from tqdm import tqdm

def find_time_zero_index(data):
    """
    x=0において最初に非NaN値が現れるインデックスを見つける（実際のt=0に対応）
    
    Parameters:
    -----------
    data : np.ndarray
        入力B-scanデータ (time x traces)
    
    Returns:
    --------
    t_zero_index : int
        t=0に対応する実際のデータインデックス
    """
    if data.shape[1] == 0:
        return 0
    
    # x=0（最初のトレース）のデータを取得
    first_trace = data[:, 0]
    
    # 最初に非NaN値が現れるインデックスを見つける
    valid_indices = np.where(~np.isnan(first_trace))[0]
    
    if len(valid_indices) == 0:
        print("警告: x=0において有効なデータが見つかりません。t=0をインデックス0とします。")
        return 0
    
    t_zero_index = valid_indices[0]
    print(f"t=0インデックス: {t_zero_index} （x=0で最初の有効データ位置）")
    return t_zero_index

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

def fk_transform_representative_points(data, dt, dx, window_time, window_trace, representative_points):
    """
    代表点周辺の窓でf-k変換を実行
    
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
    representative_points : list
        代表点座標のリスト [(x1, t1), (x2, t2), ...]、x[m], t[ns]
    
    Returns:
    --------
    results : list
        各代表点の変換結果 [{'data': KK_shifted, 'f_MHz': f_MHz, 'K': K, 'pos': (t_start, x_start), 'center': (x, t), 'window_data': window_data}]
    """
    print(f'代表点f-k変換を実行中... (窓サイズ: {window_time} x {window_trace})')
    
    N_time, N_traces = data.shape
    results = []
    
    print(f'データ形状: {N_time} x {N_traces}')
    print(f'代表点数: {len(representative_points)}')
    
    # t=0に対応する実際のデータインデックスを取得
    t_zero_index = find_time_zero_index(data)
    
    # 各代表点での処理
    for point_idx, (x_center_m, t_center_ns) in enumerate(tqdm(representative_points, desc="代表点処理")):
        # 物理座標をサンプル/トレースインデックスに変換
        # 時間座標はt=0インデックスからの相対位置として計算
        t_center_sample = t_zero_index + int(t_center_ns * 1e-9 / dt)
        x_center_trace = int(x_center_m / dx)
        
        # 窓の範囲を計算（中心座標を基準）
        half_window_time = window_time // 2
        half_window_trace = window_trace // 2
        
        t_start = max(0, t_center_sample - half_window_time)
        t_end = min(N_time, t_center_sample + half_window_time)
        x_start = max(0, x_center_trace - half_window_trace)
        x_end = min(N_traces, x_center_trace + half_window_trace)
        
        # 実際の窓サイズを確保（境界で調整された場合）
        actual_window_time = t_end - t_start
        actual_window_trace = x_end - x_start
        
        # 窓データを抽出
        window_data = data[t_start:t_end, x_start:x_end]
        
        # デバッグ情報：窓の詳細を表示
        actual_t_start_ns = t_start * dt * 1e9
        actual_t_end_ns = t_end * dt * 1e9  
        actual_x_start_m = x_start * dx
        actual_x_end_m = x_end * dx
        
        print(f"代表点 {point_idx+1} (x={x_center_m}, t={t_center_ns}): " +
              f"窓範囲 x=[{actual_x_start_m:.1f}, {actual_x_end_m:.1f}]m, " +
              f"t=[{actual_t_start_ns:.1f}, {actual_t_end_ns:.1f}]ns, " +
              f"サイズ={actual_window_trace}×{actual_window_time}")
        
        # 窓が小さすぎる場合はスキップ
        if window_data.shape[0] < 10 or window_data.shape[1] < 10:
            print(f"  → 窓が小さすぎます - スキップ")
            continue
        
        # データ範囲の確認
        if x_center_trace >= N_traces or t_center_sample >= N_time:
            print(f"  → データ範囲外 - スキップ")
            continue
        
        # NaN値の確認
        if np.any(np.isnan(window_data)):
            print(f"  → NaN値が含まれています - 処理を続行")
        
        # 有効なデータがあるかチェック
        if np.all(np.isnan(window_data)) or np.nanmax(np.abs(window_data)) == 0:
            print(f"  → 有効なデータがありません - スキップ")
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
                'center': (x_center_m, t_center_ns),
                'point_idx': point_idx + 1,
                'window_data': window_data  # B-scanデータも保存
            })
            
            print(f"  → 処理完了")
            
        except Exception as e:
            print(f"  → エラー: {e}")
            continue
    
    return results

def plot_fk_result(KK_shifted, f_MHz, K, output_path, title="F-K Transform"):
    """
    f-k変換結果をプロット・保存（単体版）
    """
    # 対数スケールで振幅を計算（NaN値を処理）
    KK_abs = np.abs(KK_shifted)
    KK_abs_clean = np.nan_to_num(KK_abs, nan=1e-10, posinf=1e10, neginf=1e-10)
    KK_power_log = 20 * np.log10(KK_abs_clean + 1e-10)  # 小さい値を追加してlog(0)を防ぐ
    
    plt.figure(figsize=(12, 8))
    
    # imshow plot
    im = plt.imshow(KK_power_log, aspect='auto',
                    extent=(K.min(), K.max(), f_MHz.min(), f_MHz.max()),
                    cmap='turbo', origin='lower',
                    vmin=np.percentile(KK_power_log, 5),
                    vmax=np.percentile(KK_power_log, 99))
    
    # ラベル設定（フォントサイズ20、明示的な位置指定）
    ax = plt.gca()
    ax.set_xlabel('Wavenumber [1/m]', fontsize=20)
    ax.set_ylabel('Frequency [MHz]', fontsize=20)
    ax.tick_params(labelsize=16)
    
    # ラベル位置を明示的に指定
    ax.xaxis.set_label_position('bottom')  # xラベルを下に
    ax.yaxis.set_label_position('left')    # yラベルを左に

    # Colorbar
    divider = axgrid1.make_axes_locatable(plt.gca())
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Amplitude [dB]', fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    
    
    
    plt.tight_layout()
    
    # PNG とPDF両方で保存
    base_path = output_path.rsplit('.', 1)[0]  # 拡張子を除去
    plt.savefig(f'{base_path}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{base_path}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_combined_bscan_fk(window_data, KK_shifted, f_MHz, K, output_path, 
                          x_start_m, t_start_ns, dt, dx, global_data_max):
    """
    B-scanとf-k変換を併記した画像を出力
    
    Parameters:
    -----------
    window_data : np.ndarray
        窓切り取りしたB-scanデータ
    KK_shifted : np.ndarray  
        f-k変換結果
    f_MHz : np.ndarray
        周波数軸 [MHz]
    K : np.ndarray
        波数軸 [1/m]
    output_path : str
        出力ファイルパス
    x_start_m : float
        切り取り開始位置 [m]
    t_start_ns : float
        切り取り開始時間 [ns]
    dt : float
        時間サンプリング間隔 [s]
    dx : float
        空間サンプリング間隔 [m]
    global_data_max : float
        全データの最大値（カラーバー範囲統一用）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left plot: B-scan window
    t_axis_ns = np.arange(window_data.shape[0]) * dt * 1e9 + t_start_ns
    x_axis_m = np.arange(window_data.shape[1]) * dx + x_start_m
    
    # B-scanのカラーバー範囲を全データ共通にする
    vmin_bscan = -global_data_max / 15
    vmax_bscan = global_data_max / 15
    
    im1 = ax1.imshow(window_data, aspect='auto', cmap='viridis',
                     extent=[x_axis_m[0], x_axis_m[-1], t_axis_ns[-1], t_axis_ns[0]],
                     vmin=vmin_bscan, vmax=vmax_bscan)
    ax1.set_xlabel('Position [m]', fontsize=20)
    ax1.set_ylabel('Time [ns]', fontsize=20)
    ax1.set_title(f'B-scan Window\nx_start = {x_start_m:.1f} [m], t_start = {t_start_ns:.1f} [ns]', fontsize=20)
    ax1.tick_params(labelsize=16)
    
    # Colorbar for B-scan (x軸下に配置)
    divider1 = axgrid1.make_axes_locatable(ax1)
    cax1 = divider1.append_axes('bottom', size='5%', pad=0.7)
    cbar1 = plt.colorbar(im1, cax=cax1, orientation='horizontal')
    cbar1.set_label('Amplitude', fontsize=20)
    cbar1.ax.tick_params(labelsize=16)
    
    # Right plot: F-K transform（NaN値を処理）
    KK_abs = np.abs(KK_shifted)
    KK_abs_clean = np.nan_to_num(KK_abs, nan=1e-10, posinf=1e10, neginf=1e-10)
    KK_power_log = 20 * np.log10(KK_abs_clean + 1e-10)
    im2 = ax2.imshow(KK_power_log, aspect='auto',
                     extent=(K.min(), K.max(), f_MHz.min(), f_MHz.max()),
                     cmap='turbo', origin='lower',
                     vmin=np.percentile(KK_power_log, 5),
                     vmax=np.percentile(KK_power_log, 99))
    
    ax2.set_xlabel('Wavenumber [1/m]', fontsize=20)
    ax2.set_ylabel('Frequency [MHz]', fontsize=20)
    ax2.set_title(f'F-K Transform\nx_start = {x_start_m:.1f} [m], t_start = {t_start_ns:.1f} [ns]', fontsize=20)
    ax2.tick_params(labelsize=16)
    
    # Colorbar for F-K transform (x軸下に配置)
    divider2 = axgrid1.make_axes_locatable(ax2)
    cax2 = divider2.append_axes('bottom', size='5%', pad=0.7)
    cbar2 = plt.colorbar(im2, cax=cax2, orientation='horizontal')
    cbar2.set_label('Amplitude [dB]', fontsize=20)
    cbar2.ax.tick_params(labelsize=16)
    
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
    print("1. 全体f-k変換")
    print("2. 区切りf-k変換")
    print("3. テストモード（代表的な地点での区切りf-k変換）")
    
    mode = input("モード番号を入力 (1/2/3): ").strip()
    if mode not in ['1', '2', '3']:
        raise ValueError("無効なモードです。1, 2, 3のいずれかを選択してください。")
    
    # パラメータ設定
    dt = 0.312500e-9  # [s] Sample interval
    dx = 3.6e-2       # [m] Trace interval
    
    # テストモード用の代表点座標定義
    if mode == '3':
        print("\n*** 注意: テストモードは terrain_corrected データでの座標を使用します ***")
        print("代表的な地点での f-k 変換を実行します")
        
        # 代表点座標 (x [m], t [ns])
        representative_points = [
            (322, 65), (364, 70), (247, 65), (600, 90), 
            (1180, 90), (710, 130), (850, 80), (325, 190), (1050, 380)
        ]
        
        print("代表点座標 (terrain_corrected):")
        for i, (x, t) in enumerate(representative_points):
            print(f"  {i+1}: (x={x} m, t={t} ns)")
    
    # 窓サイズ設定（区切りf-k変換とテストモードの場合）
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
    
    # 出力ディレクトリ作成
    output_dir = os.path.join(os.path.dirname(data_path), 'fk_transformation')
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
    
    # モード1: 全体f-k変換
    if mode == '1':
        print("\n=== 全体f-k変換 ===")
        KK_shifted, f_MHz, K = fk_transform_full(Bscan_data, dt, dx)
        
        # プロット・保存
        output_path = os.path.join(output_dir, 'full_fk_transform.png')
        plot_fk_result(KK_shifted, f_MHz, K, output_path, "Full F-K Transform")
        print(f"全体f-k変換結果を保存: {output_path}")
        
        # データも保存
        np.savetxt(os.path.join(output_dir, 'full_fk_data.txt'), 
                   np.abs(KK_shifted), delimiter=' ')
    
    # モード2: 区切りf-k変換
    if mode == '2':
        print(f"\n=== 区切りf-k変換 ===")
        
        # 全データの最大値を計算（カラーバー範囲統一用、NaN値を考慮）
        global_data_max = np.nanmax(np.abs(Bscan_data))
        if np.isnan(global_data_max) or global_data_max == 0:
            global_data_max = 1.0
        
        # t=0に対応する実際のデータインデックスを取得
        t_zero_index = find_time_zero_index(Bscan_data)
        
        # 区切りf-k変換を実行
        windowed_results = fk_transform_windowed(Bscan_data, dt, dx, 
                                               window_time, window_trace)
        
        print(f"処理完了: {len(windowed_results)} 個の窓")
        
        # 出力ディレクトリ名を窓サイズで決定
        window_time_ns_display = window_time * dt * 1e9  # サンプル → ns
        window_trace_m_display = window_trace * dx  # トレース → m
        windowed_dir = os.path.join(output_dir, f'windowed_results_x{window_trace_m_display:.1f}m_t{window_time_ns_display:.1f}ns')
        os.makedirs(windowed_dir, exist_ok=True)
        
        print("各窓の結果を保存中...")
        for idx, result in enumerate(tqdm(windowed_results, desc="保存中")):
            t_start_sample, x_start_sample = result['pos']
            
            # 物理単位での開始位置を計算（t=0インデックス補正適用）
            t_start_ns = (t_start_sample - t_zero_index) * dt * 1e9  # サンプル → ns
            x_start_m = x_start_sample * dx  # トレース → m
            
            # ファイル名作成（物理単位使用）
            filename = f'window_x{x_start_m:.1f}m_t{t_start_ns:.1f}ns_combined.png'
            output_path = os.path.join(windowed_dir, filename)
            
            # 併記プロットで保存（全データ最大値を渡す）
            plot_combined_bscan_fk(
                result['window_data'], result['data'], result['f_MHz'], result['K'],
                output_path, x_start_m, t_start_ns, dt, dx, global_data_max
            )
        
        print(f"区切りf-k変換結果を保存: {windowed_dir}")
        print(f"総ファイル数: {len(windowed_results)}")
    
    # モード3: テストモード（代表点での区切りf-k変換）
    if mode == '3':
        print(f"\n=== テストモード: 代表点での区切りf-k変換 ===")
        
        # 全データの最大値を計算（カラーバー範囲統一用、NaN値を考慮）
        global_data_max = np.nanmax(np.abs(Bscan_data))
        if np.isnan(global_data_max) or global_data_max == 0:
            global_data_max = 1.0
        
        # 代表点での f-k変換を実行
        test_results = fk_transform_representative_points(
            Bscan_data, dt, dx, window_time, window_trace, representative_points
        )
        
        print(f"処理完了: {len(test_results)} 個の代表点")
        
        # 出力ディレクトリ名を決定
        window_time_ns_display = window_time * dt * 1e9  # サンプル → ns
        window_trace_m_display = window_trace * dx  # トレース → m
        windowed_dir = os.path.join(output_dir, f'windowed_results_x{window_trace_m_display:.1f}m_t{window_time_ns_display:.1f}ns')
        test_dir = os.path.join(windowed_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)
        
        print("各代表点の結果を保存中...")
        for result in tqdm(test_results, desc="保存中"):
            x_center, t_center = result['center']
            
            # ファイル名作成（代表点座標使用）
            png_filename = f'x={x_center}_t={t_center}.png'
            pdf_filename = f'x={x_center}_t={t_center}.pdf'
            png_path = os.path.join(test_dir, png_filename)
            pdf_path = os.path.join(test_dir, pdf_filename)
            
            # 併記プロットで保存（PNG）
            plot_combined_bscan_fk(
                result['window_data'], result['data'], result['f_MHz'], result['K'],
                png_path, x_center, t_center, dt, dx, global_data_max
            )
            
            # PDF版も保存
            plot_combined_bscan_fk(
                result['window_data'], result['data'], result['f_MHz'], result['K'],
                pdf_path, x_center, t_center, dt, dx, global_data_max
            )
        
        print(f"テストモード結果を保存: {test_dir}")
        print(f"代表点数: {len(test_results)}")
        print("各代表点について PNG と PDF ファイルを生成しました")
    
    print(f"\n処理完了! 結果は {output_dir} に保存されました。")

if __name__ == "__main__":
    main()