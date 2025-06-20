import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter # ガウス平滑化用
from scipy.linalg import pinv # 疑似逆行列用
from numba import jit, prange # prange は parallel=True のループで必要


def apply_gaussian_smoothing(data: np.ndarray, sigma: float) -> np.ndarray:
    """
    データにガウス平滑化を適用する。
    Local SimilarityのS_m関数の実装として使用されることを想定。

    Args:
        data (np.ndarray): 平滑化を適用するデータ。
        sigma (float): ガウスカーネルの標準偏差。大きいほど強く平滑化される。

    Returns:
        np.ndarray: 平滑化適用後のデータ。
    """
    return gaussian_filter(data, sigma=sigma, mode='nearest')


@jit(nopython=True) # NumbaはNumPyのpinvを直接サポートしていない可能性があるので注意
def _compute_local_similarity_element(sub_A_flat, sub_B_flat):
    """
    Local Similarityの単一要素計算をNumbaで高速化するヘルパー関数。
    この関数は calculate_local_similarity_spectrum から呼び出される。
    """
    n = len(sub_A_flat)
    if n == 0:
        return 0.0

    # A_diag, B_diag は対角行列だが、要素ごとの演算で処理する
    # A^T A (対角行列) の対角要素は sub_A_flat の要素の二乗
    # B^T B (対角行列) の対角要素は sub_B_flat の要素の二乗
    
    # lambda_1, lambda_2 の計算 (スペクトルノルム = 対角要素の絶対値の最大値)
    # np.linalg.norm(A_diag, ord=2) の代わり
    lambda1_val = 0.0
    for k in range(n):
        abs_val = abs(sub_A_flat[k])
        if abs_val > lambda1_val:
            lambda1_val = abs_val
    lambda1_val = lambda1_val ** 2 # A^T Aのノルムなので、要素の二乗の最大値

    lambda2_val = 0.0
    for k in range(n):
        abs_val = abs(sub_B_flat[k])
        if abs_val > lambda2_val:
            lambda2_val = abs_val
    lambda2_val = lambda2_val ** 2 # B^T Bのノルムなので、要素の二乗の最大値

    # A_diag.T @ sub_B_flat は要素ごとの積のベクトル
    # B_diag.T @ sub_A_flat は要素ごとの積のベクトル
    AT_b = sub_A_flat * sub_B_flat
    BT_a = sub_B_flat * sub_A_flat

    # term1_inv_c1 の計算: [lambda_1^2 I + (ATA_diag - lambda_1^2 I)]^-1
    # これは ATA_diag の逆行列に等しい
    # ATA_diag の対角要素は sub_A_flat の二乗
    # その逆行列の対角要素は 1 / (sub_A_flat の二乗)
    # ただし、ゼロ除算を避ける
    inv_ATA_diag_elements = np.zeros(n)
    epsilon = 1e-12 # 極めて小さな値を追加し、ゼロ除算を避ける

    for k in range(n):
        denom_A = sub_A_flat[k]**2 + epsilon # ここにepsilonを追加
        inv_ATA_diag_elements[k] = 1.0 / denom_A

        denom_B = sub_B_flat[k]**2 + epsilon # ここにepsilonを追加
        inv_BTB_diag_elements[k] = 1.0 / denom_B

    # term1_inv_c2 の計算: 同様
    inv_BTB_diag_elements = np.zeros(n)
    for k in range(n):
        if sub_B_flat[k]**2 != 0:
            inv_BTB_diag_elements[k] = 1.0 / (sub_B_flat[k]**2)
        else:
            inv_BTB_diag_elements[k] = 0.0

    # c1 = inv(ATA_diag) @ AT_b  (S_m = I の場合)
    c1 = np.zeros(n)
    for k in range(n):
        c1[k] = inv_ATA_diag_elements[k] * AT_b[k]

    # c2 = inv(BTB_diag) @ BT_a (S_m = I の場合)
    c2 = np.zeros(n)
    for k in range(n):
        c2[k] = inv_BTB_diag_elements[k] * BT_a[k]
    
    # c = sqrt(c1^H c2) = sqrt(dot(c1, c2))
    local_similarity = np.sqrt(np.dot(c1, c2))

    return local_similarity

@jit(nopython=True, parallel=True)
def calculate_local_similarity_spectrum_fast(data_A: np.ndarray, data_B: np.ndarray,
                                             dx: float, dz: float,
                                             sigma_smoothing: float) -> np.ndarray:
    """
    高速化されたLocal Similarityスペクトル計算関数。
    Numbaと行列演算の最適化を適用。
    """
    rows, cols = data_A.shape
    local_similarity_spectrum = np.zeros((rows, cols))

    time_win_pixels = max(1, int(1.0 / dz))
    space_win_pixels = max(1, int(1.0 / dx))

    # Numbaの並列ループ (prange)
    for i in prange(rows): # prange を使用して並列化
        for j in prange(cols): # prange を使用して並列化
            r_start = max(0, i - time_win_pixels // 2)
            r_end = min(rows, i + time_win_pixels // 2 + (time_win_pixels % 2))
            c_start = max(0, j - space_win_pixels // 2)
            c_end = min(cols, j + space_win_pixels // 2 + (space_win_pixels % 2))

            # NumPyのスライシングはNumbaでも効率的に動作する
            sub_A_flat = data_A[r_start:r_end, c_start:c_end].flatten()
            sub_B_flat = data_B[r_start:r_end, c_start:c_end].flatten()

            local_similarity_spectrum[i, j] = _compute_local_similarity_element(sub_A_flat, sub_B_flat)

    # Local Similarityスペクトル全体に平滑化を適用
    # Numbaのjit関数内では、scipy.ndimage.gaussian_filterを直接呼び出すとエラーになる場合がある
    # そのため、この部分はjit関数外で実行するか、Numbaで実装する。
    # ここでは、呼び出し元で平滑化を行うように修正を推奨。
    # 実際には、このsigma_smoothingは calculate_local_similarity_spectrum_fast の引数から削除し、
    # main関数でこの関数の後に apply_gaussian_smoothing を呼び出す形に変更する。
    # （後述の修正案に反映）
    return local_similarity_spectrum


def calculate_local_similarity_spectrum(data_A: np.ndarray, data_B: np.ndarray,
                                       dx: float, dz: float,
                                       sigma_smoothing: float) -> np.ndarray:
    """
    2つのマイグレーション済みデータセット間のLocal Similarityスペクトルを計算する。
    論文のセクション2.3 および Appendix B に基づく。

    Args:
        data_A (np.ndarray): フィルター適用後のデータセットA (時間/深度 x 空間)。
        data_B (np.ndarray): フィルター適用後のデータセットB (時間/深度 x 空間)。
        dx (float): 空間方向のサンプリング間隔 (m)。
        dz (float): 時間/深度方向のサンプリング間隔 (m)。
        sigma_smoothing (float): S_m関数（ガウス平滑化）の標準偏差。

    Returns:
        np.ndarray: Local Similarityスペクトル。
    """
    rows, cols = data_A.shape
    local_similarity_spectrum = np.zeros((rows, cols))

    # ウィンドウサイズを1m x 1mに設定
    # ピクセル単位に変換
    time_win_pixels = max(1, int(1.0 / dz)) # 深度1mに対応するピクセル数
    space_win_pixels = max(1, int(1.0 / dx)) # 距離1mに対応するピクセル数

    # ウィンドウの中心を基準にループ
    for i in range(rows):
        for j in tqdm(range(cols), desc = f"Calculating Local Similarity at Row {i+1}/{rows}"):
            # 現在のピクセルを中心としたウィンドウを定義
            # 境界処理は'reflect'や'wrap'なども検討可能だが、ここでは単純に切り詰める
            r_start = max(0, i - time_win_pixels // 2)
            r_end = min(rows, i + time_win_pixels // 2 + (time_win_pixels % 2)) # 奇数ウィンドウサイズに対応
            c_start = max(0, j - space_win_pixels // 2)
            c_end = min(cols, j + space_win_pixels // 2 + (space_win_pixels % 2)) # 奇数ウィンドウサイズに対応

            # ウィンドウ内のデータ抽出とフラット化
            sub_A_flat = data_A[r_start:r_end, c_start:c_end].flatten()
            sub_B_flat = data_B[r_start:r_end, c_start:c_end].flatten()

            if len(sub_A_flat) == 0 or len(sub_B_flat) == 0:
                local_similarity_spectrum[i, j] = 0.0
                continue

            # Local Similarityの計算 (式 A1-A7)
            # sub_A_flat と sub_B_flat をそれぞれ論文のベクトルaとbと解釈する。
            # A, Bはa, bを主対角要素とする対角行列

            # 対角行列 A_diag, B_diag の作成
            # NumPyのnp.diagは1D配列から対角行列を作成できる
            A_diag = np.diag(sub_A_flat)
            B_diag = np.diag(sub_B_flat)

            # S_m の適用
            # S_mは平滑化を促進する関数
            # ここではガウスカーネルを適用した結果をS_mの作用として解釈する。
            # 厳密には、S_mは行列形式であるべきだが、ここではA^T AやB^T Bに直接平滑化を適用する形で実装。
            # この解釈は論文の不明瞭な点を補完するための仮定である点に注意。
            # 論文の式(A4)と(A5)のS_mは、行列の要素に作用する関数というよりは、行列そのものにかかる係数のような表現。
            # ここでは、S_mを局所的な平滑化フィルタリング作用として解釈し、計算対象の行列に適用する。
            # これも論文の厳密な定義ではない可能性があるため、注意が必要。

            # 論文の式 (A4), (A5) のS_m (smoothing promotion) の解釈には複数の可能性があり、
            # 最も一般的な解釈は、S_mがペナルティ項を導入するような役割を果たすことです。
            # ここでは、S_mを外に出して、計算後に平滑化を行う。あるいは、A^T AとB^T Bを直接平滑化する。
            # A^T A と B^T B はそれぞれ対角行列の積なので、要素はベクトルa, bの要素の二乗になる。

            # 論文の式(A4)(A5)の構造から、S_mはA^T AとB^T Bにかかる演算子である。
            # しかし、詳細がないため、ここではA^T A (対角行列) の要素に平滑化を適用するという仮定を置く。
            # これは、通常の画像処理における「局所平滑化」をS_mの意図として解釈している。

            # しかし、A^T A が対角行列であるという定義を維持すると、
            # S_m (A^T A - lambda_1^2 I) の計算は、対角要素に対して行われる。
            # S_mを関数として定義し、行列の各要素に作用させることを想定すると、
            # gaussian_filter を直接的に行列に適用するのは自然。

            # 最も素直な実装は、S_mをガウスカーネルとして定義し、行列A^T AとB^T Bに作用させること。
            # しかし、A^T A, B^T Bはもはやシンプルなベクトルではなく、ウィンドウ内のデータから作られた行列。
            # 論文の記述は非常に抽象的であるため、最も妥当な解釈を試みる。
            # S_mを要素ごとの乗算や、畳み込みとして実装する。

            # 暫定的なS_mの解釈:
            # S_mは平滑化を「促進する」関数なので、逆問題の解法において正則化項として現れることが多い。
            # 式 (A4) と (A5) の形は、Tikhonov正則化に似ている。
            # S_mが単位行列であれば、単純な正則化になる。

            # ここでは、S_mの具体的な実装として、A^T AとB^T Bの計算結果の対角要素にガウス平滑化を適用すると仮定する。
            # これは論文の厳密な定義ではないが、平滑化の意図を汲むための現実的なアプローチ。

            # A^T A と B^T B の計算 (対角行列の積なので、対角要素は要素の二乗)
            ATA_diag = np.diag(sub_A_flat ** 2)
            BTB_diag = np.diag(sub_B_flat ** 2)

            # S_m の適用 (ここでは、平滑化をA^T AとB^T Bの要素に適用すると解釈)
            # ガウス平滑化は主に空間的な平滑化なので、1D配列に適用する場合は注意が必要。
            # ここではS_mを要素ごとの重み付けとして解釈し、
            # apply_gaussian_smoothing関数を呼び出す形式はここでは不自然なので、
            # S_mを一時的な行列として扱う。
            # S_mが単位行列でない場合、ピン逆行列の計算が複雑になる。

            # 論文の式 (A4) と (A5) は、S_mが正則化行列（例えば、勾配のノルムを最小化するような行列）
            # または要素ごとの重み付け行列であることを示唆している。
            # 「S_m is a function for smoothness promotion」

            # 論文の意図を最もシンプルに解釈すると、S_mは平滑化係数（スカラー）か、
            # 局所的な平滑化フィルタリングを適用する操作と考えるのが妥当。
            # S_mが行列である場合、そのサイズはA^T AやB^T Bと同じ。

            # ここでは、S_mを単純なスカラー乗算因子として扱い、平滑化は後で適用すると解釈を変更します。
            # 論文の図5 (c) (d) のように、Local similarity spectrum自体を平滑化することを示唆している可能性もある。
            # しかし、式の $S_m (A^T A - \lambda_1^2 I)$ の位置は、行列演算の一部。
            # この曖昧さがあるため、実装が複雑になります。

            # **暫定的な対応**:
            # 論文の式 (A4) と (A5) はそのままに、S_mを単位行列として実装します。
            # これは「平滑化を促進する関数」という記述とは異なりますが、具体的なS_mの定義がないため、
            # 最初の実装としてはこのアプローチが最も安全です。
            # 平滑化は、Local Similarityスペクトル計算後に`apply_gaussian_smoothing`関数で別途行うことを推奨します。
            # これにより、論文の図5(c)から(d)への変換（ソフト閾値適用後の平滑化）をシミュレートできます。

            # $\lambda_1, \lambda_2$ の計算 (A6, A7)
            # ||A^T A||_2 はFrobeniusノルムかスペクトルノルムか不明。ここではFrobeniusノルムとする。
            # Aが対角行列なので、A^T Aも対角行列。その対角要素は sub_A_flat の各要素の二乗。
            # 対角行列のFrobeniusノルムは、対角要素の二乗和の平方根。
            # 対角行列のスペクトルノルムは、対角要素の絶対値の最大値。

            # 論文の図 (Figure 5) の例を見ると、0〜2の範囲のスペクトル値。
            # これは相関係数に近い値を示唆している可能性もあるが、ノルムの計算方法によって異なる。
            # ここでは、スペクトルノルム（最大特異値）として計算します。
            # 対角行列の場合、スペクトルノルムは対角要素の絶対値の最大値。
            lambda1_val = np.linalg.norm(A_diag, ord=2) # スペクトルノルム
            lambda2_val = np.linalg.norm(B_diag, ord=2) # スペクトルノルム

            # 式 (A4) c1 = [lambda_1^2 I + S_m (A^T A - lambda_1^2 I)]^-1 S_m A^T b
            # S_m を単位行列 I と仮定
            if len(sub_A_flat) == 0: # ゼロ除算回避
                c1 = np.zeros_like(sub_A_flat)
            else:
                term1_inv_c1 = pinv(lambda1_val**2 * np.eye(len(sub_A_flat)) + (ATA_diag - lambda1_val**2 * np.eye(len(sub_A_flat))))
                c1 = term1_inv_c1 @ A_diag.T @ sub_B_flat

            # 式 (A5) c2 = [lambda_2^2 I + S_m (B^T B - lambda_2^2 I)]^-1 S_m B^T a
            if len(sub_B_flat) == 0: # ゼロ除算回避
                c2 = np.zeros_like(sub_B_flat)
            else:
                term1_inv_c2 = pinv(lambda2_val**2 * np.eye(len(sub_B_flat)) + (BTB_diag - lambda2_val**2 * np.eye(len(sub_B_flat))))
                c2 = term1_inv_c2 @ B_diag.T @ sub_A_flat

            # 式 (A1) c = sqrt(c1^H c2)
            # c1^H は c1 の共役転置。c1が実数の場合、ただの転置 (内積)。
            # 論文の図を見ると実数値なので、内積でOK。
            if c1.size == 0 or c2.size == 0:
                local_similarity = 0.0
            else:
                local_similarity = np.sqrt(np.dot(c1, c2))

            local_similarity_spectrum[i, j] = local_similarity

    # Local Similarityスペクトル全体に平滑化を適用（S_mの意図を補完）
    # 論文の図5 (d) に類似度スペクトルがソフト閾値後に平滑化されているように見えるため、ここで適用。
    if sigma_smoothing > 0:
        local_similarity_spectrum = apply_gaussian_smoothing(local_similarity_spectrum, sigma=sigma_smoothing)

    return local_similarity_spectrum


def apply_soft_threshold(similarity_spectrum: np.ndarray, threshold: float) -> np.ndarray:
    """
    Local Similarityスペクトルにソフト閾値関数を適用する 。
    論文の式(8) に基づく 。
    """
    # 論文の式 (8):
    # c_ij'(D_A, D_B) = c_ij(D_A, D_B) if c_ij(D_A, D_B) > epsilon
    #                  = 0          if c_ij(D_A, D_B) <= epsilon
    thresholded_spectrum = np.where(similarity_spectrum > threshold, similarity_spectrum, 0)
    return thresholded_spectrum


def mute_reflection_area(spectrum: np.ndarray, reflection_area_mask) -> np.ndarray:
    """
    反射領域をミュートする。
    reflection_area_maskは、ミュートする領域のインデックス（例: [(r_start, r_end, c_start, c_end), ...])
    またはブールマスクとして提供されると想定。
    """
    muted_spectrum = spectrum.copy()
    if isinstance(reflection_area_mask, np.ndarray) and reflection_area_mask.dtype == bool:
        muted_spectrum[reflection_area_mask] = 0
    elif isinstance(reflection_area_mask, list):
        for area in reflection_area_mask:
            r_s, r_e, c_s, c_e = area
            muted_spectrum[r_s:r_e, c_s:c_e] = 0
    return muted_spectrum


from scipy.ndimage import maximum_filter

def extract_local_maximums(spectrum: np.ndarray, neighborhood_size: int = 3) -> list[tuple[int, int]]:
    """
    Local Similarityスペクトルから局所最大値を抽出し、岩石の位置を特定する 。
    """
    # 局所最大値を検出するためのフィルターを適用
    # neighborhood_sizeは、ピークを検出するための近傍サイズ。
    # マイグレーション後のデータはシャープなので、小さめの値が適切。
    local_max = (spectrum == maximum_filter(spectrum, size=neighborhood_size))

    # 閾値0以上の点のみを対象とする (ソフト閾値処理後なので、ほとんどのノイズは0になっているはず)
    rock_locations = np.argwhere(local_max & (spectrum > 0)).tolist()

    return rock_locations


def plot(data, x_axis: np.ndarray, z_axis: np.ndarray, output_dir: str, output_name: str):
    fig, ax = plt.subplots(figsize=(18, 6))
    im = ax.imshow(data, aspect='auto', cmap='turbo',
                        extent=[x_axis.min(), x_axis.max(), z_axis.max(), z_axis.min()],
                        origin='lower')
    ax.set_xlabel('Distance (m)', fontsize=20)
    ax.set_ylabel('Depth (m)', fontsize=20)
    ax.tick_params(labelsize=18)

    # カラーバーの追加
    delvider = axgrid1.make_axes_locatable(plt.gca())
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Amplitude', fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_name + '.pdf'), dpi=300)
    plt.show()


def plot_rock_locations_on_spectrum(data, x_axis: np.ndarray, z_axis: np.ndarray, output_dir: str, output_name: str, rock_locations: list):
    fig, ax = plt.subplots(figsize=(18, 6))
    im = ax.imshow(data, aspect='auto', cmap='turbo',
                        extent=[x_axis.min(), x_axis.max(), z_axis.max(), z_axis.min()],
                        origin='lower') # origin='lower'でZ軸が下向きになるよう調整

    ax.set_xlabel('Distance (m)', fontsize=20)
    ax.set_ylabel('Depth (m)', fontsize=20)
    ax.tick_params(labelsize=18)

    # 岩石の位置をオーバーレイ
    if rock_locations: # リストが空でない場合のみプロット
        # rock_locationsは (row_idx, col_idx) なので、x_plotはcol_idx*dx, z_plotはrow_idx*dz
        # しかしimshowのextentとorigin='lower'の関係で、z_axisは最大値から最小値の順になるので
        # プロットするz座標はそのままdz * row_idxで良い。
        rock_x_coords = [loc[1] * x_axis[1] for loc in rock_locations] # col_idx * dx
        rock_z_coords = [loc[0] * z_axis[1] for loc in rock_locations] # row_idx * dz
        ax.plot(rock_x_coords, rock_z_coords, 'ro', markersize=5, alpha=0.7) # 赤い点でオーバーレイ

    # カラーバーの追加
    divider = axgrid1.make_axes_locatable(ax) # plt.gca() -> ax
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Amplitude', fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_name + '.pdf'), dpi=300)
    plt.show()


def main():
    # Set constants
    dt = 0.312500e-9  # Sample interval in seconds
    dx = 3.6e-2  # Trace interval in meters, from Li et al. (2020), Sci. Adv.
    c = 299792458  # Speed of light in m/s
    epsilon_r = 4.5  # Relative permittivity, from Feng et al. (2024)
    dz = dt * c / np.sqrt(epsilon_r) / 2  # Depth interval in meters
    sigma_smoothing = 0.5  # ガウス平滑化の標準偏差
    soft_threshold_value = 0.1  # ソフト閾値の値
    reflection_mask = None  # 反射領域のマスク (必要に応t場合はリストで指定)
    local_max_neighborhood = 3  # 局所最大値を検出するための近傍サイズ

    # Input paths
    data_paths = [
        ["/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2A/Processed_Data/4_Gain_function/4_Bscan_gain.txt",
            "/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Processed_Data/4_Gain_function/4_Bscan_gain.txt"],
        ["/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2A/Processed_Data/5_Terrain_correction/5_Terrain_correction.txt",
            "/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Processed_Data/5_Terrain_correction/5_Terrain_correction.txt"]
    ]

    # User input for data type selection
    data_type = input("Select data type (1 for Gained data, 2 for Terrain corrected data): ").strip()
    if data_type not in ['1', '2']:
        raise ValueError("Invalid data type selected. Please choose 1 or 2.")

    # Determine the data paths based on user input
    if data_type == '1':
        data1_path, data2_path = data_paths[0]
    else:
        data1_path, data2_path = data_paths[1]

    if not os.path.exists(data1_path):
        raise FileNotFoundError(f"The file {data1_path} does not exist.")
    if not os.path.exists(data2_path):
        raise FileNotFoundError(f"The file {data2_path} does not exist.")

    # Load data
    print("Loading data...")
    data1 = np.loadtxt(data1_path)
    print(f"Data 1 shape: {data1.shape}")
    data2 = np.loadtxt(data2_path)
    print(f"Data 2 shape: {data2.shape}")
    print(" ")

    # Set axis
    x_axis = np.arange(data1.shape[1]) * dx  # x-axis in meters
    z_axis = np.arange(data1.shape[0]) * dz # z-axis in meters

    # Output directory
    base_dir = "/Volumes/SSD_Kanda_SAMSUNG/LPR/Local_similarity/local_similarity"
    if data_type == '1':
        output_dir = os.path.join(base_dir, '4_Gain_function')
    else:
        output_dir = os.path.join(base_dir, '5_Terrain_correction')
    os.makedirs(output_dir, exist_ok=True)

    """
    マイグレーション済みLPRデータから岩石の位置を特定するエンドツーエンドの処理パイプライン。
    """
    # Local Similarityスペクトルの計算
    # S_mの平滑化はcalculate_local_similarity_spectrum内で実行
    print("Calculating Local Similarity Spectrum...")
    similarity_spectrum = calculate_local_similarity_spectrum_fast(data1, data2, dx, dz, sigma_smoothing) # sigma_smoothing はここでは0に
    np.savetxt(os.path.join(output_dir, 'local_similarity_spectrum_raw.txt'), similarity_spectrum, fmt='%.6f')
    print("Finished")
    print(" ")

    # ソフト閾値関数の適用
    # Local Similarityスペクトル全体に平滑化を適用
    if sigma_smoothing > 0:
        print("Applying Gaussian smoothing to similarity spectrum...")
        thresholded_spectrum = apply_gaussian_smoothing(similarity_spectrum, sigma=sigma_smoothing)
        np.savetxt(os.path.join(output_dir, 'local_similarity_spectrum_smoothed.txt'), similarity_spectrum, fmt='%.6f')
        print("Finished")
        print(" ")
    else:
        thresholded_spectrum = similarity_spectrum

    # 反射領域のミュート (オプション)
    print("Muting reflection area...")
    if reflection_mask is not None:
        muted_spectrum = mute_reflection_area(thresholded_spectrum, reflection_mask)
    else:
        muted_spectrum = thresholded_spectrum
    np.savetxt(os.path.join(output_dir, 'local_similarity_spectrum_muted.txt'), muted_spectrum, fmt='%.6f')
    print("Finished")
    print(" ")

    # 局所最大値の抽出
    print("Extracting local maximums for rock locations...")
    rock_locations = extract_local_maximums(muted_spectrum, local_max_neighborhood)
    print(f"Detected {len(rock_locations)} rock locations.")
    rock_locations_file = os.path.join(output_dir, 'rock_locations.txt')
    with open(rock_locations_file, 'w') as f:
        for loc in rock_locations:
            f.write(f"{loc[0]} {loc[1]}\n")
    print("Finished")
    print(" ")

    # Plot
    print("Plotting results...")
    plot(similarity_spectrum, x_axis, z_axis, output_dir, 'local_similarity')
    plot(thresholded_spectrum, x_axis, z_axis, output_dir, 'local_similarity_thresholded')
    if reflection_mask is not None:
        plot(muted_spectrum, x_axis, z_axis, output_dir, 'local_similarity_muted')
    plot_rock_locations_on_spectrum(rock_locations, x_axis, z_axis, output_dir, 'rock_locations', rock_locations)

if __name__ == "__main__":
    main()