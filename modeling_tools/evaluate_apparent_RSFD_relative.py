import numpy as np
import pandas as pd
import matplotlib
# 並列処理時のクラッシュ防止と描画高速化のため、非対話型バックエンドを指定
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
import json
import concurrent.futures # 並列化のために追加
from tqdm import tqdm # 進捗バーのために追加

# ============================================================
# --- 0. 実行モード設定（ハードコーディング） ---
# ============================================================
# 'N' : 岩石数 N_true を入力とする既存の系（出力先 = BASE_DIR_N）
# 'K' : 累積岩石密度 k_true を入力とする系（出力先 = BASE_DIR_K）
#
# k_true と N_true は k_true = N_true / AREA_SIZE_M2 で厳密に対応するため、
# K系でも N_true = round(k_true * AREA_SIZE_M2) として一意に復元される。
# したがって N を必要とする出力（累積個数プロット、D_max 算出等）も
# 両モードで同一に生成される。
INPUT_MODE = 'K'   # <<< ここを 'N' / 'K' で切り替える

# 出力の親ディレクトリ
BASE_DIR_N = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD_relative'
BASE_DIR_K = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD_relative_kinput'

# K系で計算する k_true [1/m^2] のリスト（D >= ROCK_SIZE_MIN の累積岩石密度）
K_TRUE_LIST = [5e-3, 10e-3, 50e-3, 100e-3]

# 最大岩石サイズの上限 [m]（D_MAX_MODE='auto' のときのみ有効）
MAX_ROCK_SIZE_LIMIT = 10.0

# ============================================================
# --- 0b. 最大岩石サイズ D_max の決定方法（ハードコーディング） ---
# ============================================================
# 'fixed': ROCK_SIZE_MAX を最大岩石サイズとして使う（推奨）
#          D_max が N・r に依存しないため、全ての (r, k) 組み合わせが計算可能。
#          月地下に存在しうる最大の岩石サイズを物理的な根拠から与える、という考え方。
# 'auto' : D_max = D_min * N^(1/r)（従来方式。N(>D_max)=1 となるサイズ）
#          D_max が非現実的に大きくなる条件は MAX_ROCK_SIZE_LIMIT で打ち切る。
D_MAX_MODE = 'fixed'

# D_MAX_MODE='fixed' のときの最大岩石サイズ [m]
#   1.0 : CE-4/LPR で観測された最大岩石 46.8 cm の約2倍
#   2.0 : r_true が 1.0 付近まで届く（切断による丸まりが小さくなるため）
# 切断の丸まりにより、フィットで得られる r_true は入力 r より大きくなる。
#   ROCK_SIZE_MAX=1.0 -> 入力 r=0.6/1.0/2.0/4.0 で r_true=1.19/1.36/2.02/3.81
#   ROCK_SIZE_MAX=2.0 -> 同                      r_true=1.00/1.21/1.95/3.84
ROCK_SIZE_MAX = 2.0

# --- 1. 設定クラス ---
class RadarConfig:
    """
    シミュレーションの全パラメータを管理するクラス
    """
    def __init__(self, total_rocks=10000):
        # --- A. レーダーシステム ---
        self.FREQ = 500e6  # 500 MHz
        self.C_0 = 3e8
        
        # 閾値（背景除去後の平均強度プロファイルで、表面反射に対する最低強度を参考に決定）
        # 表面反射で規格化した相対強度なので単位は [dB]（dBm ではない）
        self.NOISE_FLOOR_DB = -90.0
        self.H_ANTENNA = 0.3 # アンテナ地上高 [m]
        
        # --- 物理定数 ---
        self.EPSILON_0 = 8.854e-12
        self.MU_0 = 4 * np.pi * 1e-7
        
        # --- B. 環境・媒質パラメータ ---
        self.EPSILON_R_AIR = 1.0   # 真空(空気)の比誘電率
        self.EPSILON_R_REG = 3.0
        self.LOSS_TANGENT = 0.004
        self.EPSILON_R_ROCK = 9.0
        
        # --- C. ばらつきパラメータ ---
        self.SIGMA_RANDOM_DB = 3.0 # Random項の標準偏差 [dB] (任意に調整してください)

        # --- D. シミュレーション空間設定 ---
        self.MAX_DEPTH = 12.0
        # すべての面積計算のベースとなる統一パラメータ
        self.AREA_SIZE_M2 = 18300.0 # 深さ12 m x 奥行き1525 mのエリア
        # 岩石サイズの下限 [m]。
        # LPR CH2 のサイズ推定下限であり、RSFD のフィット下限でもある。
        # （これより小さい岩石が存在しない、という意味ではない）
        self.ROCK_SIZE_MIN = 0.06
        # 岩石サイズの上限 [m]（D_MAX_MODE='fixed' のときに使用）
        self.ROCK_SIZE_MAX = ROCK_SIZE_MAX
        self.TOTAL_ROCKS = total_rocks

    def d_max(self, r_true):
        """
        最大岩石サイズ D_max [m] を返す。

          'fixed': ROCK_SIZE_MAX（N・r に依存しない固定値）
          'auto' : D_min * N^(1/r)（N(>D_max)=1 となるサイズ。従来方式）

        'auto' では D_max が (r, N) に強く依存し、r が小さく N が大きい条件で
        非現実的な値（数十 m）になるため、多くの (r, k) 組み合わせが計算不能になる。
        """
        if D_MAX_MODE == 'fixed':
            return self.ROCK_SIZE_MAX
        return self.ROCK_SIZE_MIN * (self.TOTAL_ROCKS ** (1.0 / r_true))

    @property
    def lambda_0(self):
        return self.C_0 / self.FREQ

    @property
    def lambda_reg(self):
        """レゴリス中の波長 [m]。岩石RCSの散乱領域分けに使用する。"""
        return self.lambda_0 / np.sqrt(self.EPSILON_R_REG)

    @property
    def lambda_g(self):
        """後方互換用エイリアス（= lambda_reg）"""
        return self.lambda_reg

    @property
    def attenuation_np_m(self):
        """
        電界の減衰定数 alpha [Np/m]   (Jol 2009, Eq. 2.14)

            alpha = 2*pi*f * sqrt( (mu*eps/2) * ( sqrt(1+tan^2 delta) - 1 ) )

        平方根は「(mu*eps/2) と (sqrt(1+tan^2 delta)-1) の積」全体にかかる。
        8.686 [dB/Np] はここでは掛けない（非dB形式で統一するため）。
        """
        omega = 2 * np.pi * self.FREQ
        epsilon = self.EPSILON_R_REG * self.EPSILON_0
        mu = self.MU_0  # mu_r = 1
        inner = 0.5 * mu * epsilon * (np.sqrt(1 + self.LOSS_TANGENT**2) - 1)
        return omega * np.sqrt(inner)

    @property
    def attenuation_db_m(self):
        """参考値: 片道パワー減衰率 [dB/m]。表示専用で計算本体では使用しない。"""
        return (20.0 / np.log(10.0)) * self.attenuation_np_m

    @property
    def noise_floor_linear(self):
        """検出しきい値を線形比に変換 (-90 dB -> 1e-9)"""
        return 10.0 ** (self.NOISE_FLOOR_DB / 10.0)

    @property
    def true_rock_density(self):
        """true RSFD の累積岩石密度 k_true [1/m^2]（D >= ROCK_SIZE_MIN）"""
        return self.TOTAL_ROCKS / self.AREA_SIZE_M2

    @property
    def reflection_coeff(self):
        # レゴリスと岩石の反射係数
        n1 = np.sqrt(self.EPSILON_R_REG)
        n2 = np.sqrt(self.EPSILON_R_ROCK)
        return ((n1 - n2) / (n1 + n2))

    @property
    def surface_reflection_coeff(self):
        # 空気(真空)とレゴリスの表面反射係数
        n_air = np.sqrt(self.EPSILON_R_AIR)
        n_reg = np.sqrt(self.EPSILON_R_REG)
        return ((n_air - n_reg) / (n_air + n_reg))
    
    @property
    def fresnel_radius(self):
        """
        地表面での第1フレネル半径 [m]。
        アンテナ-地表面間の伝搬は真空(空気)中で起こるため lambda_0 を用いる。
        """
        return np.sqrt(self.lambda_0 * self.H_ANTENNA / 2.0)

    @property
    def surface_rcs(self):
        """
        地表面のレーダー断面積 [m^2]（線形）。

            sigma_surf = Gamma_surf^2 * 4*pi*A^2 / lambda_0^2,  A = pi * r_fres^2

        入射波・散乱波ともに空気中の場なので lambda は lambda_0。
        代入すると lambda_0 は約分され sigma_surf = Gamma^2 * pi^3 * h_ante^2 となる。
        """
        A = np.pi * self.fresnel_radius**2
        return self.surface_reflection_coeff**2 * 4.0 * np.pi * A**2 / self.lambda_0**2

# --- 2. 物理モデルクラス ---
class RockModel:
    """
    岩石の生成とレーダー方程式の適用を行うクラス
    """
    def generate_rocks(self, r_true, config, quiet=False):
        D_min = config.ROCK_SIZE_MIN
        D_max = config.d_max(r_true)

        if D_max <= D_min:
            raise ValueError(
                f"D_max ({D_max:.3f} m) が D_min ({D_min:.3f} m) 以下です。"
                "ROCK_SIZE_MAX の設定を確認してください。"
            )

        if not quiet:
            mode_note = ("固定値" if D_MAX_MODE == 'fixed' else "N と r から自動決定")
            print(f"岩石を生成中... (True Slope = -{r_true}, N={config.TOTAL_ROCKS})")
            print(f"  -> 最大岩石サイズ (D_max): {D_max:.2f} m [{mode_note}]")
        
        # 0から1の範囲を TOTAL_ROCKS 個で等間隔に分割した配列を作成（分位数）
        # u = (np.arange(config.TOTAL_ROCKS) + 0.5) / config.TOTAL_ROCKS
        # サイズが常に大きい順（または小さい順）に並んでしまうのを防ぐため、配列をシャッフルする
        # np.random.shuffle(u)

        # 均等割りではなく、一様分布からランダムにサンプリングする
        u = np.random.uniform(0.0, 1.0, size=config.TOTAL_ROCKS)
        
        # [D_min, D_max] の切断べき乗分布から逆関数法でサンプリング
        term1 = D_max ** (-r_true)
        term2 = D_min ** (-r_true)
        
        diameters = ( (term1 - term2) * u + term2 ) ** (-1.0 / r_true)
        depths = np.random.uniform(0, config.MAX_DEPTH, size=len(diameters))
        
        df = pd.DataFrame({
            'diameter': diameters,
            'depth': depths
        })
        return df

    def calculate_rcs(self, diameter_array, config):
        """
        岩石のレーダー断面積 [m^2]（線形）。dB には変換しない。

        散乱領域の境界は、散乱体を取り囲む媒質（レゴリス）中の波長の 1/2。
          幾何光学領域 (D >= lambda_reg/2): sigma = Gamma^2 * pi * (D/2)^2
          レイリー領域   (D <  lambda_reg/2): sigma = sigma|_{D=lambda_reg/2} * (D/(lambda_reg/2))^6
        """
        boundary_size = config.lambda_reg / 2.0
        gamma2 = config.reflection_coeff**2

        sigma_optical = gamma2 * np.pi * (diameter_array / 2.0)**2

        sigma_opt_at_bound = gamma2 * np.pi * (boundary_size / 2.0)**2
        sigma_rayleigh = sigma_opt_at_bound * (diameter_array / boundary_size)**6

        return np.where(diameter_array >= boundary_size, sigma_optical, sigma_rayleigh)

    def apply_radar_equation(self, rocks_df, config, quiet=False):
        """
        地表面反射で規格化した受信強度 P_norm（線形比）を計算する。

            P_norm = (sigma_rock/sigma_surf) * (h_ante/R_rock)^4
                     * exp(-4*alpha_reg*d_rock) * 10^(X/10),  X ~ N(0, sigma_dB^2)

            R_rock = h_ante + d_rock  : GPR-岩石間距離
            d_rock                    : 岩石の深さ（吸収損失を受ける片道経路長）
        """
        if not quiet:
            print("レーダーシミュレーション実行中 (相対値モデル・非dB形式)...")
            print(f"  減衰定数 alpha = {config.attenuation_np_m:.5f} Np/m "
                  f"({config.attenuation_db_m:.5f} dB/m 片道)")

        depth = rocks_df['depth'].values          # d_rock [m]
        h_ante = config.H_ANTENNA                 # = R_surf
        R_rock = h_ante + depth                   # GPR-岩石間距離 [m]

        # 1. RCS 比（線形）: sigma_rock / sigma_surf
        sigma_rock = self.calculate_rcs(rocks_df['diameter'].values, config)
        rcs_ratio = sigma_rock / config.surface_rcs

        # 2. 幾何学的減衰比: (h_ante / R_rock)^4
        spread_ratio = (h_ante / R_rock)**4

        # 3. 媒質による吸収損失（往復）: exp(-4 * alpha * d_rock)
        #    片道パワー exp(-2*alpha*d) を往復（経路長 2d）にするので指数は -4*alpha*d。
        #    吸収はレゴリス中でのみ起こるため R_rock ではなく depth を使う。
        attenuation = np.exp(-4.0 * config.attenuation_np_m * depth)

        # 4. Random項（対数正規）: dB での N(0, sigma_dB) に対応する乗算因子
        random_factor = 10.0 ** (
            np.random.normal(0.0, config.SIGMA_RANDOM_DB, size=len(rocks_df)) / 10.0
        )

        # 規格化強度 P_norm（線形比）
        p_norm = rcs_ratio * spread_ratio * attenuation * random_factor

        # 閾値判定（線形量どうしで比較）
        is_detected = p_norm > config.noise_floor_linear

        result_df = rocks_df.copy()
        result_df['rcs'] = sigma_rock                      # [m^2]
        result_df['p_norm'] = p_norm                       # 線形比
        result_df['p_norm_db'] = 10.0 * np.log10(p_norm)   # 表示・プロット用のみ
        result_df['is_detected'] = is_detected

        return result_df

# --- 3. 解析クラス ---
class Analyzer:
    """
    検出された岩石データからRSFDを計算・プロットするクラス
    """
    def calculate_slope(self, diameters, area=1.0):
        """
        累積密度分布の傾きを算出する
        個数を面積(area)で規格化し、密度としてフィッティングを行う
        """
        if len(diameters) < 2:
            return np.nan, np.nan, np.array([]), np.array([])
            
        sorted_d = np.sort(diameters)
        y_cumulative = np.arange(len(sorted_d), 0, -1) / area
        
        x_log = np.log10(sorted_d)
        y_log = np.log10(y_cumulative)
        
        n = len(x_log)
        sx = x_log.sum(); sy = y_log.sum()
        sxx = np.dot(x_log, x_log); sxy = np.dot(x_log, y_log)
        denom = n * sxx - sx * sx
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n

        return slope, intercept, x_log, y_log

    def run_depth_analysis(self, detected_df, config, step=1.0, quiet=False):
        """
        既存の累積深さ範囲ごとの解析
        """
        if not quiet:
            print("深さ方向の感度解析(Range)を実行中...")
        results = []
        depth_ranges = np.arange(1.0, config.MAX_DEPTH + 0.1, step)
        
        # 面積の共通化：エリアの幅(Length)を算出
        length = config.AREA_SIZE_M2 / config.MAX_DEPTH
        d_min_cm = config.ROCK_SIZE_MIN * 100.0

        # ループ前に1回だけnumpy配列へ変換・ソート
        is_det = detected_df['is_detected'].values
        depths = detected_df['depth'].values
        diameters = detected_df['diameter'].values

        det_depths = depths[is_det]
        det_diameters = diameters[is_det]
        det_order = np.argsort(det_depths)
        det_depths_sorted = det_depths[det_order]
        det_diameters_by_depth = det_diameters[det_order]
        all_depths_sorted = np.sort(depths)

        for d in depth_ranges:
            n_det = int(np.searchsorted(det_depths_sorted, d, side='right'))
            n_all = int(np.searchsorted(all_depths_sorted, d, side='right'))

            count_detected = n_det
            count_all = n_all

            # 深さごとの面積を算出
            area = d * length

            if count_detected > 5:
                # cm単位に変換して傾きを計算
                diameters_cm = det_diameters_by_depth[:n_det] * 100.0
                slope, intercept, _, _ = self.calculate_slope(diameters_cm, area)
                r_apparent = -slope
                k_apparent = (10**intercept) * (d_min_cm**slope)
            else:
                r_apparent = np.nan
                k_apparent = np.nan

            detection_rate = count_detected / count_all if count_all > 0 else 0.0
            rock_density = count_detected / area if area > 0 else 0.0

            results.append({
                'depth_range': d,
                'r_apparent': r_apparent,
                'k_apparent': k_apparent,
                'count_detected': count_detected,
                'count_all': count_all,
                'detection_rate': detection_rate,
                'rock_density': rock_density
            })

        return pd.DataFrame(results)

    def run_moving_window_analysis(self, detected_df, config, window_size=2.0, step_ratio=0.2, quiet=False):
        """
        指定した窓幅(window_size)と移動幅(step_ratio)での移動窓解析
        """
        if not quiet:
            print("深さ方向の移動窓解析(Moving Window)を実行中...")
        results = []
        step_size = window_size * step_ratio
        
        d_starts = np.arange(0.0, config.MAX_DEPTH - window_size + 1e-5, step_size)
        
        # 面積の共通化
        length = config.AREA_SIZE_M2 / config.MAX_DEPTH
        area = window_size * length
        d_min_cm = config.ROCK_SIZE_MIN * 100.0

        # ループ前に1回だけnumpy配列へ変換・ソート
        is_det = detected_df['is_detected'].values
        depths = detected_df['depth'].values
        diameters = detected_df['diameter'].values

        det_depths = depths[is_det]
        det_diameters = diameters[is_det]
        det_order = np.argsort(det_depths)
        det_depths_sorted = det_depths[det_order]
        det_diameters_by_depth = det_diameters[det_order]
        all_depths_sorted = np.sort(depths)

        for d_start in d_starts:
            d_end = d_start + window_size
            d_center = d_start + window_size / 2.0

            # searchsorted で O(log N) のウィンドウ検索
            det_s = int(np.searchsorted(det_depths_sorted, d_start, side='right'))
            det_e = int(np.searchsorted(det_depths_sorted, d_end, side='right'))
            all_s = int(np.searchsorted(all_depths_sorted, d_start, side='right'))
            all_e = int(np.searchsorted(all_depths_sorted, d_end, side='right'))

            count_detected = det_e - det_s
            count_all = all_e - all_s

            if count_detected > 5:
                # cm単位に変換して傾きを計算
                diameters_cm = det_diameters_by_depth[det_s:det_e] * 100.0
                slope, intercept, _, _ = self.calculate_slope(diameters_cm, area)
                r_apparent = -slope
                k_apparent = (10**intercept) * (d_min_cm**slope)
            else:
                r_apparent = np.nan
                k_apparent = np.nan

            detection_rate = count_detected / count_all if count_all > 0 else 0.0
            rock_density = count_detected / area if area > 0 else 0.0

            results.append({
                'depth_center': d_center,
                'r_apparent': r_apparent,
                'k_apparent': k_apparent,
                'count_detected': count_detected,
                'count_all': count_all,
                'detection_rate': detection_rate,
                'rock_density': rock_density
            })
            
        return pd.DataFrame(results)

    # === 信号強度のScatterプロット ===
    def plot_power_scatter(self, df, output_prefix, config):
        """
        横軸：深さ、縦軸：岩石サイズ の空間に、
        色で信号強度（p_norm）をマッピングする散布図。
        計算は線形で行い、表示のみ dB に変換している。
        """
        plt.figure(figsize=(10, 8))
        
        sc = plt.scatter(df['depth'], df['diameter'], 
                         c=df['p_norm_db'], cmap='jet', 
                         alpha=0.8, s=20, edgecolors='none', vmax=0, vmin=-120)
        
        cbar = plt.colorbar(sc)
        cbar.set_label('Received Power $P_{norm}$ [dB]', fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.axhline(config.NOISE_FLOOR_DB, color='red', linestyle='--', linewidth=2)

        plt.xlabel('Depth [m]', fontsize=18)
        plt.ylabel('Diameter [m]', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        # plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    # === 個別出力用プロットメソッド (True Fitもpolyfitで描画 / cm単位) ===
    def plot_csfd(self, detected_df, all_rocks_df, output_prefix, r_true, config):
        plt.figure(figsize=(10, 8))
        
        # 全計算で一貫して同じAREA_SIZE_M2を使用
        overall_area = config.AREA_SIZE_M2
        d_min_cm = config.ROCK_SIZE_MIN * 100.0

        # --- True Fit ---
        diameters_true_cm = all_rocks_df['diameter'].values * 100.0
        if len(diameters_true_cm) > 0:
            slope_true, intercept_true, x_log_true, y_log_true = self.calculate_slope(diameters_true_cm, overall_area)
            if len(x_log_true) > 0:
                plt.scatter(10**x_log_true, 10**y_log_true, s=20, color='gray', label='Generated Rocks', marker='D', alpha=0.7)
            if not np.isnan(slope_true):
                x_fit_true = np.logspace(min(x_log_true), max(x_log_true), 100)
                y_fit_true = slope_true * np.log10(x_fit_true) + intercept_true
                k_true = (10**intercept_true) * (d_min_cm**slope_true)
                plt.plot(x_fit_true, 10**y_fit_true, 'k--', linewidth=2.0, 
                         label=f'True Fit (r={-slope_true:.2f}, k={k_true:.2e})')

        # --- Apparent Fit ---
        diameters_det_cm = detected_df['diameter'].values[detected_df['is_detected'].values] * 100.0
        if len(diameters_det_cm) > 0:
            slope_det, intercept_det, x_log_det, y_log_det = self.calculate_slope(diameters_det_cm, overall_area)
            if len(x_log_det) > 0:
                plt.scatter(10**x_log_det, 10**y_log_det, s=20, color='blue', label='Detected Rocks', marker='o')
            if not np.isnan(slope_det):
                x_fit_det = np.logspace(min(x_log_det), max(x_log_det), 100)
                y_fit_det = slope_det * np.log10(x_fit_det) + intercept_det
                k_det = (10**intercept_det) * (d_min_cm**slope_det)
                plt.plot(x_fit_det, 10**y_fit_det, 'r-', linewidth=2.5, 
                         label=f'Apparent Fit (r={-slope_det:.2f}, k={k_det:.2e})')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Diameter [m]', fontsize=18)
        plt.xticks([6, 10, 20, 50, 100, 200, 500], [6, 10, 20, 50, 100, 200, 500], fontsize=16)
        plt.xlim(d_min_cm * 0.8, max(diameters_true_cm.max(), diameters_det_cm.max()) * 1.2)
        plt.ylabel('Cumulative Rock Density N(>D) [1/m²]', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        # plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_depth_analysis(self, analysis_df, output_prefix, r_true, x_col='depth_range', xlabel='Depth Range [m]'):
        plt.figure(figsize=(10, 8))
        plt.plot(analysis_df[x_col], analysis_df['r_apparent'], marker='o', linestyle='-', color='blue', label='Apparent r')
        plt.axhline(y=r_true, color='k', linestyle='--', label='True r')
        
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel('Apparent Slope r', fontsize=18)
        plt.xticks(np.arange(2, 14, 2), ['0-2', '2-4', '4-6', '6-8', '8-10', '10-12'], fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=16)
        plt.grid(True, ls='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        # plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_k_analysis(self, analysis_df, output_prefix, true_k, x_col='depth_range', xlabel='Depth Range [m]'):
        plt.figure(figsize=(10, 8))
        plt.plot(analysis_df[x_col], analysis_df['k_apparent'], marker='^', linestyle='-', color='red', label='Apparent k')
        plt.axhline(y=true_k, color='k', linestyle='--', label=f'True k: {true_k:.2e}')
        
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel('Apparent Density Coefficient k [1/m²]', fontsize=18)
        plt.xticks(np.arange(2, 14, 2), ['0-2', '2-4', '4-6', '6-8', '8-10', '10-12'], fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=16)
        plt.grid(True, ls='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        # plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_detection_rate(self, analysis_df, output_prefix, x_col='depth_range', xlabel='Depth Range [m]'):
        plt.figure(figsize=(10, 8))
        plt.plot(analysis_df[x_col], analysis_df['detection_rate'], marker='o', linestyle='-', color='k', label='Detection Rate')
        plt.axhline(y=1.0, color='k', linestyle='--', label='100% Detection Rate')
        
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel('Detection Rate', fontsize=18)
        plt.xticks(np.arange(2, 14, 2), ['0-2', '2-4', '4-6', '6-8', '8-10', '10-12'], fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        
        plt.ylim(0, 1.05) 
        plt.legend(fontsize=16)
        plt.grid(True, ls='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        # plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    # === 統計出力用プロットメソッド (エラー範囲、±ラベルの追加) ===
    def plot_csfd_stats(self, csfd_stats_df, output_prefix, r_true, config):
        plt.figure(figsize=(10, 8))

        d_min_cm = config.ROCK_SIZE_MIN * 100.0
        N_total = config.TOTAL_ROCKS
        D_max_auto_cm = config.d_max(r_true) * 100.0

        x_min_log = np.log10(d_min_cm)
        x_max_log = np.log10(D_max_auto_cm)
        x_log_common = np.linspace(x_min_log, x_max_log, 100)
        x_common = 10**x_log_common

        # --- True Fit ---
        y_true_all = []
        for _, row in csfd_stats_df.iterrows():
            if not np.isnan(row['slope_true']):
                y_true_all.append(row['slope_true'] * x_log_common + row['intercept_true'])
        
        if y_true_all:
            y_true_all = np.array(y_true_all)
            y_true_mean = np.mean(y_true_all, axis=0)
            y_true_std = np.std(y_true_all, axis=0)
            
            r_true_all = -csfd_stats_df['slope_true'].dropna()
            mean_r_true = np.mean(r_true_all)
            std_r_true = np.std(r_true_all)
            
            k_true_all = (10**csfd_stats_df['intercept_true'].dropna()) * (d_min_cm**csfd_stats_df['slope_true'].dropna())
            mean_k_true = np.mean(k_true_all)
            std_k_true = np.std(k_true_all)
            
            plt.plot(x_common, 10**y_true_mean, 'k--', linewidth=2.0, 
                     label=f'True fit: r={mean_r_true:.2f} $\\pm$ {std_r_true:.2f}, k={mean_k_true:.2e} $\\pm$ {std_k_true:.2e}')
            plt.fill_between(x_common, 10**(y_true_mean - y_true_std), 10**(y_true_mean + y_true_std), color='gray', alpha=0.3)

        # --- Apparent Fit ---
        y_det_all = []
        for _, row in csfd_stats_df.iterrows():
            if not np.isnan(row['slope_det']):
                y_det_all.append(row['slope_det'] * x_log_common + row['intercept_det'])

        if y_det_all:
            y_det_all = np.array(y_det_all)
            y_det_mean = np.mean(y_det_all, axis=0)
            y_det_std = np.std(y_det_all, axis=0)
            
            r_det_all = -csfd_stats_df['slope_det'].dropna()
            mean_r_det = np.mean(r_det_all)
            std_r_det = np.std(r_det_all)
            
            k_det_all = (10**csfd_stats_df['intercept_det'].dropna()) * (d_min_cm**csfd_stats_df['slope_det'].dropna())
            mean_k_det = np.mean(k_det_all)
            std_k_det = np.std(k_det_all)
            
            plt.plot(x_common, 10**y_det_mean, 'r-', linewidth=2.5, 
                     label=f'Apparent fit: r={mean_r_det:.2f} $\\pm$ {std_r_det:.2f}, k={mean_k_det:.2e} $\\pm$ {std_k_det:.2e}')
            plt.fill_between(x_common, 10**(y_det_mean - y_det_std), 10**(y_det_mean + y_det_std), color='red', alpha=0.3)

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Diameter [cm]', fontsize=18)
        plt.xticks([6, 10, 20, 50, 100, 200, 500], [6, 10, 20, 50, 100, 200, 500], fontsize=16)
        plt.xlim(d_min_cm * 0.8, D_max_auto_cm * 1.2)
        plt.ylabel('Cumulative Rock Density N(>D) [1/m²]', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_depth_analysis_stats(self, stats_df, output_prefix, r_true, x_col='depth_range', xlabel='Depth Range [m]'):
        plt.figure(figsize=(10, 8))
        
        x = stats_df[x_col]
        y_mean = stats_df['r_apparent_mean']
        y_std = stats_df['r_apparent_std']

        plt.plot(x, y_mean, marker='o', linestyle='-', color='blue', label='Mean Apparent r')
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='blue', alpha=0.3, label='±1 Std Dev')
        plt.axhline(y=r_true, color='k', linestyle='--', label='True r')
        
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel('Apparent Slope r', fontsize=18)
        if x_col == 'depth_range':
            tick_vals = np.arange(2, 13, 2)
            plt.xticks(tick_vals, [f"0-{int(v)}" for v in tick_vals], fontsize=16)
        else:
            plt.tick_params(axis='x', which='major', labelsize=16)
            
        plt.tick_params(axis='y', which='major', labelsize=16)
        plt.ylim(0, r_true + 1.5)
        plt.legend(fontsize=16)
        plt.grid(True, ls='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_k_analysis_stats(self, stats_df, output_prefix, true_k, x_col='depth_range', xlabel='Depth Range [m]'):
        plt.figure(figsize=(10, 8))
        
        x = stats_df[x_col]
        y_mean = stats_df['k_apparent_mean']
        y_std = stats_df['k_apparent_std']

        plt.plot(x, y_mean, marker='^', linestyle='-', color='red', label='Mean Apparent k')
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='red', alpha=0.3, label='±1 Std Dev')
        plt.axhline(y=true_k, color='k', linestyle='--', label=f'True k: {true_k:.2e}')
        
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel('Apparent Density Coefficient k [1/m²]', fontsize=18)
        if x_col == 'depth_range':
            tick_vals = np.arange(2, 13, 2)
            plt.xticks(tick_vals, [f"0-{int(v)}" for v in tick_vals], fontsize=16)
        else:
            plt.tick_params(axis='x', which='major', labelsize=16)
            
        plt.tick_params(axis='y', which='major', labelsize=16)
        plt.legend(fontsize=16)
        plt.grid(True, ls='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_detection_rate_stats(self, stats_df, output_prefix, x_col='depth_range', xlabel='Depth Range [m]'):
        plt.figure(figsize=(10, 8))
        
        x = stats_df[x_col]
        y_mean = stats_df['detection_rate_mean']
        y_std = stats_df['detection_rate_std']

        plt.plot(x, y_mean, marker='o', linestyle='-', color='k', label='Mean Detection Rate')
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='k', alpha=0.3, label='±1 Std Dev')
        plt.axhline(y=1.0, color='k', linestyle='--', label='100% Detection Rate')
        
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel('Detection Rate', fontsize=18)
        if x_col == 'depth_range':
            tick_vals = np.arange(2, 13, 2)
            plt.xticks(tick_vals, [f"0-{int(v)}" for v in tick_vals], fontsize=16)
        else:
            plt.tick_params(axis='x', which='major', labelsize=16)
            
        plt.tick_params(axis='y', which='major', labelsize=16)
        
        plt.ylim(0, 1.05) 
        plt.legend(fontsize=16)
        plt.grid(True, ls='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_csfd_count_stats(self, csfd_stats_df, output_prefix, r_true, config):
        plt.figure(figsize=(10, 8))

        d_min_cm = config.ROCK_SIZE_MIN * 100.0
        N_total = config.TOTAL_ROCKS
        D_max_auto_cm = config.d_max(r_true) * 100.0

        x_min_log = np.log10(d_min_cm)
        x_max_log = np.log10(D_max_auto_cm)
        x_log_common = np.linspace(x_min_log, x_max_log, 100)
        x_common = 10**x_log_common

        # --- True Fit (Count) ---
        y_true_all = []
        for _, row in csfd_stats_df.iterrows():
            if not np.isnan(row['slope_true_cnt']):
                y_true_all.append(row['slope_true_cnt'] * x_log_common + row['intercept_true_cnt'])

        if y_true_all:
            y_true_all = np.array(y_true_all)
            y_true_mean = np.mean(y_true_all, axis=0)
            y_true_std = np.std(y_true_all, axis=0)

            r_true_cnt_all = -csfd_stats_df['slope_true_cnt'].dropna()
            mean_r_true_cnt = np.mean(r_true_cnt_all)
            std_r_true_cnt = np.std(r_true_cnt_all)

            k_true_cnt_all = (10**csfd_stats_df['intercept_true_cnt'].dropna()) * (d_min_cm**csfd_stats_df['slope_true_cnt'].dropna())
            mean_k_true_cnt = np.mean(k_true_cnt_all)
            std_k_true_cnt = np.std(k_true_cnt_all)

            plt.plot(x_common, 10**y_true_mean, 'k--', linewidth=2.0,
                     label=f'True fit: r={mean_r_true_cnt:.2f} $\\pm$ {std_r_true_cnt:.2f}, k={mean_k_true_cnt:.2e} $\\pm$ {std_k_true_cnt:.2e}')
            plt.fill_between(x_common, 10**(y_true_mean - y_true_std), 10**(y_true_mean + y_true_std), color='gray', alpha=0.3)

        # --- Apparent Fit (Count) ---
        y_det_all = []
        for _, row in csfd_stats_df.iterrows():
            if not np.isnan(row['slope_det_cnt']):
                y_det_all.append(row['slope_det_cnt'] * x_log_common + row['intercept_det_cnt'])

        if y_det_all:
            y_det_all = np.array(y_det_all)
            y_det_mean = np.mean(y_det_all, axis=0)
            y_det_std = np.std(y_det_all, axis=0)

            r_det_cnt_all = -csfd_stats_df['slope_det_cnt'].dropna()
            mean_r_det_cnt = np.mean(r_det_cnt_all)
            std_r_det_cnt = np.std(r_det_cnt_all)

            k_det_cnt_all = (10**csfd_stats_df['intercept_det_cnt'].dropna()) * (d_min_cm**csfd_stats_df['slope_det_cnt'].dropna())
            mean_k_det_cnt = np.mean(k_det_cnt_all)
            std_k_det_cnt = np.std(k_det_cnt_all)

            plt.plot(x_common, 10**y_det_mean, 'r-', linewidth=2.5,
                     label=f'Apparent fit: r={mean_r_det_cnt:.2f} $\\pm$ {std_r_det_cnt:.2f}, k={mean_k_det_cnt:.2e} $\\pm$ {std_k_det_cnt:.2e}')
            plt.fill_between(x_common, 10**(y_det_mean - y_det_std), 10**(y_det_mean + y_det_std), color='red', alpha=0.3)

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Diameter [cm]', fontsize=18)
        plt.xticks([6, 10, 20, 50, 100, 200, 500], [6, 10, 20, 50, 100, 200, 500], fontsize=16)
        plt.xlim(d_min_cm * 0.8, D_max_auto_cm * 1.2)
        plt.ylabel('Cumulative Rock Count N(>D)', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

# --- 新規: 並列処理用のワーカー関数 ---
def process_iteration(args):
    i, r_true, total_rocks, output_dir, overall_area, true_rock_density = args
    
    config = RadarConfig(total_rocks=total_rocks)
    model = RockModel()
    analyzer = Analyzer()
    
    iter_dir = f"{output_dir}/each_iteration/iter_{i:02d}"
    os.makedirs(iter_dir, exist_ok=True)

    all_rocks_df = model.generate_rocks(r_true, config, quiet=True)
    all_rocks_df.to_csv(f"{iter_dir}/truth_rocks.csv", index=False)

    detected_df = model.apply_radar_equation(all_rocks_df, config, quiet=True)
    detected_df.to_csv(f"{iter_dir}/simulated_detection.csv", index=False)

    analyzer.plot_power_scatter(detected_df, f"{iter_dir}/power_scatter", config)

    d_min_cm = config.ROCK_SIZE_MIN * 100.0

    # True Fit (cm)
    diameters_true_cm = all_rocks_df['diameter'].values * 100.0
    slope_true, intercept_true, _, _ = analyzer.calculate_slope(diameters_true_cm, overall_area)
    r_true_iter = -slope_true if not np.isnan(slope_true) else np.nan
    k_true_iter = (10**intercept_true) * (d_min_cm**slope_true) if not np.isnan(slope_true) else np.nan

    # Apparent Fit (cm)
    diameters_det_cm = detected_df['diameter'].values[detected_df['is_detected'].values] * 100.0
    slope_det, intercept_det, _, _ = analyzer.calculate_slope(diameters_det_cm, overall_area)
    r_det_iter = -slope_det if not np.isnan(slope_det) else np.nan
    k_det_iter = (10**intercept_det) * (d_min_cm**slope_det) if not np.isnan(slope_det) else np.nan

    # Count Fits (cm)
    slope_true_cnt, intercept_true_cnt, _, _ = analyzer.calculate_slope(diameters_true_cm, 1.0)
    slope_det_cnt, intercept_det_cnt, _, _ = analyzer.calculate_slope(diameters_det_cm, 1.0)

    csfd_stats = {
        'iteration': i,
        'slope_true': slope_true,
        'intercept_true': intercept_true,
        'r_true': r_true_iter,
        'k_true': k_true_iter,
        'slope_det': slope_det,
        'intercept_det': intercept_det,
        'r_det': r_det_iter,
        'k_det': k_det_iter,
        'slope_true_cnt': slope_true_cnt,
        'intercept_true_cnt': intercept_true_cnt,
        'slope_det_cnt': slope_det_cnt,
        'intercept_det_cnt': intercept_det_cnt
    }

    analyzer.plot_csfd(detected_df, all_rocks_df, f"{iter_dir}/csfd_comparison", r_true, config)

    analysis_range = analyzer.run_depth_analysis(detected_df, config, step=1.0, quiet=True)
    analysis_range['iteration'] = i 
    analysis_range.to_csv(f"{iter_dir}/depth_analysis_range.csv", index=False)
    
    analyzer.plot_depth_analysis(analysis_range, f"{iter_dir}/powerlaw_exp_range", r_true, x_col='depth_range', xlabel='Depth Range [m]')
    analyzer.plot_k_analysis(analysis_range, f"{iter_dir}/powerlaw_k_range", true_rock_density, x_col='depth_range', xlabel='Depth Range [m]')
    analyzer.plot_detection_rate(analysis_range, f"{iter_dir}/detection_rate_range", x_col='depth_range', xlabel='Depth Range [m]')

    analysis_moving = analyzer.run_moving_window_analysis(detected_df, config, window_size=2.0, step_ratio=0.2, quiet=True)
    analysis_moving['iteration'] = i 
    analysis_moving.to_csv(f"{iter_dir}/depth_analysis_moving.csv", index=False)
    
    analyzer.plot_depth_analysis(analysis_moving, f"{iter_dir}/powerlaw_exp_moving", r_true, x_col='depth_center', xlabel='Depth Center [m]')
    analyzer.plot_k_analysis(analysis_moving, f"{iter_dir}/powerlaw_k_moving", true_rock_density, x_col='depth_center', xlabel='Depth Center [m]')

    return csfd_stats, analysis_range, analysis_moving

# --- 4. メイン処理 ---
def build_cases():
    """
    INPUT_MODE に応じて計算ケースのリストを構築する。

    各ケースは以下のキーを持つ:
      total_rocks : 生成する岩石数 N_true
      k_nominal   : 指定した k_true [1/m^2]（N系では N/AREA から算出）
      dir_name    : 出力サブディレクトリ名
      json_key    : rsfd_summary.json 内のキー
      label       : 画面表示用のラベル
    """
    area = RadarConfig().AREA_SIZE_M2

    if INPUT_MODE == 'K':
        cases = []
        for k_true in K_TRUE_LIST:
            n = int(round(k_true * area))
            if n < 2:
                print(f"  -> [警告] k={k_true:g} は岩石数 {n} 個となりフィット不能のためスキップします。")
                continue
            cases.append({
                'total_rocks': n,
                'k_nominal': k_true,
                'dir_name': f"k_{k_true:g}",
                'json_key': f"{k_true:g}",
                'label': f"k_true = {k_true:g} [1/m2] (N = {n})",
            })
        return cases

    # --- N系（既存の挙動） ---
    try:
        rock_counts_str = input("計算する岩石数をカンマ区切りで入力してください (例: 100,500,1000,5000) を入力: ")
        rock_counts = [int(x.strip()) for x in rock_counts_str.split(',')]
    except ValueError:
        print("数値の入力ではありません。デフォルト値を使用します。")
        rock_counts = [100, 500, 1000, 5000]

    return [{
        'total_rocks': n,
        'k_nominal': n / area,
        'dir_name': f"{n}",
        'json_key': f"{n}",
        'label': f"岩石数 {n} (k_true = {n / area:.4e} [1/m2])",
    } for n in rock_counts]


def main():
    print("--- 岩石見逃しモデル 統計シミュレーション (相対値モデル・非dB形式) ---")
    print(f"--- 入力モード: {INPUT_MODE} "
          f"({'k_true 入力系' if INPUT_MODE == 'K' else 'N_true 入力系'}) ---")
    
    try:
        r_input_str = input("真のべき指数(r)を入力してください (例: 1.0): ")
        r_true = float(r_input_str)
    except ValueError:
        print("数値ではない入力です。デフォルト値 1.0 を使用します。")
        r_true = 1.0
    
    cases = build_cases()

    base_dir = BASE_DIR_K if INPUT_MODE == 'K' else BASE_DIR_N
    os.makedirs(base_dir, exist_ok=True)
    
    json_path = os.path.join(base_dir, 'rsfd_summary.json')

    NUM_ITERATIONS = 100 
    
    for case in cases:
        total_rocks = case['total_rocks']
        print(f"\n=== {case['label']} のシミュレーションを開始 ({NUM_ITERATIONS}回) ===")
        
        config = RadarConfig(total_rocks=total_rocks)
        
        # --- 最大岩石サイズの事前チェック ---
        # D_MAX_MODE='fixed' では D_max が N・r に依存しないため、この打ち切りは不要。
        D_max_used = config.d_max(r_true)
        if D_MAX_MODE == 'auto' and D_max_used > MAX_ROCK_SIZE_LIMIT:
            print(f"  -> [警告] 計算上の最大岩石サイズ ({D_max_used:.2f} m) が "
                  f"{MAX_ROCK_SIZE_LIMIT:.1f} m を超えるため、計算を中止します。")
            
            # JSONへの「Calculation stop」の記録
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        summary_data = json.load(f)
                except json.JSONDecodeError:
                    summary_data = {}
            else:
                summary_data = {}

            r_key = str(r_true)
            case_key = case['json_key']
            if r_key not in summary_data:
                summary_data[r_key] = {}
            
            summary_data[r_key][case_key] = "Calculation stop"
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=4)
            print(f"  -> {json_path} にストップを記録し、次の条件へスキップします。")
            continue
        # --------------------------------------------------------

        output_dir = f"{base_dir}/r_{r_true}/{case['dir_name']}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"出力ディレクトリ: {output_dir}")

        overall_area = config.AREA_SIZE_M2
        # k_true = N_true / A （全岩石が D >= ROCK_SIZE_MIN のため厳密に成立）
        true_rock_density = config.true_rock_density

        with open(f"{output_dir}/parameters.txt", "w") as f:
            f.write(f"Input Mode: {INPUT_MODE}\n")
            f.write(f"True Slope (r): {r_true}\n")
            f.write(f"Total Rocks: {config.TOTAL_ROCKS}\n")
            f.write(f"Nominal k_true [1/m2]: {case['k_nominal']:.6e}\n")
            f.write(f"Realized k_true (= N/A) [1/m2]: {true_rock_density:.6e}\n")
            f.write(f"Observation Area [m2]: {overall_area}\n")
            f.write(f"D_max Mode: {D_MAX_MODE}\n")
            f.write(f"D_max [m]: {D_max_used:.4f}\n")
            f.write(f"D_min [m]: {config.ROCK_SIZE_MIN:.4f}\n")
            f.write(f"Total Iterations: {NUM_ITERATIONS}\n")

        all_range_results = []
        all_moving_results = []
        overall_csfd_stats = []

        print("  -> 並列処理でイテレーションを実行中...")
        
        tasks = [(i, r_true, total_rocks, output_dir, overall_area, true_rock_density) for i in range(1, NUM_ITERATIONS + 1)]
        results = [None] * NUM_ITERATIONS

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_index = {executor.submit(process_iteration, task): i for i, task in enumerate(tasks)}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_index), total=NUM_ITERATIONS, desc="進捗", unit="iter"):
                original_idx = future_to_index[future]
                try:
                    results[original_idx] = future.result()
                except Exception as exc:
                    print(f"\nイテレーションでエラーが発生しました: {exc}")

        for res in results:
            if res is not None:
                csfd_stats, analysis_range, analysis_moving = res
                overall_csfd_stats.append(csfd_stats)
                all_range_results.append(analysis_range)
                all_moving_results.append(analysis_moving)

        print("  -> 統計データの集計とプロットを生成中...")
        
        analyzer = Analyzer()
        
        csfd_stats_df = pd.DataFrame(overall_csfd_stats)
        csfd_stats_df = csfd_stats_df.sort_values('iteration')
        csfd_stats_df.to_csv(f"{output_dir}/csfd_fits_stats.csv", index=False)
        analyzer.plot_csfd_stats(csfd_stats_df, f"{output_dir}/RSFD_comparison_stats", r_true, config)
        analyzer.plot_csfd_count_stats(csfd_stats_df, f"{output_dir}/RSFD_count_stats", r_true, config)

        r_true_mean = csfd_stats_df['r_true'].mean()
        r_true_std  = csfd_stats_df['r_true'].std()
        k_true_mean = csfd_stats_df['k_true'].mean()
        k_true_std  = csfd_stats_df['k_true'].std()

        r_det_mean = csfd_stats_df['r_det'].mean()
        r_det_std  = csfd_stats_df['r_det'].std()
        k_det_mean = csfd_stats_df['k_det'].mean()
        k_det_std  = csfd_stats_df['k_det'].std()

        with open(f"{output_dir}/overall_rsfd_stats.txt", "w") as f:
            f.write(f"=== Overall RSFD (0-{config.MAX_DEPTH}m) {NUM_ITERATIONS} Iterations Statistics ===\n\n")
            f.write(f"[True Rocks (Generated)]\n")
            f.write(f"  r = {r_true_mean:.4f} +/- {r_true_std:.4f}\n")
            f.write(f"  k = {k_true_mean:.4e} +/- {k_true_std:.4e}\n\n")
            f.write(f"[Detected Rocks (Apparent)]\n")
            f.write(f"  r_apparent = {r_det_mean:.4f} +/- {r_det_std:.4f}\n")
            f.write(f"  k_apparent = {k_det_mean:.4e} +/- {k_det_std:.4e}\n")

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
            except json.JSONDecodeError:
                summary_data = {}
        else:
            summary_data = {}

        r_key = str(r_true)
        case_key = case['json_key']

        if r_key not in summary_data:
            summary_data[r_key] = {}

        summary_data[r_key][case_key] = {
            "input_mode": INPUT_MODE,
            "N_true": int(total_rocks),
            "k_true_input": float(case['k_nominal']),
            "k_true_realized": float(true_rock_density),
            "r_true_mean": float(r_true_mean),
            "r_true_std": float(r_true_std),
            "k_true_mean": float(k_true_mean),
            "k_true_std": float(k_true_std),
            "r_apparent_mean": float(r_det_mean),
            "r_apparent_std": float(r_det_std),
            "k_apparent_mean": float(k_det_mean),
            "k_apparent_std": float(k_det_std)
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4)
        print(f"  -> {json_path} を更新しました。")

        combined_range = pd.concat(all_range_results)
        stats_range = combined_range.groupby('depth_range').agg(
            r_apparent_mean=('r_apparent', 'mean'),
            r_apparent_std=('r_apparent', 'std'),
            k_apparent_mean=('k_apparent', 'mean'),
            k_apparent_std=('k_apparent', 'std'),
            detection_rate_mean=('detection_rate', 'mean'),
            detection_rate_std=('detection_rate', 'std'),
            rock_density_mean=('rock_density', 'mean'),
            rock_density_std=('rock_density', 'std')
        ).reset_index()

        stats_range.to_csv(f"{output_dir}/aggregated_stats_range.csv", index=False)

        analyzer.plot_depth_analysis_stats(stats_range, f"{output_dir}/powerlaw_exp_range_stats", r_true, x_col='depth_range', xlabel='Depth Range [m]')
        analyzer.plot_k_analysis_stats(stats_range, f"{output_dir}/powerlaw_k_range_stats", true_rock_density, x_col='depth_range', xlabel='Depth Range [m]')
        analyzer.plot_detection_rate_stats(stats_range, f"{output_dir}/detection_rate_range_stats", x_col='depth_range', xlabel='Depth Range [m]')

        combined_moving = pd.concat(all_moving_results)
        stats_moving = combined_moving.groupby('depth_center').agg(
            r_apparent_mean=('r_apparent', 'mean'),
            r_apparent_std=('r_apparent', 'std'),
            k_apparent_mean=('k_apparent', 'mean'),
            k_apparent_std=('k_apparent', 'std'),
            detection_rate_mean=('detection_rate', 'mean'),
            detection_rate_std=('detection_rate', 'std'),
            rock_density_mean=('rock_density', 'mean'),
            rock_density_std=('rock_density', 'std')
        ).reset_index()

        stats_moving.to_csv(f"{output_dir}/aggregated_stats_moving.csv", index=False)

        analyzer.plot_depth_analysis_stats(stats_moving, f"{output_dir}/powerlaw_exp_moving_stats", r_true, x_col='depth_center', xlabel='Depth Center [m]')
        analyzer.plot_k_analysis_stats(stats_moving, f"{output_dir}/powerlaw_k_moving_stats", true_rock_density, x_col='depth_center', xlabel='Depth Center [m]')

        print(f"=== {case['label']} の処理完了 ===\n")

    print("全ての反復処理と統計出力が完了しました。")

if __name__ == "__main__":
    main()