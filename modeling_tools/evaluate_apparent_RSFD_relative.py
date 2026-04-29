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
        self.NOISE_FLOOR_DBM = -90.0 
        self.H_ANTENNA = 0.3 # アンテナ地上高 [m]
        
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
        self.ROCK_SIZE_MIN = 0.06
        self.TOTAL_ROCKS = total_rocks

    @property
    def lambda_0(self):
        return self.C_0 / self.FREQ

    @property
    def lambda_g(self):
        return self.lambda_0 / np.sqrt(self.EPSILON_R_REG)

    @property
    def attenuation_db_m(self):
        omega = 2 * np.pi * self.FREQ
        epsilon_0 = 8.854e-12
        mu_0 = 4 * np.pi * 1e-7
        epsilon = self.EPSILON_R_REG * epsilon_0
        mu = mu_0
        term1 = omega * (np.sqrt(mu * epsilon))
        term2 = np.sqrt(0.5 * (np.sqrt(1 + self.LOSS_TANGENT**2) - 1))
        return (term1 * term2) * 8.686

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
    def surface_rcs(self):
        # 表面のレーダー断面積 (RCS) の計算
        sigma_surf = self.surface_reflection_coeff**2 * np.pi**3 * self.H_ANTENNA**2
        return sigma_surf

# --- 2. 物理モデルクラス ---
class RockModel:
    """
    岩石の生成とレーダー方程式の適用を行うクラス
    """
    def generate_rocks(self, r_true, config, quiet=False):
        D_min = config.ROCK_SIZE_MIN
        
        # Nとrから導かれる自然な最大岩石サイズ (D_max_auto) を算出
        D_max_auto = D_min * (config.TOTAL_ROCKS ** (1.0 / r_true))

        if not quiet:
            print(f"岩石を生成中... (True Slope = -{r_true}, N={config.TOTAL_ROCKS})")
            print(f"  -> 自然な最大岩石サイズ (D_max_auto): {D_max_auto:.2f} m")
        
        # 0から1の範囲を TOTAL_ROCKS 個で等間隔に分割した配列を作成（分位数）
        # u = (np.arange(config.TOTAL_ROCKS) + 0.5) / config.TOTAL_ROCKS
        # サイズが常に大きい順（または小さい順）に並んでしまうのを防ぐため、配列をシャッフルする
        # np.random.shuffle(u)

        # 均等割りではなく、一様分布からランダムにサンプリングする
        u = np.random.uniform(0.0, 1.0, size=config.TOTAL_ROCKS)
        
        # 算出した自然な最大サイズを上限として適用
        term1 = D_max_auto ** (-r_true)
        term2 = D_min ** (-r_true)
        
        diameters = ( (term1 - term2) * u + term2 ) ** (-1.0 / r_true)
        depths = np.random.uniform(0, config.MAX_DEPTH, size=len(diameters))
        
        df = pd.DataFrame({
            'diameter': diameters,
            'depth': depths
        })
        return df

    def calculate_rcs_db(self, diameter_array, config):
        lambda_g = config.lambda_g
        boundary_size = lambda_g / 2.0
        
        sigma_optical = config.reflection_coeff**2 * np.pi * (diameter_array / 2.0)**2
        
        sigma_opt_at_bound = config.reflection_coeff**2 * np.pi * (boundary_size / 2.0)**2
        k_rayleigh = sigma_opt_at_bound / (boundary_size ** 6)
        sigma_rayleigh = k_rayleigh * (diameter_array ** 6)
        
        sigma = np.where(diameter_array >= boundary_size, sigma_optical, sigma_rayleigh)
        return 10 * np.log10(np.maximum(sigma, 1e-15))

    def apply_radar_equation(self, rocks_df, config, quiet=False):
        if not quiet:
            print("レーダーシミュレーション実行中 (相対値モデル)...")
        
        # 岩石のRCS (10 * log10(sigma(r)))
        rcs_db = self.calculate_rcs_db(rocks_df['diameter'].values, config)
        
        # 距離の計算
        R_surf = config.H_ANTENNA
        R_sub = rocks_df['depth'].values + R_surf
        
        # 1. 幾何学的減衰項: 40 * log10(R_surf / R_sub)
        spread_term = 40 * np.log10(R_surf / R_sub)
        
        # 2. 媒質による減衰項 (往復): -2 * attenuation_db_m * depth
        attenuation_term = -2 * config.attenuation_db_m * rocks_df['depth'].values
        
        # 3. 反射強度の比率項: 10 * log10(sigma(r) / sigma_surf)
        # 対数の引き算として rcs_db - 10*log10(sigma_surf) で計算
        sigma_surf_db = 10 * np.log10(config.surface_rcs)
        rcs_ratio_term = rcs_db - sigma_surf_db
        
        # 4. Random項 (正規分布)
        random_term = np.random.normal(0, config.SIGMA_RANDOM_DB, size=len(rocks_df))
        
        # 規格化強度の計算 (S_norm)
        received_power = spread_term + attenuation_term + rcs_ratio_term + random_term
        
        # 閾値判定
        is_detected = received_power > config.NOISE_FLOOR_DBM
        
        result_df = rocks_df.copy()
        result_df['rcs_db'] = rcs_db
        result_df['received_power'] = received_power
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
        
        slope, intercept = np.polyfit(x_log, y_log, 1)
        
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
        
        for d in depth_ranges:
            subset_detected = detected_df[
                (detected_df['is_detected'] == True) & 
                (detected_df['depth'] <= d)
            ]
            subset_all = detected_df[detected_df['depth'] <= d]
            
            count_detected = len(subset_detected)
            count_all = len(subset_all)
            
            # 深さごとの面積を算出
            area = d * length
            
            if count_detected > 5:
                # cm単位に変換して傾きを計算
                diameters_cm = subset_detected['diameter'].values * 100.0
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
        
        for d_start in d_starts:
            d_end = d_start + window_size
            d_center = d_start + window_size / 2.0
            
            subset_detected = detected_df[
                (detected_df['is_detected'] == True) & 
                (detected_df['depth'] > d_start) & 
                (detected_df['depth'] <= d_end)
            ]
            subset_all = detected_df[
                (detected_df['depth'] > d_start) & 
                (detected_df['depth'] <= d_end)
            ]
            
            count_detected = len(subset_detected)
            count_all = len(subset_all)
            
            if count_detected > 5:
                # cm単位に変換して傾きを計算
                diameters_cm = subset_detected['diameter'].values * 100.0
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
        色で信号強度（received_power）をマッピングする散布図
        """
        plt.figure(figsize=(10, 8))
        
        sc = plt.scatter(df['depth'], df['diameter'], 
                         c=df['received_power'], cmap='jet', 
                         alpha=0.8, s=20, edgecolors='none', vmax=0, vmin=-120)
        
        cbar = plt.colorbar(sc)
        cbar.set_label('Received Power $S_{norm}$ [dB]', fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.axhline(config.NOISE_FLOOR_DBM, color='red', linestyle='--', linewidth=2)

        plt.xlabel('Depth [m]', fontsize=18)
        plt.ylabel('Diameter [m]', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
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
        diameters_det_cm = detected_df[detected_df['is_detected'] == True]['diameter'].values * 100.0
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
        plt.xlabel('Diameter [cm]', fontsize=18)
        plt.xticks([6, 10, 20, 50, 100, 200, 500], [6, 10, 20, 50, 100, 200, 500], fontsize=16)
        plt.xlim(d_min_cm * 0.8, max(diameters_true_cm.max(), diameters_det_cm.max()) * 1.2)
        plt.ylabel('Cumulative Rock Density N(>D) [1/m²]', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
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
        plt.savefig(f"{output_prefix}.pdf")
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
        plt.savefig(f"{output_prefix}.pdf")
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
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    # === 統計出力用プロットメソッド (エラー範囲、±ラベルの追加) ===
    def plot_csfd_stats(self, csfd_stats_df, output_prefix, r_true, config):
        plt.figure(figsize=(10, 8))

        d_min_cm = config.ROCK_SIZE_MIN * 100.0
        N_total = config.TOTAL_ROCKS
        D_max_auto_cm = d_min_cm * (N_total ** (1.0 / r_true))

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
        D_max_auto_cm = d_min_cm * (N_total ** (1.0 / r_true))

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
    diameters_det_cm = detected_df[detected_df['is_detected'] == True]['diameter'].values * 100.0
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
def main():
    print("--- 岩石見逃しモデル 統計シミュレーション (相対値モデル) ---")
    
    try:
        r_input_str = input("真のべき指数(r)を入力してください (例: 1.0): ")
        r_true = float(r_input_str)
    except ValueError:
        print("数値ではない入力です。デフォルト値 1.0 を使用します。")
        r_true = 1.0
    
    try:
        rock_counts_str = input("計算する岩石数をカンマ区切りで入力してください (例: 100,500,1000,5000) を入力: ")
        rock_counts = [int(x.strip()) for x in rock_counts_str.split(',')]
    except ValueError:
        print("数値の入力ではありません。デフォルト値を使用します。")
        rock_counts = [100, 500, 1000, 5000]

    base_dir = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD_relative' 
    os.makedirs(base_dir, exist_ok=True)
    
    json_path = os.path.join(base_dir, 'rsfd_summary.json')

    NUM_ITERATIONS = 100 
    
    for total_rocks in rock_counts:
        print(f"\n=== 岩石数: {total_rocks} のシミュレーションを開始 ({NUM_ITERATIONS}回) ===")
        
        config = RadarConfig(total_rocks=total_rocks)
        
        # --- 最大岩石サイズの事前チェック (6.0 m 制限) ---
        D_max_auto = config.ROCK_SIZE_MIN * (total_rocks ** (1.0 / r_true))
        if D_max_auto > 6.0:
            print(f"  -> [警告] 計算上の最大岩石サイズ ({D_max_auto:.2f} m) が 6.0 m を超えるため、計算を中止します。")
            
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
            n_key = str(total_rocks)
            if r_key not in summary_data:
                summary_data[r_key] = {}
            
            summary_data[r_key][n_key] = "Calculation stop"
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=4)
            print(f"  -> {json_path} にストップを記録し、次の条件へスキップします。")
            continue
        # --------------------------------------------------------

        output_dir = f"{base_dir}/r_{r_true}/{total_rocks}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"出力ディレクトリ: {output_dir}")

        overall_area = config.AREA_SIZE_M2
        true_rock_density = config.TOTAL_ROCKS / overall_area

        with open(f"{output_dir}/parameters.txt", "w") as f:
            f.write(f"True Slope (r): {r_true}\n")
            f.write(f"Total Rocks: {config.TOTAL_ROCKS}\n")
            f.write(f"True Rock Density [1/m2]: {true_rock_density:.6f}\n")
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
        n_key = str(total_rocks)

        if r_key not in summary_data:
            summary_data[r_key] = {}

        summary_data[r_key][n_key] = {
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

        print(f"=== 岩石数: {total_rocks} の処理完了 ===\n")

    print("全ての反復処理と統計出力が完了しました。")

if __name__ == "__main__":
    main()