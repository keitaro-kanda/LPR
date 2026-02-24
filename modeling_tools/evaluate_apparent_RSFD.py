import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. 設定クラス ---
class RadarConfig:
    """
    シミュレーションの全パラメータを管理するクラス
    """
    def __init__(self, total_rocks=10000):
        # --- A. レーダーシステム ---
        self.FREQ = 500e6  # 500 MHz
        self.C_0 = 3e8
        
        self.TX_POWER_DBM = 62.0
        self.ANTENNA_GAIN_DBI = -7.5
        self.SYSTEM_LOSS_DB = 0
        self.NOISE_FLOOR_DBM = -90.0
        
        # --- B. 環境・媒質パラメータ ---
        self.EPSILON_R_REG = 3.0
        self.LOSS_TANGENT = 0.004
        self.EPSILON_R_ROCK = 9.0
        
        # --- C. シミュレーション空間設定 ---
        self.MAX_DEPTH = 12.0
        self.AREA_SIZE_M2 = 4500.0 # 幅3 m x 奥行き1500 mのエリア
        self.ROCK_SIZE_MIN = 0.01
        self.ROCK_SIZE_MAX = 1.0
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
        n1 = np.sqrt(self.EPSILON_R_REG)
        n2 = np.sqrt(self.EPSILON_R_ROCK)
        return ((n1 - n2) / (n1 + n2))**2

# --- 2. 物理モデルクラス ---
class RockModel:
    """
    岩石の生成とレーダー方程式の適用を行うクラス
    """
    def generate_rocks(self, r_true, config, quiet=False):
        if not quiet:
            print(f"岩石を生成中... (True Slope = -{r_true}, N={config.TOTAL_ROCKS})")
        
        u = np.random.rand(config.TOTAL_ROCKS)
        
        D_min = config.ROCK_SIZE_MIN
        D_max = config.ROCK_SIZE_MAX
        
        term1 = D_max ** (-r_true)
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
        
        sigma_optical = config.reflection_coeff * np.pi * (diameter_array / 2.0)**2
        
        sigma_opt_at_bound = config.reflection_coeff * np.pi * (boundary_size / 2.0)**2
        k_rayleigh = sigma_opt_at_bound / (boundary_size ** 6)
        sigma_rayleigh = k_rayleigh * (diameter_array ** 6)
        
        sigma = np.where(diameter_array >= boundary_size, sigma_optical, sigma_rayleigh)
        return 10 * np.log10(sigma)

    def apply_radar_equation(self, rocks_df, config, quiet=False):
        if not quiet:
            print("レーダーシミュレーション実行中...")
        
        rcs_db = self.calculate_rcs_db(rocks_df['diameter'].values, config)
        spread_loss_db = 40 * np.log10(rocks_df['depth'].values + 0.1)
        attenuation_loss_db = 2 * config.attenuation_db_m * rocks_df['depth'].values
        
        const_gain = config.TX_POWER_DBM + 2 * config.ANTENNA_GAIN_DBI - config.SYSTEM_LOSS_DB
        wavelength_factor = 10 * np.log10(config.lambda_0**2 / ((4 * np.pi)**3))
        
        received_power = (const_gain + wavelength_factor + rcs_db 
                          - spread_loss_db - attenuation_loss_db)
        
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
            return np.nan, np.nan, [], []
            
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
        
        for d in depth_ranges:
            subset_detected = detected_df[
                (detected_df['is_detected'] == True) & 
                (detected_df['depth'] <= d)
            ]
            subset_all = detected_df[detected_df['depth'] <= d]
            
            count_detected = len(subset_detected)
            count_all = len(subset_all)
            
            area = d * 1500.0
            
            if count_detected > 5:
                slope, intercept, _, _ = self.calculate_slope(subset_detected['diameter'].values, area)
                r_apparent = -slope
                k_apparent = (10**intercept) * (config.ROCK_SIZE_MIN**slope)
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
        [新規] 指定した窓幅(window_size)と移動幅(step_ratio)での移動窓解析
        """
        if not quiet:
            print("深さ方向の移動窓解析(Moving Window)を実行中...")
        results = []
        step_size = window_size * step_ratio
        
        d_starts = np.arange(0.0, config.MAX_DEPTH - window_size + 1e-5, step_size)
        area = window_size * 1500.0 # 窓内の面積は常に一定
        
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
                slope, intercept, _, _ = self.calculate_slope(subset_detected['diameter'].values, area)
                r_apparent = -slope
                k_apparent = (10**intercept) * (config.ROCK_SIZE_MIN**slope)
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


    # === 個別出力用プロットメソッド ===
    def plot_csfd(self, detected_df, all_rocks_df, output_prefix, r_true, config):
        plt.figure(figsize=(10, 8))
        
        # 全岩石が分布する面積で規格化
        overall_area = config.MAX_DEPTH * 1500.0

        diameters_true = all_rocks_df['diameter'].values
        if len(diameters_true) > 0:
            slope_true, intercept_true, x_log_true, y_log_true = self.calculate_slope(diameters_true, overall_area)
            plt.scatter(10**x_log_true, 10**y_log_true, s=20, color='gray', label='Generated Rocks', marker='D', alpha=0.7)
            if not np.isnan(slope_true):
                x_fit_true = np.linspace(min(x_log_true), max(x_log_true), 100)
                y_fit_true = slope_true * x_fit_true + intercept_true
                d_min = config.ROCK_SIZE_MIN
                k_true = (10**intercept_true) * (d_min**slope_true)
                plt.plot(10**x_fit_true, 10**y_fit_true, 'k--', linewidth=2.0, 
                         label=f'True Fit (r={-slope_true:.2f}, k={k_true:.2e})')

        diameters_det = detected_df[detected_df['is_detected'] == True]['diameter'].values
        if len(diameters_det) > 0:
            slope_det, intercept_det, x_log_det, y_log_det = self.calculate_slope(diameters_det, overall_area)
            plt.scatter(10**x_log_det, 10**y_log_det, s=20, color='blue', label='Detected Rocks', marker='o')
            if not np.isnan(slope_det):
                x_fit_det = np.linspace(min(x_log_det), max(x_log_det), 100)
                y_fit_det = slope_det * x_fit_det + intercept_det
                k_det = (10**intercept_det) * (d_min**slope_det)
                plt.plot(10**x_fit_det, 10**y_fit_det, 'r-', linewidth=2.5, 
                         label=f'Apparent Fit (r={-slope_det:.2f}, k={k_det:.2e})')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Diameter [m]', fontsize=18)
        plt.ylabel('Cumulative Rock Density N(>D) [1/m²]', fontsize=18)
        # plt.title(f'Comparison of True vs Apparent RSFD\n(Input True r = {r_true})', fontsize=18)
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
        # plt.title('Depth-wise Apparent Slope Analysis', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.ylim(0, r_true + 1)
        plt.legend(fontsize=16)
        plt.grid(True, ls='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_k_analysis(self, analysis_df, output_prefix, true_k, x_col='depth_range', xlabel='Depth Range [m]'):
        plt.figure(figsize=(10, 8))
        # k のプロット色を赤に変更
        plt.plot(analysis_df[x_col], analysis_df['k_apparent'], marker='^', linestyle='-', color='red', label='Apparent k')
        plt.axhline(y=true_k, color='k', linestyle='--', label=f'True k: {true_k:.2e}')
        
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel('Apparent Density Coefficient k [1/m²]', fontsize=18)
        # plt.title('Depth-wise Apparent Coefficient k', fontsize=18)
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
        # plt.title('Detection Rate vs Depth Range', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        
        plt.ylim(0, 1.05) 
        plt.legend(fontsize=16)
        plt.grid(True, ls='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_rock_density(self, analysis_df, output_prefix, true_rock_density, x_col='depth_range', xlabel='Depth Range [m]'):
        plt.figure(figsize=(10, 8))
        plt.plot(analysis_df[x_col], analysis_df['rock_density'], marker='s', linestyle='-', color='r', label='Detected Rock Density')
        plt.axhline(y=true_rock_density, color='k', linestyle='--', label=f'True Rock Density: {true_rock_density:.2e}')
        
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel('Detected Rock Density [1/m²]', fontsize=18)
        # plt.title('Detected Rock Density vs Depth Range', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        
        plt.legend(fontsize=16)
        plt.grid(True, ls='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    # === 統計出力用プロットメソッド (平均とエラー範囲) ===
    def plot_csfd_stats(self, csfd_stats_df, output_prefix, r_true, config):
        """
        全イテレーションの近似直線を平均化し、標準偏差の帯とともに描画する
        """
        plt.figure(figsize=(10, 8))

        # 共通のX軸(Diameter)の対数スケールを作成
        x_min_log = np.log10(config.ROCK_SIZE_MIN)
        x_max_log = np.log10(config.ROCK_SIZE_MAX)
        x_log_common = np.linspace(x_min_log, x_max_log, 100)
        x_common = 10**x_log_common

        # --- 真の岩石の近似直線の平均と標準偏差 ---
        y_true_all = []
        for _, row in csfd_stats_df.iterrows():
            if not np.isnan(row['slope_true']):
                y_true_all.append(row['slope_true'] * x_log_common + row['intercept_true'])
        
        d_min = config.ROCK_SIZE_MIN

        if y_true_all:
            y_true_all = np.array(y_true_all)
            y_true_mean = np.mean(y_true_all, axis=0)
            y_true_std = np.std(y_true_all, axis=0)
            
            mean_slope_true = np.mean(csfd_stats_df['slope_true'])
            mean_intercept_true = np.mean(csfd_stats_df['intercept_true'])
            mean_k_true = (10**mean_intercept_true) * (d_min**mean_slope_true)
            
            plt.plot(x_common, 10**y_true_mean, 'k--', linewidth=2.0, 
                     label=f'Mean True Fit (r={-mean_slope_true:.2f}, k={mean_k_true:.2e})')
            plt.fill_between(x_common, 10**(y_true_mean - y_true_std), 10**(y_true_mean + y_true_std), color='gray', alpha=0.3)

        # --- 検出された岩石の近似直線の平均と標準偏差 ---
        y_det_all = []
        for _, row in csfd_stats_df.iterrows():
            if not np.isnan(row['slope_det']):
                y_det_all.append(row['slope_det'] * x_log_common + row['intercept_det'])

        if y_det_all:
            y_det_all = np.array(y_det_all)
            y_det_mean = np.mean(y_det_all, axis=0)
            y_det_std = np.std(y_det_all, axis=0)
            
            mean_slope_det = np.mean(csfd_stats_df['slope_det'])
            mean_intercept_det = np.mean(csfd_stats_df['intercept_det'])
            mean_k_det = (10**mean_intercept_det) * (d_min**mean_slope_det)
            
            plt.plot(x_common, 10**y_det_mean, 'r-', linewidth=2.5, 
                     label=f'Mean Apparent Fit (r={-mean_slope_det:.2f}, k={mean_k_det:.2e})')
            plt.fill_between(x_common, 10**(y_det_mean - y_det_std), 10**(y_det_mean + y_det_std), color='red', alpha=0.3, label='Apparent Fit ±1 Std Dev')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Diameter [m]', fontsize=18)
        plt.ylabel('Cumulative Rock Density N(>D) [1/m²]', fontsize=18)
        # plt.title(f'Statistical True vs Apparent RSFD\n(Input True r = {r_true}, {len(csfd_stats_df)} Iterations)', fontsize=18)
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
        # plt.title('Statistical Depth-wise Apparent Slope Analysis', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
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

        # k のプロット色を赤に変更
        plt.plot(x, y_mean, marker='^', linestyle='-', color='red', label='Mean Apparent k')
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='red', alpha=0.3, label='±1 Std Dev')
        plt.axhline(y=true_k, color='k', linestyle='--', label=f'True k: {true_k:.2e}')
        
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel('Apparent Density Coefficient k [1/m²]', fontsize=18)
        # plt.title('Statistical Depth-wise Apparent Coefficient k', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
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
        # plt.title('Statistical Detection Rate vs Depth Range', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        
        plt.ylim(0, 1.05) 
        plt.legend(fontsize=16)
        plt.grid(True, ls='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_rock_density_stats(self, stats_df, output_prefix, true_rock_density, x_col='depth_range', xlabel='Depth Range [m]'):
        plt.figure(figsize=(10, 8))

        x = stats_df[x_col]
        y_mean = stats_df['rock_density_mean']
        y_std = stats_df['rock_density_std']

        plt.plot(x, y_mean, marker='s', linestyle='-', color='r', label='Mean Detected Density')
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='r', alpha=0.3, label='±1 Std Dev')
        plt.axhline(y=true_rock_density, color='k', linestyle='--', label=f'True Rock Density: {true_rock_density:.2e}')
        
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel('Detected Rock Density [1/m²]', fontsize=18)
        # plt.title('Statistical Detected Rock Density vs Depth Range', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        
        plt.legend(fontsize=16)
        plt.grid(True, ls='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

# --- 4. メイン処理 ---
def main():
    print("--- 岩石見逃しモデル 統計シミュレーション ---")
    try:
        r_input_str = input("真のべき指数(r)を入力してください (例: 1.0): ")
        r_true = float(r_input_str)
    except ValueError:
        print("数値ではない入力です。デフォルト値 1.0 を使用します。")
        r_true = 1.0

    # パス設定
    base_dir = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD' 
    
    # 計算パラメータ
    rock_counts = [100, 500, 1000, 10000]
    NUM_ITERATIONS = 20  # 反復回数
    
    for total_rocks in rock_counts:
        print(f"\n=== 岩石数: {total_rocks} のシミュレーションを開始 ({NUM_ITERATIONS}回) ===")
        output_dir = f"{base_dir}/r_{r_true}/{total_rocks}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"出力ディレクトリ: {output_dir}")

        config = RadarConfig(total_rocks=total_rocks)
        model = RockModel()
        analyzer = Analyzer()

        overall_area = config.MAX_DEPTH * 1500.0
        true_rock_density = config.TOTAL_ROCKS / overall_area

        with open(f"{output_dir}/parameters.txt", "w") as f:
            f.write(f"True Slope (r): {r_true}\n")
            f.write(f"Total Rocks: {config.TOTAL_ROCKS}\n")
            f.write(f"True Rock Density [1/m2]: {true_rock_density:.6f}\n")
            f.write(f"Total Iterations: {NUM_ITERATIONS}\n")

        all_range_results = []
        all_moving_results = []
        overall_csfd_stats = [] 

        # --- 反復計算の実行 ---
        for i in range(1, NUM_ITERATIONS + 1):
            if i % 5 == 0 or i == 1:
                print(f"  -> イテレーション {i}/{NUM_ITERATIONS} 実行中...")
            
            iter_dir = f"{output_dir}/iter_{i:02d}"
            os.makedirs(iter_dir, exist_ok=True)

            all_rocks_df = model.generate_rocks(r_true, config, quiet=True)
            all_rocks_df.to_csv(f"{iter_dir}/truth_rocks.csv", index=False)

            detected_df = model.apply_radar_equation(all_rocks_df, config, quiet=True)
            detected_df.to_csv(f"{iter_dir}/simulated_detection.csv", index=False)
            
            # RSFD統計プロット情報
            slope_true, intercept_true, _, _ = analyzer.calculate_slope(all_rocks_df['diameter'].values, overall_area)
            diameters_det = detected_df[detected_df['is_detected'] == True]['diameter'].values
            slope_det, intercept_det, _, _ = analyzer.calculate_slope(diameters_det, overall_area)
            
            overall_csfd_stats.append({
                'iteration': i,
                'slope_true': slope_true,
                'intercept_true': intercept_true,
                'slope_det': slope_det,
                'intercept_det': intercept_det
            })

            analyzer.plot_csfd(detected_df, all_rocks_df, f"{iter_dir}/csfd_comparison", r_true, config)

            # --- A. 既存の深さ範囲解析 (Range) ---
            analysis_range = analyzer.run_depth_analysis(detected_df, config, step=1.0, quiet=True)
            analysis_range['iteration'] = i 
            analysis_range.to_csv(f"{iter_dir}/depth_analysis_range.csv", index=False)
            
            analyzer.plot_depth_analysis(analysis_range, f"{iter_dir}/powerlaw_exp_range", r_true, x_col='depth_range', xlabel='Depth Range [m]')
            analyzer.plot_k_analysis(analysis_range, f"{iter_dir}/powerlaw_k_range", true_rock_density, x_col='depth_range', xlabel='Depth Range [m]')
            analyzer.plot_detection_rate(analysis_range, f"{iter_dir}/detection_rate_range", x_col='depth_range', xlabel='Depth Range [m]')
            analyzer.plot_rock_density(analysis_range, f"{iter_dir}/rock_density_range", true_rock_density, x_col='depth_range', xlabel='Depth Range [m]')
            
            all_range_results.append(analysis_range)

            # --- B. 新規の移動窓解析 (Moving Window) ---
            analysis_moving = analyzer.run_moving_window_analysis(detected_df, config, window_size=2.0, step_ratio=0.2, quiet=True)
            analysis_moving['iteration'] = i 
            analysis_moving.to_csv(f"{iter_dir}/depth_analysis_moving.csv", index=False)
            
            analyzer.plot_depth_analysis(analysis_moving, f"{iter_dir}/powerlaw_exp_moving", r_true, x_col='depth_center', xlabel='Depth Center [m]')
            analyzer.plot_k_analysis(analysis_moving, f"{iter_dir}/powerlaw_k_moving", true_rock_density, x_col='depth_center', xlabel='Depth Center [m]')
            # 移動窓解析の検出率プロットは省略
            analyzer.plot_rock_density(analysis_moving, f"{iter_dir}/rock_density_moving", true_rock_density, x_col='depth_center', xlabel='Depth Center [m]')

            all_moving_results.append(analysis_moving)

        # --- 統計処理とプロットの実行 ---
        print("  -> 統計データの集計とプロットを生成中...")
        
        # 1. 全体RSFDの統計プロット
        csfd_stats_df = pd.DataFrame(overall_csfd_stats)
        csfd_stats_df.to_csv(f"{output_dir}/csfd_fits_stats.csv", index=False)
        analyzer.plot_csfd_stats(csfd_stats_df, f"{output_dir}/RSFD_comparison_stats", r_true, config)

        # 2. 既存の深さ範囲解析の統計
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
        analyzer.plot_rock_density_stats(stats_range, f"{output_dir}/rock_density_range_stats", true_rock_density, x_col='depth_range', xlabel='Depth Range [m]')

        # 3. 新規の移動窓解析の統計
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
        # 移動窓解析の検出率プロットは省略
        analyzer.plot_rock_density_stats(stats_moving, f"{output_dir}/rock_density_moving_stats", true_rock_density, x_col='depth_center', xlabel='Depth Center [m]')


        print(f"=== 岩石数: {total_rocks} の処理完了 ===\n")

    print("全ての反復処理と統計出力が完了しました。")

if __name__ == "__main__":
    main()