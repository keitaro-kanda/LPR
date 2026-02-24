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
    def calculate_slope(self, diameters):
        if len(diameters) < 2:
            return np.nan, np.nan, [], []
            
        sorted_d = np.sort(diameters)
        y_cumulative = np.arange(len(sorted_d), 0, -1)
        
        x_log = np.log10(sorted_d)
        y_log = np.log10(y_cumulative)
        
        slope, intercept = np.polyfit(x_log, y_log, 1)
        
        return slope, intercept, x_log, y_log

    def run_depth_analysis(self, detected_df, max_depth, step=1.0, quiet=False):
        if not quiet:
            print("深さ方向の感度解析を実行中...")
        results = []
        depth_ranges = np.arange(1.0, max_depth + 0.1, step)
        
        for d in depth_ranges:
            subset_detected = detected_df[
                (detected_df['is_detected'] == True) & 
                (detected_df['depth'] <= d)
            ]
            subset_all = detected_df[detected_df['depth'] <= d]
            
            count_detected = len(subset_detected)
            count_all = len(subset_all)
            
            if count_detected > 5:
                slope, _, _, _ = self.calculate_slope(subset_detected['diameter'].values)
                r_apparent = -slope
            else:
                r_apparent = np.nan
            
            detection_rate = count_detected / count_all if count_all > 0 else 0.0
            
            area = d * 1500.0
            rock_density = count_detected / area if area > 0 else 0.0
            
            results.append({
                'depth_range': d,
                'r_apparent': r_apparent,
                'count_detected': count_detected,
                'count_all': count_all,
                'detection_rate': detection_rate,
                'rock_density': rock_density
            })
            
        return pd.DataFrame(results)

    # === 個別出力用プロットメソッド ===
    def plot_csfd(self, detected_df, all_rocks_df, output_prefix, r_true):
        plt.figure(figsize=(10, 8))

        diameters_true = all_rocks_df['diameter'].values
        if len(diameters_true) > 0:
            slope_true, intercept_true, x_log_true, y_log_true = self.calculate_slope(diameters_true)
            plt.scatter(10**x_log_true, 10**y_log_true, s=20, color='gray', label='Generated Rocks', marker='D', alpha=0.7)
            if not np.isnan(slope_true):
                x_fit_true = np.linspace(min(x_log_true), max(x_log_true), 100)
                y_fit_true = slope_true * x_fit_true + intercept_true
                plt.plot(10**x_fit_true, 10**y_fit_true, 'k--', linewidth=2.0, 
                         label=f'True Fit (r = {-slope_true:.2f})')

        diameters_det = detected_df[detected_df['is_detected'] == True]['diameter'].values
        if len(diameters_det) > 0:
            slope_det, intercept_det, x_log_det, y_log_det = self.calculate_slope(diameters_det)
            plt.scatter(10**x_log_det, 10**y_log_det, s=20, color='blue', label='Detected Rocks', marker='o')
            if not np.isnan(slope_det):
                x_fit_det = np.linspace(min(x_log_det), max(x_log_det), 100)
                y_fit_det = slope_det * x_fit_det + intercept_det
                plt.plot(10**x_fit_det, 10**y_fit_det, 'r-', linewidth=2.5, 
                         label=f'Apparent Fit (r = {-slope_det:.2f})')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Diameter [m]', fontsize=18)
        plt.ylabel('Cumulative Number N(>D)', fontsize=18)
        plt.title(f'Comparison of True vs Apparent RSFD\n(Input True r = {r_true})', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_depth_analysis(self, analysis_df, output_prefix, r_true):
        plt.figure(figsize=(10, 8))
        plt.plot(analysis_df['depth_range'], analysis_df['r_apparent'], marker='o', linestyle='-', color='blue', label='Apparent r')
        plt.axhline(y=r_true, color='k', linestyle='--', label='True r')
        
        plt.xlabel('Depth Range [m]', fontsize=18)
        plt.ylabel('Apparent Slope r', fontsize=18)
        plt.title('Depth-wise Apparent Slope Analysis', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.ylim(0, r_true + 1)
        plt.legend(fontsize=16)
        plt.grid(True, ls='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_detection_rate(self, analysis_df, output_prefix):
        plt.figure(figsize=(10, 8))
        plt.plot(analysis_df['depth_range'], analysis_df['detection_rate'], marker='o', linestyle='-', color='k', label='Detection Rate')
        plt.axhline(y=1.0, color='k', linestyle='--', label='100% Detection Rate')
        
        plt.xlabel('Depth Range [m]', fontsize=18)
        plt.ylabel('Detection Rate', fontsize=18)
        plt.title('Detection Rate vs Depth Range', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        
        plt.ylim(0, 1.05) 
        plt.legend(fontsize=16)
        plt.grid(True, ls='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_rock_density(self, analysis_df, output_prefix, true_rock_density):
        plt.figure(figsize=(10, 8))
        plt.plot(analysis_df['depth_range'], analysis_df['rock_density'], marker='s', linestyle='-', color='r', label='Detected Rock Density')
        plt.axhline(y=true_rock_density, color='k', linestyle='--', label=f'True Rock Density: {true_rock_density:.2f}')
        
        plt.xlabel('Depth Range [m]', fontsize=18)
        plt.ylabel('Detected Rock Density [1/m²]', fontsize=18)
        plt.title('Detected Rock Density vs Depth Range', fontsize=18)
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
        
        if y_true_all:
            y_true_all = np.array(y_true_all)
            y_true_mean = np.mean(y_true_all, axis=0)
            y_true_std = np.std(y_true_all, axis=0)
            mean_slope_true = np.mean(csfd_stats_df['slope_true'])
            
            plt.plot(x_common, 10**y_true_mean, 'k--', linewidth=2.0, 
                     label=f'Mean True Fit (r = {-mean_slope_true:.2f})')
            # 視認性を上げるため真の岩石のエラーバンドは省略するか薄く描画
            plt.fill_between(x_common, 10**(y_true_mean - y_true_std), 10**(y_true_mean + y_true_std), color='gray', alpha=0.1)

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
            
            plt.plot(x_common, 10**y_det_mean, 'r-', linewidth=2.5, 
                     label=f'Mean Apparent Fit (r = {-mean_slope_det:.2f})')
            plt.fill_between(x_common, 10**(y_det_mean - y_det_std), 10**(y_det_mean + y_det_std), color='red', alpha=0.2, label='Apparent Fit ±1 Std Dev')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Diameter [m]', fontsize=18)
        plt.ylabel('Cumulative Number N(>D)', fontsize=18)
        plt.title(f'Statistical True vs Apparent RSFD\n(Input True r = {r_true}, {len(csfd_stats_df)} Iterations)', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_depth_analysis_stats(self, stats_df, output_prefix, r_true):
        plt.figure(figsize=(10, 8))
        
        x = stats_df['depth_range']
        y_mean = stats_df['r_apparent_mean']
        y_std = stats_df['r_apparent_std']

        plt.plot(x, y_mean, marker='o', linestyle='-', color='blue', label='Mean Apparent r')
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='blue', alpha=0.2, label='±1 Std Dev')
        plt.axhline(y=r_true, color='k', linestyle='--', label='True r')
        
        plt.xlabel('Depth Range [m]', fontsize=18)
        plt.ylabel('Apparent Slope r', fontsize=18)
        plt.title('Statistical Depth-wise Apparent Slope Analysis', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.ylim(0, r_true + 1.5)
        plt.legend(fontsize=16)
        plt.grid(True, ls='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_detection_rate_stats(self, stats_df, output_prefix):
        plt.figure(figsize=(10, 8))
        
        x = stats_df['depth_range']
        y_mean = stats_df['detection_rate_mean']
        y_std = stats_df['detection_rate_std']

        plt.plot(x, y_mean, marker='o', linestyle='-', color='k', label='Mean Detection Rate')
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='k', alpha=0.2, label='±1 Std Dev')
        plt.axhline(y=1.0, color='k', linestyle='--', label='100% Detection Rate')
        
        plt.xlabel('Depth Range [m]', fontsize=18)
        plt.ylabel('Detection Rate', fontsize=18)
        plt.title('Statistical Detection Rate vs Depth Range', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        
        plt.ylim(0, 1.05) 
        plt.legend(fontsize=16)
        plt.grid(True, ls='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.png")
        plt.savefig(f"{output_prefix}.pdf")
        plt.close()

    def plot_rock_density_stats(self, stats_df, output_prefix, true_rock_density):
        plt.figure(figsize=(10, 8))

        x = stats_df['depth_range']
        y_mean = stats_df['rock_density_mean']
        y_std = stats_df['rock_density_std']

        plt.plot(x, y_mean, marker='s', linestyle='-', color='r', label='Mean Detected Density')
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='r', alpha=0.2, label='±1 Std Dev')
        plt.axhline(y=true_rock_density, color='k', linestyle='--', label=f'True Rock Density: {true_rock_density:.2f}')
        
        plt.xlabel('Depth Range [m]', fontsize=18)
        plt.ylabel('Detected Rock Density [1/m²]', fontsize=18)
        plt.title('Statistical Detected Rock Density vs Depth Range', fontsize=18)
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

        true_rock_density = config.TOTAL_ROCKS / (config.MAX_DEPTH * 1500.0)

        # 全体パラメータファイルの出力
        with open(f"{output_dir}/parameters.txt", "w") as f:
            f.write(f"True Slope (r): {r_true}\n")
            f.write(f"Total Rocks: {config.TOTAL_ROCKS}\n")
            f.write(f"True Rock Density [1/m2]: {true_rock_density:.6f}\n")
            f.write(f"Total Iterations: {NUM_ITERATIONS}\n")

        all_analysis_results = []
        overall_csfd_stats = [] # RSFD統計出力用のリスト

        # --- 反復計算の実行 ---
        for i in range(1, NUM_ITERATIONS + 1):
            if i % 5 == 0 or i == 1:
                print(f"  -> イテレーション {i}/{NUM_ITERATIONS} 実行中...")
            
            # 各イテレーション用のサブディレクトリ作成
            iter_dir = f"{output_dir}/iter_{i:02d}"
            os.makedirs(iter_dir, exist_ok=True)

            # 1. 岩石生成
            all_rocks_df = model.generate_rocks(r_true, config, quiet=True)
            all_rocks_df.to_csv(f"{iter_dir}/truth_rocks.csv", index=False)

            # 2. レーダー検出
            detected_df = model.apply_radar_equation(all_rocks_df, config, quiet=True)
            detected_df.to_csv(f"{iter_dir}/simulated_detection.csv", index=False)
            
            # RSFD統計プロット用に傾きと切片を計算・保存
            slope_true, intercept_true, _, _ = analyzer.calculate_slope(all_rocks_df['diameter'].values)
            diameters_det = detected_df[detected_df['is_detected'] == True]['diameter'].values
            slope_det, intercept_det, _, _ = analyzer.calculate_slope(diameters_det)
            
            overall_csfd_stats.append({
                'iteration': i,
                'slope_true': slope_true,
                'intercept_true': intercept_true,
                'slope_det': slope_det,
                'intercept_det': intercept_det
            })

            # 3. 個別プロット
            analyzer.plot_csfd(detected_df, all_rocks_df, f"{iter_dir}/csfd_comparison", r_true)

            # 4. 深さ解析
            analysis_results = analyzer.run_depth_analysis(detected_df, config.MAX_DEPTH, step=1.0, quiet=True)
            analysis_results['iteration'] = i  # イテレーション番号を付与
            analysis_results.to_csv(f"{iter_dir}/depth_analysis_results.csv", index=False)
            
            # 5. 個別深さ解析のプロット
            analyzer.plot_depth_analysis(analysis_results, f"{iter_dir}/depth_analysis", r_true)
            analyzer.plot_detection_rate(analysis_results, f"{iter_dir}/detection_rate")
            analyzer.plot_rock_density(analysis_results, f"{iter_dir}/rock_density", true_rock_density)

            # 結果をリストに保存
            all_analysis_results.append(analysis_results)

        # --- 統計処理とプロットの実行 ---
        print("  -> 統計データの集計とプロットを生成中...")
        
        # --- 全体RSFDの統計プロット ---
        csfd_stats_df = pd.DataFrame(overall_csfd_stats)
        csfd_stats_df.to_csv(f"{output_dir}/csfd_fits_stats.csv", index=False)
        analyzer.plot_csfd_stats(csfd_stats_df, f"{output_dir}/csfd_comparison_stats", r_true, config)

        # --- 深さ解析の統計プロット ---
        # 全イテレーションのデータを結合
        combined_df = pd.concat(all_analysis_results)
        
        # 深さ範囲(depth_range)ごとに平均と標準偏差を計算
        stats_df = combined_df.groupby('depth_range').agg(
            r_apparent_mean=('r_apparent', 'mean'),
            r_apparent_std=('r_apparent', 'std'),
            detection_rate_mean=('detection_rate', 'mean'),
            detection_rate_std=('detection_rate', 'std'),
            rock_density_mean=('rock_density', 'mean'),
            rock_density_std=('rock_density', 'std')
        ).reset_index()

        # 統計データのCSV保存
        stats_df.to_csv(f"{output_dir}/aggregated_stats.csv", index=False)

        # ファイル名を変更してプロット出力
        analyzer.plot_depth_analysis_stats(stats_df, f"{output_dir}/powerlaw_exp_stats", r_true)
        analyzer.plot_detection_rate_stats(stats_df, f"{output_dir}/detection_rate_stats")
        analyzer.plot_rock_density_stats(stats_df, f"{output_dir}/rock_density_stats", true_rock_density)

        print(f"=== 岩石数: {total_rocks} の処理完了 ===\n")

    print("全ての反復処理と統計出力が完了しました。")

if __name__ == "__main__":
    main()