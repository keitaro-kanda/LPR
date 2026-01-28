import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

# --- 1. 設定クラス (変更なし) ---
class RadarConfig:
    """
    シミュレーションの全パラメータを管理するクラス
    """
    def __init__(self):
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
        self.AREA_SIZE_M2 = 4500.0
        self.ROCK_SIZE_MIN = 0.01
        self.ROCK_SIZE_MAX = 1.0
        self.TOTAL_ROCKS = 10000

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

# --- 2. 物理モデルクラス (生成ロジックを厳密化) ---
class RockModel:
    """
    岩石の生成とレーダー方程式の適用を行うクラス
    """
    def generate_rocks(self, r_true, config):
        """
        【重要】有界パレート分布 (Bounded Pareto Distribution) に従う岩石を生成
        単純な (1-u) 方式ではなく、Min/Max の範囲内で正規化された逆関数法を用いることで、
        データの端（最大サイズ）までフィッティングに含めても傾きがズレないようにする。
        """
        print(f"岩石を生成中... (True Slope = -{r_true}, N={config.TOTAL_ROCKS})")
        
        # 一様乱数 [0, 1)
        u = np.random.rand(config.TOTAL_ROCKS)
        
        D_min = config.ROCK_SIZE_MIN
        D_max = config.ROCK_SIZE_MAX
        
        # --- Bounded Pareto Inverse Transform Sampling ---
        # 公式: D = [ (D_max^-r - D_min^-r) * u + D_min^-r ] ^ (-1/r)
        
        term1 = D_max ** (-r_true)
        term2 = D_min ** (-r_true)
        
        diameters = ( (term1 - term2) * u + term2 ) ** (-1.0 / r_true)
        
        # 深さをランダムに割り当て
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

    def apply_radar_equation(self, rocks_df, config):
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

# --- 3. 解析クラス (全データ使用へ修正) ---
class Analyzer:
    """
    検出された岩石データからRSFDを計算・プロットするクラス
    """
    def calculate_slope(self, diameters):
        """
        累積個数分布の傾きを算出する
        【修正】フィルタリング条件を撤廃し、全てのデータ(N>=1)を使用する
        """
        # データ数が極端に少ない場合のみエラー回避
        if len(diameters) < 2:
            return np.nan, np.nan, [], []
            
        # 累積分布の作成
        sorted_d = np.sort(diameters)
        y_cumulative = np.arange(len(sorted_d), 0, -1)
        
        # 最小二乗法のためにログをとる
        x_log = np.log10(sorted_d)
        y_log = np.log10(y_cumulative)
        
        # --- 全データを用いてフィッティング ---
        # ご要望通り、岩石サイズが大きい（累積数が小さい）データも全て含めます
        slope, intercept = np.polyfit(x_log, y_log, 1)
        
        return slope, intercept, x_log, y_log

    def run_depth_analysis(self, detected_df, max_depth, step=1.0):
        print("深さ方向の感度解析を実行中...")
        results = []
        depth_ranges = np.arange(1.0, max_depth + 0.1, step)
        
        for d in depth_ranges:
            subset = detected_df[
                (detected_df['is_detected'] == True) & 
                (detected_df['depth'] <= d)
            ]
            count = len(subset)
            if count > 5: # 最低限の点数があれば計算
                slope, _, _, _ = self.calculate_slope(subset['diameter'].values)
                r_apparent = -slope
            else:
                r_apparent = np.nan
            
            results.append({
                'depth_range': d,
                'r_apparent': r_apparent,
                'count': count
            })
            
        return pd.DataFrame(results)

    def plot_csfd(self, detected_df, all_rocks_df, output_path, r_true):
        plt.figure(figsize=(10, 8))

        # --- 1. 真の分布 (Ground Truth) ---
        diameters_true = all_rocks_df['diameter'].values
        if len(diameters_true) > 0:
            slope_true, intercept_true, x_log_true, y_log_true = self.calculate_slope(diameters_true)
            
            plt.scatter(10**x_log_true, 10**y_log_true, s=20, color='gray', label='Generated Rocks', marker='D', alpha=0.5)
            
            if not np.isnan(slope_true):
                x_fit_true = np.linspace(min(x_log_true), max(x_log_true), 100)
                y_fit_true = slope_true * x_fit_true + intercept_true
                plt.plot(10**x_fit_true, 10**y_fit_true, 'k--', linewidth=2.0, 
                         label=f'True Fit (r = {-slope_true:.2f})')

        # --- 2. 見かけの分布 (Detected) ---
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
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()

# --- 4. メイン処理 (変更なし) ---
def main():
    print("--- 岩石見逃しモデル シミュレーション ---")
    try:
        r_input_str = input("真のべき指数(r)を入力してください (例: 1.0): ")
        r_true = float(r_input_str)
    except ValueError:
        print("数値ではない入力です。デフォルト値 1.0 を使用します。")
        r_true = 1.0

    # パス設定
    base_dir = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD' 
    output_dir = f"{base_dir}/r_{r_true}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"出力ディレクトリ: {output_dir}")

    config = RadarConfig()
    model = RockModel()
    analyzer = Analyzer()

    with open(f"{output_dir}/parameters.txt", "w") as f:
        f.write(f"True Slope (r): {r_true}\n")
        f.write(f"Total Rocks: {config.TOTAL_ROCKS}\n")

    # 1. 岩石生成
    all_rocks_df = model.generate_rocks(r_true, config)
    all_rocks_df.to_csv(f"{output_dir}/truth_rocks.csv", index=False)

    # 2. レーダー検出
    detected_df = model.apply_radar_equation(all_rocks_df, config)
    detected_df.to_csv(f"{output_dir}/simulated_detection.csv", index=False)
    
    print(f"検出数: {detected_df['is_detected'].sum()} / {len(detected_df)}")

    # 3. プロット (全データ使用)
    analyzer.plot_csfd(detected_df, all_rocks_df, f"{output_dir}/csfd_comparison.png", r_true)

    # 4. 深さ解析 (全データ使用)
    analysis_results = analyzer.run_depth_analysis(detected_df, config.MAX_DEPTH, step=1.0)
    analysis_results.to_csv(f"{output_dir}/depth_analysis_results.csv", index=False)

    print("処理完了。")

if __name__ == "__main__":
    main()