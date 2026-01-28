import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

# --- 1. 設定クラス ---
class RadarConfig:
    """
    シミュレーションの全パラメータを管理するクラス
    """
    def __init__(self):
        # --- A. レーダーシステム (Chang'e-3 LPR Channel 2 参考) ---
        self.FREQ = 500e6  # 周波数 [Hz] (500 MHz), Fang+2014
        self.C_0 = 3e8     # 光速 [m/s]
        
        # システム定数・電力設定
        self.TX_POWER_DBM = 62.0  # 送信電力 [dBm]
        self.ANTENNA_GAIN_DBI = -7.5 # アンテナ利得 [dBi] (送受合計で2倍する), Fang+2014
        self.SYSTEM_LOSS_DB = 0   # システム損失 [dB]
        
        # ノイズフロア (検出限界)
        self.NOISE_FLOOR_DBM = -90.0
        
        # --- B. 環境・媒質パラメータ ---
        self.EPSILON_R_REG = 3.0     # レゴリスの比誘電率
        self.LOSS_TANGENT = 0.004    # ロスタンジェント (tan δ)

        self.EPSILON_R_ROCK = 9.0    # 岩石の比誘電率
        
        # --- C. シミュレーション空間設定 ---
        self.MAX_DEPTH = 12.0        # 最大深度 [m]
        self.AREA_SIZE_M2 = 4500.0   # シミュレーション領域面積 [m^2]
        self.ROCK_SIZE_MIN = 0.01    # 最小岩石サイズ [m]
        self.ROCK_SIZE_MAX = 1.0     # 最大岩石サイズ [m]
        self.TOTAL_ROCKS = 10000     # 生成する岩石の総数

    @property
    def lambda_0(self):
        return self.C_0 / self.FREQ

    @property
    def lambda_g(self):
        # 媒質(レゴリス)中の波長
        return self.lambda_0 / np.sqrt(self.EPSILON_R_REG)

    @property
    def attenuation_db_m(self):
        """
        Loss Tangent から減衰定数 [dB/m] を計算するプロパティ
        """
        omega = 2 * np.pi * self.FREQ
        epsilon_0 = 8.854e-12        # 真空の誘電率
        mu_0 = 4 * np.pi * 1e-7      # 真空の透磁率

        epsilon = self.EPSILON_R_REG * epsilon_0
        mu = mu_0  # 非磁性体と仮定
        
        # 減衰係数 alpha (Neper/m) の計算
        # sqrt(mu * epsilon) = 1/v = sqrt(epsilon_r)/c
        term1 = omega * (np.sqrt(mu * epsilon))
        term2 = np.sqrt(0.5 * (np.sqrt(1 + self.LOSS_TANGENT**2) - 1))
        
        alpha_nepers = term1 * term2
        
        # dB/m に変換
        return alpha_nepers * 8.686 # 8.686は20*log10(e)の近似値

    @property
    def reflection_coeff(self):
        # 岩石表面の反射係数
        n1 = np.sqrt(self.EPSILON_R_REG)
        n2 = np.sqrt(self.EPSILON_R_ROCK)
        return ((n1 - n2) / (n1 + n2))**2

# --- 2. 物理モデルクラス ---
class RockModel:
    """
    岩石の生成とレーダー方程式の適用を行うクラス
    """
    def generate_rocks(self, r_true, config):
        """
        指定されたべき指数 r_true に従う岩石を生成する
        """
        print(f"岩石を生成中... (True Slope = -{r_true}, N={config.TOTAL_ROCKS})")
        
        # 一様乱数 [0, 1)
        u = np.random.rand(config.TOTAL_ROCKS)
        
        # べき分布に従う直径 D を生成
        diameters = config.ROCK_SIZE_MIN * (1 - u) ** (-1.0 / r_true)
        
        # 設定した最大サイズを超えるものは除外
        diameters = diameters[diameters <= config.ROCK_SIZE_MAX]
        
        # 深さをランダムに割り当て
        depths = np.random.uniform(0, config.MAX_DEPTH, size=len(diameters))
        
        df = pd.DataFrame({
            'diameter': diameters,
            'depth': depths
        })
        return df

    def calculate_rcs_db(self, diameter_array, config):
        """
        岩石のレーダー断面積(RCS)を計算する [dBsm]
        """
        lambda_g = config.lambda_g
        boundary_size = lambda_g / 2.0 # 境界値 (波長の半分)
        
        # 光学領域
        sigma_optical = config.reflection_coeff * np.pi * (diameter_array / 2.0)**2
        
        # レイリー領域
        sigma_opt_at_bound = config.reflection_coeff * np.pi * (boundary_size / 2.0)**2
        k_rayleigh = sigma_opt_at_bound / (boundary_size ** 6)
        sigma_rayleigh = k_rayleigh * (diameter_array ** 6)
        
        # サイズに応じて適用
        sigma = np.where(diameter_array >= boundary_size, sigma_optical, sigma_rayleigh)
        
        return 10 * np.log10(sigma)

    def apply_radar_equation(self, rocks_df, config):
        """
        レーダー方程式を適用し、受信電力と検出フラグを計算する
        """
        print("レーダーシミュレーション実行中...")
        
        # 1. RCS計算
        rcs_db = self.calculate_rcs_db(rocks_df['diameter'].values, config)
        
        # 2. 幾何学的減衰
        spread_loss_db = 40 * np.log10(rocks_df['depth'].values + 0.1)
        
        # 3. 媒質による減衰
        attenuation_loss_db = 2 * config.attenuation_db_m * rocks_df['depth'].values
        
        # 4. 受信電力計算 [dBm]
        const_gain = config.TX_POWER_DBM + 2 * config.ANTENNA_GAIN_DBI - config.SYSTEM_LOSS_DB
        wavelength_factor = 10 * np.log10(config.lambda_0**2 / ((4 * np.pi)**3))
        
        received_power = (const_gain + wavelength_factor + rcs_db 
                          - spread_loss_db - attenuation_loss_db)
        
        # 5. 判定
        is_detected = received_power > config.NOISE_FLOOR_DBM
        
        # 結果をDataFrameに追加
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
        """
        累積個数分布の傾きを算出する
        Return: (slope, intercept, x_log, y_log)
        """
        if len(diameters) < 5:
            return np.nan, np.nan, [], []
            
        # 累積分布の作成
        sorted_d = np.sort(diameters)
        # 累積個数 (N > D)
        y_cumulative = np.arange(len(sorted_d), 0, -1)
        
        # 最小二乗法のためにログをとる
        x_log = np.log10(sorted_d)
        y_log = np.log10(y_cumulative)
        
        # フィッティング (1次関数)
        slope, intercept = np.polyfit(x_log, y_log, 1)
        
        return slope, intercept, x_log, y_log

    def run_depth_analysis(self, detected_df, max_depth, step=1.0):
        """
        深さ範囲を少しずつ広げながら、その範囲内の岩石だけで傾きを計算する
        """
        print("深さ方向の感度解析を実行中...")
        results = []
        
        depth_ranges = np.arange(1.0, max_depth + 0.1, step)
        
        for d in depth_ranges:
            subset = detected_df[
                (detected_df['is_detected'] == True) & 
                (detected_df['depth'] <= d)
            ]
            
            count = len(subset)
            if count > 10:
                slope, _, _, _ = self.calculate_slope(subset['diameter'].values)
                r_apparent = -slope # 傾きは負なので、正のべき指数に変換
            else:
                r_apparent = np.nan
            
            results.append({
                'depth_range': d,
                'r_apparent': r_apparent,
                'count': count
            })
            
        return pd.DataFrame(results)

    def plot_csfd(self, detected_df, all_rocks_df, output_path, r_true):
        """
        真のRSFD（生成された全岩石）と見かけのRSFD（検出された岩石）を
        重ねてプロットし、比較できるようにする。
        """
        plt.figure(figsize=(10, 8))

        # --- 1. 真の分布 (Ground Truth) のプロット ---
        # 全岩石データの取得
        diameters_true = all_rocks_df['diameter'].values
        
        if len(diameters_true) > 0:
            slope_true, intercept_true, x_log_true, y_log_true = self.calculate_slope(diameters_true)
            
            # 散布図 (薄いグレーで背景に表示)
            plt.scatter(10**x_log_true, 10**y_log_true, s=20, color='gray', label='Generated Rocks', marker='D')
            
            # 近似直線 (真の分布のフィッティング)
            if not np.isnan(slope_true):
                x_fit_true = np.linspace(min(x_log_true), max(x_log_true), 100)
                y_fit_true = slope_true * x_fit_true + intercept_true
                # r_true (入力値) と 計算された slope_true はほぼ一致するはずだが、確認のため表示
                plt.plot(10**x_fit_true, 10**y_fit_true, 'k--', linewidth=1.5, 
                         label=f'True Fit (r = {-slope_true:.2f})')

        # --- 2. 見かけの分布 (Apparent / Detected) のプロット ---
        # 検出された岩石のみ抽出
        diameters_det = detected_df[detected_df['is_detected'] == True]['diameter'].values
        
        if len(diameters_det) > 0:
            slope_det, intercept_det, x_log_det, y_log_det = self.calculate_slope(diameters_det)
            
            # 散布図 (目立つ色で表示)
            plt.scatter(10**x_log_det, 10**y_log_det, s=20, color='blue', label='Detected Rocks', marker='o')
            
            # 近似直線 (検出データのフィッティング)
            if not np.isnan(slope_det):
                x_fit_det = np.linspace(min(x_log_det), max(x_log_det), 100)
                y_fit_det = slope_det * x_fit_det + intercept_det
                plt.plot(10**x_fit_det, 10**y_fit_det, 'r-', linewidth=2.5, 
                         label=f'Apparent Fit (r = {-slope_det:.2f})')

        # グラフの装飾
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Diameter [m]', fontsize=18)
        plt.ylabel('Cumulative Number N(>D)', fontsize=18)
        plt.title(f'Comparison of True vs Apparent RSFD\n(Input True r = {r_true})', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        
        # 凡例の表示
        plt.legend(fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()

# --- 4. メイン処理 ---
def main():
    # ユーザー入力
    print("--- 岩石見逃しモデル シミュレーション ---")
    try:
        r_input_str = input("真のべき指数(r)を入力してください (例: 2.5): ")
        r_true = float(r_input_str)
    except ValueError:
        print("数値ではない入力です。デフォルト値 2.5 を使用します。")
        r_true = 2.5

    # ディレクトリ作成
    base_dir = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD'
    output_dir = f"{base_dir}/r_{r_true}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"出力ディレクトリ: {output_dir}")

    # インスタンス化
    config = RadarConfig()
    model = RockModel()
    analyzer = Analyzer()

    # パラメータの保存
    with open(f"{output_dir}/parameters.txt", "w") as f:
        f.write(f"True Slope (r): {r_true}\n")
        f.write(f"Frequency: {config.FREQ/1e6} MHz\n")
        f.write(f"Regolith Epsilon: {config.EPSILON_R_REG}\n")
        f.write(f"Loss Tangent: {config.LOSS_TANGENT}\n")
        f.write(f"Noise Floor: {config.NOISE_FLOOR_DBM} dBm\n")

    # 1. 岩石生成 (Ground Truth)
    all_rocks_df = model.generate_rocks(r_true, config)
    all_rocks_df.to_csv(f"{output_dir}/truth_rocks.csv", index=False)

    # 2. レーダーシミュレーション (検出判定)
    detected_df = model.apply_radar_equation(all_rocks_df, config)
    detected_df.to_csv(f"{output_dir}/simulated_detection.csv", index=False)
    
    print(f"検出数: {detected_df['is_detected'].sum()} / {len(detected_df)}")

    # 3. CSFDプロット (全体: 真の値と見かけの値の比較)
    # 【変更点】all_rocks_df を引数に追加
    analyzer.plot_csfd(detected_df, all_rocks_df, f"{output_dir}/csfd_comparison.png", r_true)

    # 4. 深さごとの見かけべき指数解析
    analysis_results = analyzer.run_depth_analysis(detected_df, config.MAX_DEPTH, step=1.0)
    analysis_results.to_csv(f"{output_dir}/depth_analysis_results.csv", index=False)

    # 5. 結果プロット (深さ vs べき指数)
    plt.figure(figsize=(8, 6))
    plt.plot(analysis_results['depth_range'], analysis_results['r_apparent'], 'o-', label='Apparent r')
    plt.axhline(y=r_true, color='r', linestyle='--', label=f'True r ({r_true})')
    
    plt.xlabel('Depth Range Used for Analysis [m]', fontsize=18)
    plt.ylabel('Apparent Power-Law Index (r)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.ylim(0, r_true * 1.5) # グラフを見やすく調整
    
    plt.savefig(f"{output_dir}/r_evolution_by_depth.png")
    plt.close()

    print("処理完了。")

if __name__ == "__main__":
    main()