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
        self.FREQ = 500e6  # 周波数 [Hz] (500 MHz)
        self.C_0 = 3e8     # 光速 [m/s]
        
        # システム定数・電力設定 (簡易モデル用)
        # 基準: 深さ0mの最小岩石が十分検出でき、かつ深部で減衰するレベルに設定
        # 論文値より: 送信電圧~400V, インピーダンス100Ω -> Pt ~ 1600W (62 dBm)
        # しかしここでは相対的なS/Nが重要なので、システム利得等を調整した「実効的な定数」として扱う
        self.TX_POWER_DBM = 62.0  # 送信電力 [dBm]
        self.ANTENNA_GAIN_DBI = -7.5 # アンテナ利得 [dBi] (送受合計で2倍する)
        self.SYSTEM_LOSS_DB = 10.0   # システム損失 (ケーブルロス等) [dB]
        
        # ノイズフロア (検出限界)
        # 論文のダイナミックレンジ 96.7dB と変動(sigma=0.6)を考慮
        # 受信最大電力が 0dBm 付近とした場合、ノイズは -90dBm 程度と仮定
        self.NOISE_FLOOR_DBM = -90.0
        
        # --- B. 環境・媒質パラメータ ---
        self.EPSILON_R_REG = 3.1     # レゴリスの比誘電率 (実部)
        self.ATTENUATION_DB_M = 0.5  # 減衰定数 [dB/m] (往復ではなく片道の減衰)
        # 注: 月のレゴリス減衰は 0.1~2.0 dB/m 程度。ここでは標準的な値を採用
        
        self.EPSILON_R_ROCK = 7.0    # 岩石の比誘電率
        
        # --- C. シミュレーション空間設定 ---
        self.MAX_DEPTH = 20.0        # 最大深度 [m]
        self.AREA_SIZE_M2 = 10000.0  # シミュレーション領域面積 [m^2]
        self.ROCK_SIZE_MIN = 0.05    # 最小岩石サイズ [m] (5cm)
        self.ROCK_SIZE_MAX = 5.0     # 最大岩石サイズ [m]
        self.TOTAL_ROCKS = 50000     # 生成する岩石の総数 (統計的有意性のため多めに)

    @property
    def lambda_0(self):
        return self.C_0 / self.FREQ

    @property
    def lambda_g(self):
        # 媒質(レゴリス)中の波長
        return self.lambda_0 / np.sqrt(self.EPSILON_R_REG)

    @property
    def reflection_coeff(self):
        # 岩石表面の反射係数 (簡易的なフレネル反射垂直入射近似)
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
        指定されたべき指数 r_true (累積分布の傾き) に従う岩石を生成する
        N(>D) ∝ D^(-r)  => PDF(D) ∝ D^(-r-1)
        逆関数法を用いて乱数生成する
        """
        print(f"岩石を生成中... (True Slope = -{r_true}, N={config.TOTAL_ROCKS})")
        
        # 一様乱数 [0, 1)
        u = np.random.rand(config.TOTAL_ROCKS)
        
        # べき分布に従う直径 D を生成
        # D = D_min * (1 - u) ^ (-1/r)
        diameters = config.ROCK_SIZE_MIN * (1 - u) ** (-1.0 / r_true)
        
        # 設定した最大サイズを超えるものは除外（あるいはクリップ）
        diameters = diameters[diameters <= config.ROCK_SIZE_MAX]
        
        # 深さをランダムに割り当て (0 ~ MAX_DEPTH)
        depths = np.random.uniform(0, config.MAX_DEPTH, size=len(diameters))
        
        df = pd.DataFrame({
            'diameter': diameters,
            'depth': depths
        })
        return df

    def calculate_rcs_db(self, diameter_array, config):
        """
        岩石のレーダー断面積(RCS)を計算する [dBsm]
        レイリー散乱と光学領域の分岐を含む
        """
        lambda_g = config.lambda_g
        boundary_size = lambda_g / 2.0 # 境界値 (波長の半分)
        
        # 光学領域 (Optical Region) のRCS: 幾何学的断面積 * 反射係数
        # σ = Γ * π * (D/2)^2
        sigma_optical = config.reflection_coeff * np.pi * (diameter_array / 2.0)**2
        
        # レイリー領域 (Rayleigh Region): σ ∝ D^6
        # 境界値で連続になるように係数 k を決める: k * D_bound^6 = sigma_optical_at_bound
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
        
        # 2. 幾何学的減衰 (Spreading Loss): 40 log10(R) (往復なのでR^4比例 -> dBで40log)
        # R = depth. (アンテナ高さ=0と仮定、あるいはdepthに含まれるとする)
        # log(0)回避のため微小値を加える
        spread_loss_db = 40 * np.log10(rocks_df['depth'].values + 0.1)
        
        # 3. 媒質による減衰 (Attenuation): 2 * alpha * depth
        attenuation_loss_db = 2 * config.ATTENUATION_DB_M * rocks_df['depth'].values
        
        # 4. 受信電力計算 [dBm]
        # Pr = Pt + G_tx + G_rx + Sigma - Loss_spread - Loss_atten - Loss_sys
        # (定数項をまとめる)
        const_gain = config.TX_POWER_DBM + 2 * config.ANTENNA_GAIN_DBI - config.SYSTEM_LOSS_DB
        
        # レーダー方程式の定数項 (波長λ^2 / (4π)^3 ) の寄与
        # Pr = (Pt G^2 λ^2 σ) / ((4π)^3 R^4)
        # dB: 10log(λ^2 / (4π)^3) を加算する必要がある
        # ここではアンテナ利得GはdBi、σはdBsm
        wavelength_factor = 10 * np.log10(config.lambda_0**2 / ((4 * np.pi)**3))
        
        received_power = (const_gain + wavelength_factor + rcs_db 
                          - spread_loss_db - attenuation_loss_db)
        
        # 5. ノイズ付加 (オプション: ガウスノイズを加える場合)
        # 今回は閾値判定のみ行うため、信号自体にはノイズを乗せず、閾値を固定とする
        # 判定
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
        # 大きい順にソート
        sorted_d = np.sort(diameters)
        # 累積個数 (N > D): 大きい方から数えた順位
        # 例: [10, 5, 2] -> 10は1個, 5以上は2個, 2以上は3個
        # n = len(d) -> 1 まで
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
        
        # 深さスライス
        depth_ranges = np.arange(1.0, max_depth + 0.1, step)
        
        for d in depth_ranges:
            # 深さd以下の検出岩石を抽出
            subset = detected_df[
                (detected_df['is_detected'] == True) & 
                (detected_df['depth'] <= d)
            ]
            
            count = len(subset)
            if count > 10:
                slope, _, _, _ = self.calculate_slope(subset['diameter'].values)
                # 傾きは負の値なので、絶対値(べき指数)にするなら -slope
                # ここでは slope そのもの（負の値）と、見かけのr（正）を保存
                r_apparent = -slope
            else:
                r_apparent = np.nan
            
            results.append({
                'depth_range': d,
                'r_apparent': r_apparent,
                'count': count
            })
            
        return pd.DataFrame(results)

    def plot_csfd(self, detected_df, output_path, r_true):
        """
        全検出岩石のCSFDプロットを作成 (確認用)
        """
        diameters = detected_df[detected_df['is_detected'] == True]['diameter'].values
        if len(diameters) == 0:
            return

        slope, intercept, x_log, y_log = self.calculate_slope(diameters)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(10**x_log, 10**y_log, s=10, alpha=0.5, label='Detected Rocks')
        
        # フィッティング線
        if not np.isnan(slope):
            x_fit = np.linspace(min(x_log), max(x_log), 100)
            y_fit = slope * x_fit + intercept
            plt.plot(10**x_fit, 10**y_fit, 'r--', label=f'Fit Slope = {slope:.2f}')
            
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Diameter [m]')
        plt.ylabel('Cumulative Number N(>D)')
        plt.title(f'Cumulative Size-Frequency Distribution\n(Input True r = {r_true})')
        plt.legend()
        plt.grid(True, which="both", ls="-")
        plt.savefig(output_path)
        plt.close()

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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_r{r_true}_{timestamp}"
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
        f.write(f"Attenuation: {config.ATTENUATION_DB_M} dB/m\n")
        f.write(f"Noise Floor: {config.NOISE_FLOOR_DBM} dBm\n")

    # 1. 岩石生成 (Ground Truth)
    all_rocks_df = model.generate_rocks(r_true, config)
    all_rocks_df.to_csv(f"{output_dir}/truth_rocks.csv", index=False)

    # 2. レーダーシミュレーション (検出判定)
    detected_df = model.apply_radar_equation(all_rocks_df, config)
    detected_df.to_csv(f"{output_dir}/simulated_detection.csv", index=False)
    
    print(f"検出数: {detected_df['is_detected'].sum()} / {len(detected_df)}")

    # 3. CSFDプロット (全体)
    analyzer.plot_csfd(detected_df, f"{output_dir}/csfd_all_detected.png", r_true)

    # 4. 深さごとの見かけべき指数解析
    analysis_results = analyzer.run_depth_analysis(detected_df, config.MAX_DEPTH, step=1.0)
    analysis_results.to_csv(f"{output_dir}/depth_analysis_results.csv", index=False)

    # 5. 結果プロット (深さ vs べき指数)
    plt.figure(figsize=(10, 6))
    plt.plot(analysis_results['depth_range'], analysis_results['r_apparent'], 'o-', label='Apparent r')
    plt.axhline(y=r_true, color='r', linestyle='--', label=f'True r ({r_true})')
    
    plt.xlabel('Depth Range Used for Analysis [m]')
    plt.ylabel('Apparent Power-Law Index (r)')
    plt.title(f'Change in Apparent Power-Law Index by Depth\n(True r = {r_true})')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, r_true * 1.5) # グラフを見やすく調整
    
    plt.savefig(f"{output_dir}/r_evolution_by_depth.png")
    plt.close()

    print("処理完了。")

if __name__ == "__main__":
    main()