# GPR岩石検出シミュレーター 取扱説明書

## 1. 概要
本プログラムは、月面地中レーダー（GPR: Ground Penetrating Radar）による岩石検出プロセスを数値的に模倣する**統計シミュレーター**です。

真の岩石分布（べき乗則）に対し、レーダー方程式に基づく信号減衰（幾何減衰および媒質減衰）を適用することで、**「深部に行くほど小さな岩石が見えなくなる（検出限界以下の信号となる）」現象**を再現します。また、その結果として観測される「見かけのべき指数（Apparent Slope）」が深度とともにどのように変化するかを解析します。

本プログラムは、複数の岩石数パターン（100, 500, 1000, 10000個）に対して、それぞれ複数回（デフォルト20回）の反復シミュレーションを実行し、統計的に安定した結果（平均±標準偏差）を出力します。

主なモデルベース：Chang'e-3 LPR (Lunar Penetrating Radar) Channel 2 (500 MHz帯)

## 2. 動作環境
以下のPythonライブラリが必要です。

* **numpy**: 数値計算
* **pandas**: データフレーム操作
* **matplotlib**: グラフ描画
* **os**: ディレクトリ操作（標準ライブラリ）

### インストールコマンド例
```bash
pip install numpy pandas matplotlib
```

## 3. 使用方法

### 実行手順
1.  ターミナル（またはコマンドプロンプト）で以下のコマンドを実行します。

```bash
python evaluate_apparent_RSFD.py
```

2.  実行後、プロンプトが表示されますので、シミュレーションしたい**真のべき指数 ($r$)** を入力してください。

```text
--- 岩石見逃しモデル 統計シミュレーション ---
真のべき指数(r)を入力してください (例: 1.0): 2.8
```
* 入力値は、累積個数分布 $N(>D) \propto D^{-r}$ の指数 $r$ です。
* 一般的な月面岩石の $r$ は 2.0 〜 3.5 程度です。
* 数値以外を入力した場合、デフォルト値 ($1.0$) が使用されます。

### 出力
実行が完了すると、以下のパスに結果が保存されます。

```
/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD/r_{入力値}/{岩石数}/
```

岩石数パターンごと（100, 500, 1000, 10000）にサブディレクトリが作成されます。

## 4. 出力ファイルの詳細

### ディレクトリ構成
```
r_{入力値}/
├── 100/                           # 岩石数100のシミュレーション結果
│   ├── parameters.txt             # パラメータ設定ログ
│   ├── csfd_fits_stats.csv        # 全イテレーションのRSFD傾き・切片統計
│   ├── aggregated_stats.csv       # 深さ解析の集計統計データ
│   ├── csfd_comparison_stats.png/pdf   # RSFD統計プロット
│   ├── powerlaw_exp_stats.png/pdf      # べき指数の深さ依存統計プロット
│   ├── detection_rate_stats.png/pdf    # 検出率統計プロット
│   ├── rock_density_stats.png/pdf      # 岩石密度統計プロット
│   ├── iter_01/                   # イテレーション1の個別結果
│   │   ├── truth_rocks.csv
│   │   ├── simulated_detection.csv
│   │   ├── depth_analysis_results.csv
│   │   ├── csfd_comparison.png/pdf
│   │   ├── depth_analysis.png/pdf
│   │   ├── detection_rate.png/pdf
│   │   └── rock_density.png/pdf
│   ├── iter_02/
│   │   └── ...
│   └── iter_20/
│       └── ...
├── 500/
├── 1000/
└── 10000/
```

### 統計出力ファイル（岩石数ごとのルートディレクトリ）

| ファイル名 | 内容 |
| :--- | :--- |
| **parameters.txt** | シミュレーションに使用したパラメータ設定値のログ（真のべき指数、岩石数、真の岩石密度、反復回数）。 |
| **csfd_fits_stats.csv** | 全イテレーションにおけるRSFDの真の傾き・切片と検出された傾き・切片の記録。 |
| **aggregated_stats.csv** | 深さ範囲ごとの見かけの傾き、検出率、岩石密度の平均値と標準偏差。 |

### 統計グラフ画像（岩石数ごとのルートディレクトリ）

| ファイル名 | 内容 |
| :--- | :--- |
| **csfd_comparison_stats.png/pdf** | 全イテレーションのRSFD近似直線を平均化し、±1標準偏差の帯とともに描画した統計プロット。真のフィット（黒破線）と見かけのフィット（赤実線）を比較。 |
| **powerlaw_exp_stats.png/pdf** | **[重要]** 解析深度範囲（横軸）と見かけのべき指数 $r$（縦軸）の関係を平均±1標準偏差で描画。深部を含めるほど $r$ が真の値よりも小さくなる傾向を統計的に確認できる。 |
| **detection_rate_stats.png/pdf** | 深さ範囲に対する検出率の平均±1標準偏差プロット。 |
| **rock_density_stats.png/pdf** | 深さ範囲に対する検出岩石密度の平均±1標準偏差プロット。真の岩石密度（黒破線）との比較。 |

### 個別イテレーション出力ファイル（`iter_XX/` ディレクトリ内）

#### データファイル (CSV)
| ファイル名 | 内容 |
| :--- | :--- |
| **truth_rocks.csv** | 生成された全岩石の真のデータ（直径、深度）。「神の視点」でのデータ。 |
| **simulated_detection.csv** | レーダー方程式適用後のデータ。RCS、受信電力、検出フラグ（`is_detected`）が付与されている。 |
| **depth_analysis_results.csv** | 深度範囲ごとの解析結果。各深度までの岩石を使って計算した見かけの傾き、検出率、岩石密度が記録されている。 |

#### グラフ画像 (PNG/PDF)
| ファイル名 | 内容 |
| :--- | :--- |
| **csfd_comparison.png/pdf** | 真の岩石と検出された岩石の累積サイズ頻度分布（CSFD）の比較プロット。 |
| **depth_analysis.png/pdf** | 解析深度範囲と見かけのべき指数の関係図（個別イテレーション）。 |
| **detection_rate.png/pdf** | 深さ範囲に対する検出率のプロット（個別イテレーション）。 |
| **rock_density.png/pdf** | 深さ範囲に対する検出岩石密度のプロット（個別イテレーション）。 |

## 5. アルゴリズムと物理モデル詳細

本コードでは `RadarConfig`, `RockModel`, `Analyzer` の3つのクラスで処理を行っています。

### A. 岩石生成モデル (`RockModel.generate_rocks`)
* **分布則**: 累積個数 $N(>D) \propto D^{-r}$ に従うよう、逆関数法を用いて乱数を生成します。
* **サイズ範囲**: 直径 $0.01 \sim 1.0\text{m}$
* **空間配置**: 深さ $0 \sim 12\text{m}$ の範囲にランダムに配置します。

### B. レーダー断面積 (RCS) の計算 (`RockModel.calculate_rcs_db`)
岩石のサイズ $D$ と媒質中の波長 $\lambda_g$ の関係により、散乱領域を分けて計算します。

1.  **レイリー領域 ($D < \lambda_g/2$)**:
    * 散乱断面積 $\sigma$ は $D^6$ に比例します（波長に対して物体が小さい場合、反射は急激に弱くなります）。
    * 境界サイズにおける光学領域のRCSとの連続性を保つよう、レイリー係数 $k$ が自動計算されます。
2.  **光学領域 ($D \ge \lambda_g/2$)**:
    * 幾何学的断面積 $\pi (D/2)^2$ に反射係数を乗じた値となります。
    * 反射係数は、レゴリス（$\epsilon_r = 3.0$）と岩石（$\epsilon_r = 9.0$）の誘電率差から計算されます。

### C. レーダー方程式 (`RockModel.apply_radar_equation`)
受信電力 $P_r$ [dBm] は以下の式に基づいて計算されます。

$$
P_r = P_t + G_{sys} + \sigma_{dB} - L_{spread} - L_{atten} + C_{\lambda}
$$

* **幾何減衰 ($L_{spread}$)**: 往復伝搬のため距離の4乗に比例して減衰します（$40 \log_{10} R$）。
* **媒質減衰 ($L_{atten}$)**: レゴリス中を伝搬する際の損失です（$2 \times \alpha \times R$）。
    * 減衰定数 $\alpha$ は loss tangent（$\tan\delta = 0.004$）から以下の式で計算されます:
    $$
    \alpha = \omega \sqrt{\mu \epsilon} \cdot \sqrt{\frac{1}{2}\left(\sqrt{1 + \tan^2\delta} - 1\right)} \times 8.686 \text{ [dB/m]}
    $$
* **検出判定**: $P_r$ がノイズフロア（デフォルト $-90 \text{ dBm}$）を超えている場合のみ「検出（Detected）」と判定されます。

### D. 統計解析 (`Analyzer`)
* **RSFD傾きの計算** (`calculate_slope`): 累積サイズ頻度分布の対数-対数プロットに対して線形フィッティングを行い、傾きと切片を求めます。
* **深さ方向の感度解析** (`run_depth_analysis`): 深さ1mステップで最大深度まで、各深さ範囲における見かけの傾き、検出率、岩石密度を計算します。
* **統計プロット**: 複数イテレーションの結果を集約し、平均値と±1標準偏差の帯を描画します。

## 6. パラメータのカスタマイズ

シミュレーション条件を変更したい場合は、コード冒頭の `RadarConfig` クラス内の変数を直接編集してください。

```python
class RadarConfig:
    def __init__(self, total_rocks=10000):
        # --- A. レーダーシステム ---
        self.FREQ = 500e6          # 周波数 (Hz)
        self.C_0 = 3e8             # 光速 (m/s)

        self.TX_POWER_DBM = 62.0       # 送信電力 (dBm)
        self.ANTENNA_GAIN_DBI = -7.5   # アンテナゲイン (dBi)
        self.SYSTEM_LOSS_DB = 0        # システム損失 (dB)
        self.NOISE_FLOOR_DBM = -90.0   # ノイズフロア (dBm)

        # --- B. 環境・媒質パラメータ ---
        self.EPSILON_R_REG = 3.0       # レゴリスの比誘電率
        self.LOSS_TANGENT = 0.004      # 損失正接 (減衰定数の計算に使用)
        self.EPSILON_R_ROCK = 9.0      # 岩石の比誘電率

        # --- C. シミュレーション空間設定 ---
        self.MAX_DEPTH = 12.0          # 最大深度 (m)
        self.AREA_SIZE_M2 = 4500.0     # シミュレーション面積 (幅3m × 奥行き1500m)
        self.ROCK_SIZE_MIN = 0.01      # 岩石直径の最小値 (m)
        self.ROCK_SIZE_MAX = 1.0       # 岩石直径の最大値 (m)
        self.TOTAL_ROCKS = total_rocks # 生成する岩石の総数
```

### 計算パラメータ（`main()` 関数内）

```python
rock_counts = [100, 500, 1000, 10000]  # シミュレーションする岩石数パターン
NUM_ITERATIONS = 20                     # 各パターンの反復回数
```

### 派生パラメータ（自動計算プロパティ）
| プロパティ | 内容 |
| :--- | :--- |
| `lambda_0` | 自由空間波長: $\lambda_0 = c / f$ |
| `lambda_g` | 媒質中波長: $\lambda_g = \lambda_0 / \sqrt{\epsilon_r}$ |
| `attenuation_db_m` | 媒質減衰定数 [dB/m]: loss tangent から計算 |
| `reflection_coeff` | 反射係数: レゴリスと岩石の誘電率差から計算 |

## 7. トラブルシューティング

* **エラー: `ZeroDivisionError` や `log(0)` 系の警告**
    * 岩石生成数が極端に少ない、または深度0mの岩石が生成された場合に発生することがあります。コード内では `+ 0.1` 等の微小値を加えて回避していますが、岩石総数 `TOTAL_ROCKS` を増やすことで安定します。
* **グラフの傾きがおかしい**
    * 生成された岩石数が少ないと、統計的なばらつきによりフィッティングがうまくいかない場合があります。`rock_counts` に大きな値（$100,000$ 程度）を追加して試してください。
* **出力先ディレクトリのエラー**
    * 出力先は外部SSD (`/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD`) に設定されています。SSDがマウントされていない場合、`main()` 関数内の `base_dir` を変更してください。
