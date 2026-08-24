# 見かけRSFD評価モデル (`evaluate_apparent_RSFD_relative.py`)

GPR（月レーダー）で地下岩石を観測する際、信号減衰によって小さい岩石・深い岩石が見逃されることで、計測される岩石サイズ頻度分布（RSFD: Rock Size-Frequency Distribution）が真の分布からどれだけ歪むかを評価するモンテカルロシミュレーションです。

Chang'E-4 搭載 Lunar Penetrating Radar (LPR) CH2 を対象としています。

真のRSFDパラメータ $(r_{true}, k_{true})$ を入力として岩石集団を生成し、レーダー方程式で各岩石の反射強度を計算、ノイズレベルを超えたものだけを「検出」として見かけのRSFDパラメータ $(r_{app}, k_{app})$ を出力します。

---

## 目次

- [物理モデル](#物理モデル)
- [必要環境](#必要環境)
- [使い方](#使い方)
- [設定パラメータ](#設定パラメータ)
- [出力ファイル](#出力ファイル)
- [コード構成](#コード構成)
- [既知の制約と注意点](#既知の制約と注意点)
- [関連スクリプト](#関連スクリプト)

---

## 物理モデル

計算はすべて**非dB形式（線形パワー）**で行っています。dB は表示・プロット用にのみ使用します。

### 規格化受信強度

LPR のシステム定数 $C_{sys}$ を厳密に決定できないため、地表面反射強度で規格化して $C_{sys}$ を打ち消します。

$$P_{norm}=\frac{P_{rock}}{P_{surf}}=\frac{\sigma_{rock}(D)}{\sigma_{surf}}\left(\frac{h_{ante}}{R_{rock}}\right)^{4}e^{-4\alpha_{reg}d_{rock}}\cdot 10^{X/10}$$

| 記号 | 意味 |
|---|---|
| $d_{rock}$ | 岩石の埋没深さ [m]（吸収損失を受ける片道経路長） |
| $R_{rock}=h_{ante}+d_{rock}$ | GPR–岩石間距離 [m] |
| $h_{ante}$ | アンテナ地上高 [m]（$=R_{surf}$） |
| $X \sim N(0,\sigma_{\mathrm{dB}}^2)$ | 反射強度のランダムなばらつき [dB] |

指数部の係数について：$\alpha$ は**電界**の減衰定数 [Np/m] なので、パワーの片道減衰が $e^{-2\alpha d}$、レーダーの往復（経路長 $2d$）で $e^{-4\alpha d}$ になります。吸収はレゴリス中でのみ起こるため、経路長には $R_{rock}$ ではなく $d_{rock}$ を使います。

### 減衰定数

Jol (2009) Eq. 2.14：

$$\alpha_{reg} = 2\pi f\sqrt{\frac{\mu_0\mu_r\varepsilon_0\varepsilon_{reg}}{2}\left(\sqrt{1+\tan^2\delta}-1\right)}\quad[\mathrm{Np/m}]$$

平方根は $(\mu\varepsilon/2)$ と $(\sqrt{1+\tan^2\delta}-1)$ の**積の全体**にかかります。デフォルト設定では $\alpha_{reg}=0.0363$ Np/m（片道 0.315 dB/m）、深さ12 m の往復損失は 7.57 dB です。

### レーダー断面積 (RCS)

**地表面**（滑らかな平板と仮定, Daniels 2004 Table 2.2）：

$$\sigma_{surf}=\Gamma_{surf}^2\frac{4\pi A^2}{\lambda_0^2},\quad A=\pi r_{fres}^2,\quad r_{fres}=\sqrt{\frac{\lambda_0 h_{ante}}{2}}\;\;\Longrightarrow\;\;\sigma_{surf}=\Gamma_{surf}^2\pi^3h_{ante}^2$$

アンテナ–地表面間の伝搬は真空（空気）中で起こり、入射波・散乱波ともに空気中の場なので $\lambda_0=c/f$ を使います。代入すると $\lambda_0$ は約分されて消えます。コードでは意図を明示するため約分前の導出形で実装しています。

**岩石**：散乱領域の境界は、散乱体を取り囲む媒質（レゴリス）中の波長 $\lambda_{reg}=c/(f\sqrt{\varepsilon_{reg}})$ の 1/2 です（デフォルトで 0.173 m）。

$$
\sigma_{rock}=
\begin{cases}
\Gamma_{rock}^2\,\pi\left(\dfrac{D}{2}\right)^2 & (D\geq\lambda_{reg}/2)\quad\text{幾何光学領域}\\[2ex]
\left.\sigma_{rock}\right|_{D=\lambda_{reg}/2}\times\left(\dfrac{D}{\lambda_{reg}/2}\right)^6 & (D<\lambda_{reg}/2)\quad\text{レイリー領域}
\end{cases}
$$

### 反射係数

$$\Gamma = \frac{\sqrt{\varepsilon_1}-\sqrt{\varepsilon_2}}{\sqrt{\varepsilon_1}+\sqrt{\varepsilon_2}}$$

$\Gamma_{surf}$ は $\varepsilon_1=\varepsilon_{air}$, $\varepsilon_2=\varepsilon_{reg}$、$\Gamma_{rock}$ は $\varepsilon_1=\varepsilon_{reg}$, $\varepsilon_2=\varepsilon_{rock}$。RCS では二乗して使うため符号は結果に影響しません。

### 検出判定

$$P_{norm} > 10^{\mathrm{NOISE\_FLOOR\_DB}/10} = 10^{-9}\quad(-90\ \mathrm{dB})$$

ノイズレベルは、CE-4/LPR の全A-scan平均プロファイルにおいて、地表面反射パワーに対して検出可能な最小パワーが $-90$ dB であったことから決定しています。

### 岩石集団の生成

累積個数分布 $N(>D)\propto D^{-r}$ を仮定し、逆関数法（Inverse Transform Sampling）で $[D_{min}, D_{max}]$ の切断べき乗分布からサンプリングします。

$$D=\left[u\left(D_{max}^{-r}-D_{min}^{-r}\right)+D_{min}^{-r}\right]^{-1/r},\qquad u\sim U[0,1)$$

$D_{max}$ は $N(>D_{max})=1$ となるサイズとして自動決定されます：

$$D_{max}=D_{min}\,N_{true}^{1/r}$$

深さは $0$–`MAX_DEPTH` の一様分布で与えます。

---

## 必要環境

- Python 3.8+
- `numpy`, `pandas`, `matplotlib`, `tqdm`

```bash
pip install numpy pandas matplotlib tqdm
```

`matplotlib` は並列処理時のクラッシュ防止のため `Agg` バックエンドを強制しています。

---

## 使い方

### 1. 入力モードの選択

ファイル冒頭のフラグを編集します（頻繁に切り替えない想定のためハードコーディング）。

```python
INPUT_MODE = 'N'   # 'N': 岩石数 N_true 入力系 / 'K': 累積岩石密度 k_true 入力系
```

| モード | 入力 | 出力先 |
|---|---|---|
| `'N'` | $r_{true}$（対話入力）＋ $N_{true}$ のリスト（対話入力） | `BASE_DIR_N` |
| `'K'` | $r_{true}$（対話入力）＋ `K_TRUE_LIST`（ハードコーディング） | `BASE_DIR_K` |

$k_{true}$ と $N_{true}$ は全岩石が $D\geq D_{min}$ であることから

$$k_{true}=\frac{N_{true}}{A_{obs}}$$

で厳密に対応します。K系では `N_true = round(k_true × AREA_SIZE_M2)` として $N$ を復元するため、**両モードで出力項目は完全に同一**です。

### 2. 出力先の設定

```python
BASE_DIR_N = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD_relative'
BASE_DIR_K = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/evaluate_apparent_RSFD_relative_kinput'
```

### 3. 実行

```bash
python evaluate_apparent_RSFD_relative.py
```

**N系の実行例**

```
真のべき指数(r)を入力してください (例: 1.0): 2.0
計算する岩石数をカンマ区切りで入力してください (例: 100,500,1000,5000) を入力: 100,500,1000,5000
```

**K系の実行例**（岩石数の入力は不要）

```
真のべき指数(r)を入力してください (例: 1.0): 2.0
```

各条件について `NUM_ITERATIONS`（デフォルト100）回のイテレーションを `ProcessPoolExecutor` で並列実行します。$r$ を変えて繰り返し実行すると、同じ `rsfd_summary.json` に追記されていきます。

---

## 設定パラメータ

### モジュールレベル定数

| 定数 | デフォルト | 説明 |
|---|---|---|
| `INPUT_MODE` | `'N'` | 入力モード切り替え |
| `BASE_DIR_N` / `BASE_DIR_K` | — | 出力の親ディレクトリ |
| `K_TRUE_LIST` | `[1e-3, 5e-3, 10e-3, 50e-3]` | K系で計算する $k_{true}$ [1/m²] |
| `MAX_ROCK_SIZE_LIMIT` | `10.0` | $D_{max}$ の上限 [m]。超える条件は計算せず JSON に `"Calculation stop"` を記録 |
| `NUM_ITERATIONS` | `100` | `main()` 内。イテレーション回数 |

### `RadarConfig`

| パラメータ | デフォルト | 説明 | 出典 |
|---|---|---|---|
| `FREQ` | `500e6` | 中心周波数 [Hz] | LPR CH2 |
| `H_ANTENNA` | `0.3` | アンテナ地上高 [m] | Fang et al. (2014) |
| `NOISE_FLOOR_DB` | `-90.0` | 検出しきい値（表面反射規格化）[dB] | LPR 観測データ |
| `EPSILON_R_AIR` | `1.0` | 空気の比誘電率 | |
| `EPSILON_R_REG` | `3.0` | レゴリスの比誘電率 | Dong+2020, Chen+2022, Feng+2022 |
| `EPSILON_R_ROCK` | `9.0` | 岩石の比誘電率 | Chung+1970 |
| `LOSS_TANGENT` | `0.004` | レゴリスのロスタンジェント | Feng+2022 |
| `SIGMA_RANDOM_DB` | `3.0` | 反射強度ばらつきの標準偏差 [dB] | |
| `MAX_DEPTH` | `12.0` | 岩石を配置する最大深さ [m] | 最上部レゴリス層の厚さ |
| `AREA_SIZE_M2` | `18300.0` | 観測領域の断面積 [m²] | 12 m × 1525 m |
| `ROCK_SIZE_MIN` | `0.06` | 最小岩石サイズ $D_{min}$ [m] | LPR CH2 のサイズ推定下限 |

主な派生プロパティ：`lambda_0`, `lambda_reg`, `fresnel_radius`, `attenuation_np_m` [Np/m], `attenuation_db_m`（表示専用）, `noise_floor_linear`, `reflection_coeff`, `surface_reflection_coeff`, `surface_rcs`, `true_rock_density`

---

## 出力ファイル

```
<BASE_DIR>/
├── rsfd_summary.json                    # 全条件の (r,k) 統計サマリ（追記式）
└── r_<r_true>/
    └── <100 または k_0.005>/            # N系はN、K系は k_<値>
        ├── parameters.txt               # 実行条件の記録
        ├── overall_rsfd_stats.txt       # 全深さでの r, k の平均±標準偏差
        ├── csfd_fits_stats.csv          # イテレーションごとのフィット結果
        ├── aggregated_stats_range.csv   # 累積深さ範囲解析の集計
        ├── aggregated_stats_moving.csv  # 移動窓解析の集計
        ├── RSFD_comparison_stats.png/pdf    # true vs apparent RSFD（密度）
        ├── RSFD_count_stats.png/pdf         # true vs apparent RSFD（個数）
        ├── powerlaw_exp_range_stats.png/pdf # r の深さ依存（累積範囲）
        ├── powerlaw_k_range_stats.png/pdf   # k の深さ依存（累積範囲）
        ├── detection_rate_range_stats.png/pdf
        ├── powerlaw_exp_moving_stats.png/pdf # r の深さ依存（移動窓）
        ├── powerlaw_k_moving_stats.png/pdf   # k の深さ依存（移動窓）
        └── each_iteration/
            └── iter_01/ ... iter_NN/
                ├── truth_rocks.csv          # 生成された岩石（直径・深さ）
                ├── simulated_detection.csv  # 強度計算・検出判定の結果
                ├── depth_analysis_range.csv
                ├── depth_analysis_moving.csv
                ├── power_scatter.png        # 深さ×サイズ空間の強度マップ
                ├── csfd_comparison.png
                ├── powerlaw_exp_range.png / powerlaw_k_range.png
                ├── detection_rate_range.png
                └── powerlaw_exp_moving.png / powerlaw_k_moving.png
```

### `simulated_detection.csv` の列

| 列 | 単位 | 説明 |
|---|---|---|
| `diameter` | m | 岩石直径 $D$ |
| `depth` | m | 埋没深さ $d_{rock}$ |
| `rcs` | m² | 岩石のレーダー断面積 $\sigma_{rock}$（線形） |
| `p_norm` | — | 規格化受信強度 $P_{norm}$（**線形比**） |
| `p_norm_db` | dB | $10\log_{10}P_{norm}$（表示・プロット用） |
| `is_detected` | bool | `p_norm > noise_floor_linear` |

### `rsfd_summary.json` の構造

```json
{
  "2.0": {
    "0.005": {
      "input_mode": "K",
      "N_true": 92,
      "k_true_input": 0.005,
      "k_true_realized": 0.005027322404371585,
      "r_true_mean": 1.9719, "r_true_std": 0.0690,
      "k_true_mean": 0.005058, "k_true_std": 0.000404,
      "r_apparent_mean": 1.5958, "r_apparent_std": 0.0829,
      "k_apparent_mean": 0.003077, "k_apparent_std": 0.000753
    }
  }
}
```

第1階層のキーが $r_{true}$、第2階層が $N_{true}$（N系）または $k_{true}$（K系）です。$D_{max}$ が上限を超えた条件は文字列 `"Calculation stop"` が入ります。

### 解析の種類

- **累積深さ範囲解析** (`run_depth_analysis`)：深さ $0$–$d$ の岩石をまとめてフィット。$d$ を 1 m 刻みで変化。
- **移動窓解析** (`run_moving_window_analysis`)：幅 2 m の窓を 20% ずつずらしながらフィット。
- どちらも窓内の検出岩石が **6個以上**ないとフィットせず `NaN` を返します。

---

## コード構成

| クラス / 関数 | 役割 |
|---|---|
| `RadarConfig` | 全パラメータと派生量（波長・減衰定数・RCS等）を保持 |
| `RockModel.generate_rocks` | 切断べき乗分布から岩石集団を生成 |
| `RockModel.calculate_rcs` | 岩石RCS [m²]（幾何光学／レイリー領域の分岐） |
| `RockModel.apply_radar_equation` | $P_{norm}$ の計算と検出判定 |
| `Analyzer.calculate_slope` | 累積分布の log-log 最小二乗フィット |
| `Analyzer.run_depth_analysis` | 累積深さ範囲ごとの $r,k$ 算出 |
| `Analyzer.run_moving_window_analysis` | 移動窓ごとの $r,k$ 算出 |
| `Analyzer.plot_*` | 個別イテレーション用プロット |
| `Analyzer.plot_*_stats` | 複数イテレーション統計（±1σ帯付き）プロット |
| `process_iteration` | 並列実行される1イテレーション分のワーカー |
| `build_cases` | `INPUT_MODE` に応じた計算ケースリストの構築 |
| `main` | 対話入力・ループ・集計・保存 |

`r` と `k` は必ず $D_{min}$（デフォルト 6 cm）を基準に定義されます。

- $r$：累積分布 $N(>D)\propto D^{-r}$ のべき指数（フィット傾きの符号反転）
- $k$：$D\geq D_{min}$ の累積岩石密度 [1/m²]（`10**intercept * d_min_cm**slope`）

$k$ を「べき乗則の係数」ではなく「$D_{min}$ における累積密度」と定義しているため、$D$ の単位（cm/m）に依存しない物理量になっています。

---

## 既知の制約と注意点

### 統計的な制約

- **`k_true = 1e-3` は $N_{true}=18$ 個**です。移動窓（幅2 m）では1窓あたり平均3個となり、6個以上というフィット条件をほぼ常に下回るため、`aggregated_stats_moving.csv` の大半が `NaN` になります。この条件で深さプロファイルが必要な場合は `window_size` を広げてください。
- **$r=1.0$ では $k_{true}\geq 10\times10^{-3}$ が計算されません**（$D_{max}$ が 10.98 m, 54.90 m となり `MAX_ROCK_SIZE_LIMIT` を超過）。

### 入力パラメータと `r_true` / `k_true` の関係

$D_{max}$ で切断された分布は $D_{max}$ 近傍で累積分布が丸まるため、**フィットで得られる $r_{true}$ は入力した $r$ より系統的に大きく出ます**（$N=100$ で最大 +5.6%、$N=1000$ で +3% 程度）。コードは入力値をそのまま使わず生成された岩石集団を毎回フィットし直しているので、比較には出力された `r_true_mean` を使ってください。

### 乱数シード

`np.random.seed()` を設定していません。macOS（`spawn` 方式）では各プロセスが独立に初期化されるため問題ありませんが、**Linux（`fork` 方式）では全イテレーションが同一結果になります**。Linux で実行する場合は `process_iteration` の冒頭に `np.random.seed(i)` 等を追加してください。

### モデルの近似

- **境界での透過損失を含んでいません。** 地表面を往復する透過損失 $(1-\Gamma_{surf}^2)^2\approx 0.86$（$-0.65$ dB）は $D$ にも $d$ にも依存しない定数なので、RSFD の傾きには影響しません。
- **地表面RCSはフレネルゾーン近似です。** 無限平面の鏡像法から厳密に逆算すると $\sigma_{surf}=\Gamma^2\pi h^2$ となり、本モデルの $\Gamma^2\pi^3h^2$ は $\pi^2$（9.94 dB）大きい値です。$h_{ante}=0.3$ m は $\lambda_0=0.6$ m の半分しかなく遠方界の仮定自体が成立しない領域のため厳密解は存在しませんが、$P_{norm}$ 全体に定数バイアスがかかっている可能性がある点は認識しておいてください。
- **幾何拡散は直線距離 $R_{rock}=h_{ante}+d_{rock}$** で計算しており、地表面での屈折や速度変化を考慮していません。
- なお $\sigma_{surf}\propto h^2$ を $R^{-4}$ の式に入れると実効的に $h^{-2}$ 依存になり、Daniels (2004) Table 2.1 の「planar reflector は (target depth)$^{-2}$」と整合します。

---

## 関連スクリプト

同じ出力ディレクトリを入力として使う後段の解析スクリプトです。どちらも N系 / K系の両方に対応しており、ディレクトリ名・JSONキーの形式から入力モードを自動判定します。

### `analyze_depth_range_RSFD.py`

各イテレーションの `simulated_detection.csv` を読み直し、**任意の深さ範囲に限定した** RSFD を再解析します。評価モデルを再実行する必要はありません。

```bash
python analyze_depth_range_RSFD.py
# 評価モデル出力のベースディレクトリのパス：/path/to/evaluate_apparent_RSFD_relative_kinput
# 解析する深さ範囲の開始値 [m] を入力してください (例: 0.0): 0.0
# 解析する深さ範囲の終了値 [m] を入力してください (例: 3.0): 3.0
```

- 面積は `(d_end - d_start) / MAX_DEPTH × AREA_SIZE_M2` で按分されます。
- 出力は `<BASE_DIR>/analysis_<範囲>/r_<r>/<ケース名>/` と `<BASE_DIR>/rsfd_summary_<範囲>.json`。
- `diameter` / `depth` / `is_detected` 列しか使わないため、非dB化の前後どちらのCSVでも動作します。
- 深さ範囲内の検出岩石が6個未満のイテレーションはフィットせず `NaN` になります。全イテレーションが該当する場合は警告を表示します（$k=10^{-3}$ のような小さい $N$ で起こりやすい）。

### `summarize_apparent_RSFD_params.py`

`rsfd_summary*.json` を読み込み、$r_{true}$ と $r_{app}$、$k_{true}$ と $k_{app}$ の関係、および CE-4/LPR 観測値との比較図を出力します。

```bash
python summarize_apparent_RSFD_params.py
# summary.jsonを含むベースディレクトリのパス：/path/to/evaluate_apparent_RSFD_relative_kinput
# 使用するJSONファイルを選択してください: 0
```

出力先は `Summary_Plots_all_range/`（全範囲）または `Summary_Plots_<範囲>/`（深さ範囲）で、以下の5種類の図を生成します。

| ファイル | 内容 |
|---|---|
| `r_apparent_plot` | Input $r$ vs Apparent $r$ |
| `r_apparent_ratio_plot` | Input $r$ vs $r_{app}/r_{true}$ |
| `k_apparent_ratio_plot` | Input $r$ vs $k_{app}/k_{true}$ |
| `k_apparent_plot` | $k_{true}$ vs $k_{app}$（両対数） |
| `apparent_r_vs_k_<i>` | Apparent $r$–$k$ 平面上での観測値との比較（LPR観測値ごとに1枚） |

凡例のケース表記は入力モードに応じて `N = 1000` / `k = 0.005` と自動で切り替わります。`apparent_r_vs_k_*` の軸範囲は従来の既定値を基準に、データと観測値が収まるよう自動で拡張されます。

比較対象の観測値はスクリプト冒頭の `LPR_obs` にハードコーディングされています。

```python
LPR_obs = [
    {'r': 1.08, 'k': 7.21e-3},  # full range
    {'r': 1.92, 'k': 19.3e-3},  # 0-3 m
    {'r': 1.00, 'k': 5.68e-3}   # 3-12 m
]
```

---

- Skolnik, M. I. (2008), *Radar Handbook*, 3rd ed. — レーダー方程式（Eq. 1.2）、レイリー領域の $D^6$ 依存
- Jol, H. M. (2009), *Ground Penetrating Radar: Theory and Applications* — 減衰定数（Eq. 2.14）、フレネル半径（Eq. 1.31）
- Daniels, D. J. (2004), *Ground Penetrating Radar*, 2nd ed. — 平板のRCS（Table 2.2）、反射係数（Eq. 4.31）、対象形状ごとの距離則（Table 2.1）
- Fang, G. et al. (2014) — LPR の諸元
- Dong, Z. et al. (2020); Chen, R. et al. (2022); Feng, J. et al. (2022) — レゴリスの誘電特性
- Chung, D. H. et al. (1970) — 月岩石の誘電率