# Statistical and Mathematical Analysis Tools

LPRデータの統計解析・数理解析ツール群です。

## 主要ツール

### 統計解析
- **`analyze_rock_label_statistics.py`**: 岩石ラベルの統計的性質をヒストグラム表示
  - 深さ・水平位置別の岩石分布ヒストグラム
  - 深さ計測データによる規格化ヒストグラム（1mあたりの平均岩石個数）
  - **薄い層規格化ヒストグラム**: 最も薄い層の厚みまでのデータで規格化
  - ラベル別フィルタリング機能（岩石のみ、全ラベル）
  - 出力ディレクトリ分離（basic/、normalized/）
  - JSON形式の詳細統計データ出力
- **`analyze_rock_label_grid.py`**: 2次元グリッドベース岩石ラベル解析
  - 指定ウィンドウサイズでの2次元グリッド分割
  - 各グリッドセル内での円グラフ表示（ラベル数を直接表示）
  - 3パターンの集計（岩石のみ1-3、非岩石のみ4-6、全ラベル1-6）
  - グリッド統計データのテキスト出力
- **`make_RSFD_plot.py`**: 岩石サイズ頻度分布（RSFD）の作成・プロット
- **`make_RSFD_from_csv.py`**: CSVからRSFDプロットを生成


### その他の解析
- **`calc_RCS.py`**: レーダー散乱断面積（RCS）計算

## 使用例

```bash
# 岩石統計解析（基本機能）
python analysis/analyze_rock_label_statistics.py
# 実行時に岩石ラベルJSONファイルを指定
# 深さ計測JSONファイル（オプション）で規格化ヒストグラムも生成

# 2次元グリッドベース解析
python analysis/analyze_rock_label_grid.py
# 実行時にJSONファイルとウィンドウサイズ（水平・深さ）を指定

# RSFD解析
python analysis/make_RSFD_plot.py

# RCS計算
python analysis/calc_RCS.py
```

## 解析対象

- **岩石分布**: サイズ・深度・水平分布の統計
- **深さ規格化分析**: 深さデータを用いた岩石密度の定量化
- **散乱特性**: 電磁波散乱の定量評価
- **サイズ頻度分布**: 岩石サイズの統計的性質

## 統計手法

- **ヒストグラム解析**: 分布の可視化
- **深さ規格化**: 1mあたりの平均岩石個数算出
- **x座標マッチング**: 許容誤差付き座標対応機能
- **べき則フィッティング**: 累積サイズ分布のモデリング
- **散乱断面積計算**: レーダー反射強度の定量化

## analyze_rock_label_statistics.py 詳細機能

### 入力データ
- **岩石ラベルJSON**: `plot_viewer_add_label.py`で作成されたラベルデータ
- **深さ計測JSON**: `plot_viewer_depth_measurement.py`で作成された深さデータ（オプション）

### 出力ディレクトリ構造
```
label_statics/{json_filename}/
├── basic/                    # 規格化なしのヒストグラム
│   ├── depth_histogram_*.png/pdf
│   ├── horizontal_histogram_*.png/pdf
│   └── *_statistics.txt
├── normalized/              # 規格化ありのヒストグラム
│   ├── normalized_horizontal_histogram_*.png/pdf
│   ├── normalized_horizontal_histogram_thin_*.png/pdf
│   └── *_statistics.txt
└── summary_statistics.txt   # 全体の概要統計
```

### 出力ファイル詳細
#### 基本ヒストグラム（`basic/`ディレクトリ）
- `depth_histogram_*.png/pdf`: 深さ別岩石分布
- `horizontal_histogram_*.png/pdf`: 水平位置別岩石分布

#### 深さ規格化ヒストグラム（`normalized/`ディレクトリ）
- `normalized_horizontal_histogram_*.png/pdf`: 深さで規格化された水平分布
- `normalized_horizontal_histogram_thin_*.png/pdf`: 薄い層規格化水平分布
- Y軸: 岩石密度 [count/m depth] - 1mあたりの平均岩石個数

#### 統計データ
- `*_statistics.txt`: 詳細な統計データ（カウント・密度・ビン情報）
- `summary_statistics.txt`: 全体概要統計

### 深さ規格化の仕組み
#### 通常の規格化
1. **x座標マッチング**: 各ビンの下限座標に最も近い深さ計測点を検索
2. **許容誤差**: ビンサイズの半分（デフォルト25m）以内の計測点を有効とする
3. **密度計算**: `岩石個数 / 深さ[m] = 密度[個/m]`
4. **欠損処理**: 深さデータがないビンは密度0として処理

#### 薄い層規格化（新機能）
1. **最薄層特定**: 深さ計測データから最小深さを特定
2. **データフィルタリング**: 最薄層の厚みまでのy座標（時間）でデータをフィルタリング
3. **有効深さ計算**: 各ビンで実際の深さと最薄層厚みの最小値を使用
4. **密度計算**: `岩石個数 / 有効深さ[m] = 密度[個/m]`

### ビン設定
#### 深さビン（0.5の倍数で区切り）
- ビンサイズ: 0.5 m
- 境界設定: -1.0, -0.5, 0, 0.5, 1.0... のように0.5の倍数で設定

#### 水平位置ビン（0スタート）
- ビンサイズ: 50.0 m
- 境界設定: 0, 50, 100, 150... のように常に0からスタート

### 設定パラメータ
- 深さビンサイズ: 0.5 m
- 水平位置ビンサイズ: 50.0 m  
- 相対誘電率: 4.5
- 座標マッチング許容誤差: 25.0 m（ビンサイズの半分）
- 深さ情報取得: ビン下限座標を基準