# Visualization Tools

LPRデータの可視化・プロットツール群です。

## 主要ツール

### B-scan可視化
- **`plot_Bscan.py`**: 基本的なB-scanプロット
- **`plot_Bscan_detected_rocks.py`**: 岩石検出結果付きB-scanプロット
- **`plot_Bscan_trim.py`**: トリミングされたB-scanプロット

### A-scan・スペクトル可視化
- **`plot_Ascan.py`**: A-scan（単一トレース）プロット
- **`plot_spectrum.py`**: 周波数スペクトル解析・プロット

### 位置・軌道データ
- **`plot_position.py`**: ローバー位置・軌道の可視化

### インタラクティブビューア (`plot_viewer/`)
- **`plot_viewer.py`**: 基本的なインタラクティブプロットビューア
- **`plot_viewer_add_label.py`**: ラベル追加機能付きビューア
- **`plot_viewer_depth_measurement.py`**: 深さ計測機能付きビューア
- **`plot_bscan_ascan_viewer.py`**: B-scan+A-scan並列表示・プロット生成ビューア

### 信号解析ツール
- **`plot_SN_comparison_tool_qt_pure.py`**: Signal-to-Noise比較解析ツール（PyQt5版）

## 使用例

```bash
# B-scanプロット
python visualization/plot_Bscan.py

# 岩石検出結果表示
python visualization/plot_Bscan_detected_rocks.py

# インタラクティブビューア
python visualization/plot_viewer/plot_viewer_add_label.py

# 深さ計測ビューア
python visualization/plot_viewer/plot_viewer_depth_measurement.py

# B-scan+A-scan並列表示ビューア
python visualization/plot_viewer/plot_bscan_ascan_viewer.py

# Signal-to-Noise比較解析ツール
python visualization/plot_SN_comparison_tool_qt_pure.py
```

## 可視化機能

### データ表示
- **時間-距離軸**: ns/m単位での表示
- **深度-距離軸**: 地形補正対応の深度表示
- **カラーマップ**: seismic, viridis, turbo等のカラーマップ

### ラベル・注釈
- **岩石ラベル**: 6種類の岩石タイプの色分け表示
- **統計情報**: ヒストグラム・統計値の重畳表示
- **スケールバー**: 距離・時間スケールの表示

### 出力フォーマット
- **高解像度画像**: PNG (120-300 DPI) + PDF (600 DPI)
- **インタラクティブ表示**: PyQtGraphによるリアルタイム操作
- **複数フォーマット**: データ・画像・PDFの同時出力
- **計測データ**: JSON形式での深さ計測結果・プロット記録出力

## 深さ計測機能

### plot_viewer_depth_measurement.py
B-scan上で縦方向の深さを計測するインタラクティブツールです。

**主な機能:**
- **2点クリック計測**: B-scan上で2回クリックして縦方向の距離を計測
- **複数計測管理**: 複数の計測結果を同時に表示・管理
- **視覚的表示**: 始点と終点を結ぶ直線と時間差[ns]の表示
- **計測結果削除**: 計測線をダブルクリックで削除
- **JSON出力**: 計測データの自動保存（終了時にx座標順でソート）

**深さ計測データ構造:**
```json
{
  "measurements": [
    {
      "id": 1,
      "x_position": 10.5,
      "start_time_ns": 100.0,
      "end_time_ns": 200.0,
      "time_diff_ns": 100.0,
      "depth_m": 5.2
    }
  ]
}
```

**物理定数:**
- sample_interval: 0.312500e-9 [s]
- trace_interval: 3.6e-2 [m]
- epsilon_r: 4.5 (相対誘電率)
- 深さ計算: `depth = (time_diff * c) / (2 * sqrt(epsilon_r))`
- t=0補正: x=0でのNaN以外の最初の値をt=0として設定

**使用方法:**
1. B-scan上で2回クリックして深さを計測
2. 測定線をダブルクリックで削除
3. 測定データは自動的にJSONファイルに保存
4. 終了時にx座標順でIDを振り直して最終保存

## B-scan+A-scan並列表示機能

### plot_bscan_ascan_viewer.py
B-scanとA-scanを横並びで表示し、指定した範囲・位置でmatplotlibプロットを生成するツールです。

**主な機能:**
- **並列表示**: B-scan（左）+ A-scan（右）の横並び表示
- **範囲選択**: Shift+ドラッグでB-scanの表示範囲を選択
- **A-scan位置指定**: 縦線をドラッグしてA-scan表示位置を設定
- **プロット生成**: matplotlib形式でPNG+PDF出力
- **JSON記録**: プロット情報の自動記録・蓄積

**操作方法:**
- **範囲選択**: Shift+ドラッグでB-scan範囲を選択（赤い矩形で表示）
- **A-scan位置**: 白い破線の縦線をドラッグして位置調整
- **パン操作**: マウスドラッグでB-scanを移動
- **ズーム操作**: Ctrl+ドラッグでB-scanを拡大・縮小

**出力仕様:**
- **保存場所**: データファイルと同階層の`Bscan_with_Ascan/`ディレクトリ
- **ファイル名**: `xmin_ymin_xmax_ymax`形式（例：`0.50_10.2_2.30_50.8`）
- **出力形式**: PNG (120 DPI) + PDF (600 DPI)
- **A-scan表示**: 横軸=強度、縦軸=時間[ns]、Amplitude（青）+ Envelope（赤）

**JSON記録データ構造:**
```json
{
  "data_file": "元データファイルのパス",
  "plots": [
    {
      "timestamp": "2025-01-24T10:30:45.123456",
      "filename": "0.50_10.2_2.30_50.8",
      "bscan_range": {
        "x_min": 0.50,
        "x_max": 2.30,
        "y_min": 10.2,
        "y_max": 50.8
      },
      "ascan_position": 1.25,
      "output_files": ["0.50_10.2_2.30_50.8.png", "0.50_10.2_2.30_50.8.pdf"]
    }
  ]
}
```

**物理定数:**
- sample_interval: 0.312500e-9 [s]
- trace_interval: 3.6e-2 [m]
- epsilon_r: 4.5 (相対誘電率)
- t=0補正: x=0でのNaN以外の最初の値をt=0として設定

**使用方法:**
1. コマンド実行後、B-scanデータファイルのパスを入力
2. GUIでShift+ドラッグして表示範囲を選択
3. 縦線をドラッグしてA-scan位置を調整
4. 「プロット生成」ボタンでmatplotlib出力を実行
5. プロット情報は自動的にJSONファイルに記録

## Signal-to-Noise比較解析機能

### plot_SN_comparison_tool_qt_pure.py
B-scanデータ上の任意の点の強度と背景ノイズを比較分析するPyQt5ベースのインタラクティブツールです。

**主な機能:**
- **B-scanデータ表示**: matplotlibベースの対話型プロット表示
- **マウスクリック選択**: 画像上の任意の点をクリックして座標・強度取得
- **グループ管理**: 複数のポイントを事前選択したグループに分類
- **背景データ解析**: run_data_processing.py準拠の背景ノイズ統計解析
- **JSON出力**: ポイント情報の自動保存・蓄積
- **統計プロット**: 背景データとポイントデータの比較可視化

**操作方法:**
- **ファイル入力**: コマンド実行後、B-scanデータファイルとbackground_data.txtのパスを入力
- **グループ選択**: 初期状態でGroup 1が選択済み、新グループボタンで追加可能
- **クリックモード**: 「Enable Click Mode」ボタンでクリック検出を有効化
- **ポイント選択**: B-scan上をクリックして現在のグループにポイント追加
- **統計解析**: 「Generate Analysis」ボタンで背景データとの比較プロット生成

**データ仕様:**
- **B-scanデータ**: 2次元配列のテキストファイル（空白区切り）
- **背景データ**: 1次元配列（各時刻における空間方向平均強度）
- **エンコーディング**: Shift-JIS/UTF-16の自動判定対応

**出力仕様:**
- **保存場所**: B-scanファイルと同階層の`SN_comparison/`ディレクトリ
- **JSONデータ**: ポイント座標・強度・グループ情報
- **統計プロット**: 全時間範囲 + 0-200ns限定の2種類のプロット
- **出力形式**: PNG (120 DPI) + PDF (600 DPI)

**JSON出力データ構造:**
```json
{
  "group_1": [
    {"x": 12.5, "time_ns": 45.2, "amplitude": 3.14},
    {"x": 12.7, "time_ns": 47.8, "amplitude": 2.96}
  ],
  "group_2": [
    {"x": 25.1, "time_ns": 32.4, "amplitude": 4.23}
  ]
}
```

**統計解析機能:**
- **背景データ処理**: log変換による平均値・標準偏差計算
- **グループ別統計**: 各グループの平均値・分散算出
- **比較プロット**: 横軸=log強度、縦軸=時間[ns]
- **エラーバー表示**: 標準偏差によるエラーバー付きscatter表示

**物理定数:**
- sample_interval: 0.312500e-9 [s]
- trace_interval: 3.6e-2 [m]
- epsilon_r: 4.5 (相対誘電率)
- c: 299792458 [m/s] (光速)

**使用方法:**
1. コマンド実行後、B-scanデータファイルのパスを入力
2. background_data.txtのパス入力（同ディレクトリの場合は空エンター）
3. GUIで「Enable Click Mode」ボタンをクリック
4. Group 1が選択された状態でB-scan上の点をクリック
5. 「New Group」ボタンで新しいグループを作成・選択
6. 「Generate Analysis」ボタンで統計解析プロット生成
7. 結果は自動的にSN_comparison/ディレクトリに保存