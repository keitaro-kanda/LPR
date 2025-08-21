# SN比較ツール（Signal-to-Noise Comparison Tool）技術仕様書

## 概要

B-scanデータ上の任意の点の強度と背景ノイズの平均・分散を比較するためのGUIベースの対話型ツール。双曲線エコーなどの特徴的な信号の強度を定量的に分析する。

## 機能要求

### 1. データ入力・読み込み
- **B-scanファイル読み込み**: `input()`によるtxtファイルパス入力
- **背景データ読み込み**: 同ディレクトリの`background_data.txt`を自動読み込み
- **データ形式**: 1次元データ（各時刻における空間方向平均強度）
- **エンコーディング対応**: Shift-JIS/UTF-16の自動判定・読み込み

### 2. GUI表示機能
- **B-scanデータ表示**: matplotlib/tkinter等を用いた対話型プロット
- **マウスクリック対応**: 画像上の任意の点をクリックして座標・強度取得
- **リアルタイム表示**: クリック位置の座標(x, t)と強度値をリアルタイム表示
- **複数点選択**: 連続的なポイント選択が可能

### 3. データ管理・保存機能
- **ポイントグルーピング**: 複数のポイントを1つのグループとして管理
- **JSON出力**: 各グループのポイント情報をJSON形式で保存
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
- **ファイル保存場所**: B-scanファイルと同ディレクトリ

### 4. 統計解析・可視化機能
- **背景データプロット**: run_data_processing.py（255-258行）と同様の形式
  - 横軸: log強度
  - 縦軸: 時間[ns]
  - 背景データの平均値・分散を表示
- **ポイントプロット重畳**: JSONファイルの全ポイントをscatterで重畳表示
- **グループ別統計**: 各グループの平均値・分散を計算・表示
- **エラーバー表示**: グループごとの統計値をエラーバー付きscatterで可視化
- **自動保存**: `SN_comparison`ディレクトリに結果プロットを自動保存
- **複数プロット生成**: 全時間範囲（全範囲）と制限時間範囲（0-200ns）の2つのプロットを作成

## 技術仕様

### システム要求
- Python 3.8+
- 必要ライブラリ: numpy, matplotlib, json, tkinter (GUI), os, sys

### ファイル命名規則
- ツール名: `plot_SN_comparison_tool.py`
- 配置場所: `/tools/visualization/`

### データ処理パラメータ
```python
sample_interval = 0.312500e-9  # [s] - 時間サンプリング間隔
trace_interval = 3.6e-2        # [m] - 空間トレース間隔
c = 299792458                  # [m/s] - 光速
epsilon_r = 4.5               # 相対誘電率
```

### GUI操作フロー
1. ファイルパス入力 → B-scan表示
2. グループ選択（初期状態：Group 1が選択済み）
3. マウスクリックでポイント選択 → 現在選択中のグループに追加
4. 'n'キーで新グループ作成・選択（Group 2, Group 3...）
5. 'e'キーでツール終了
6. 統計解析プロット自動生成・保存

### 出力ディレクトリ構造
```
B-scan_file_directory/
├── input_bscan.txt
├── background_data.txt
├── point_data.json
└── SN_comparison/
    ├── background_analysis_full.png
    ├── background_analysis_full.pdf
    ├── background_analysis_0-200ns.png
    ├── background_analysis_0-200ns.pdf
    ├── SN_comparison_plot_full.png
    └── SN_comparison_plot_0-200ns.png
```

## run_data_processing.pyとの連携

### 背景データ処理（255-258行参照）
```python
background_removed_data_log = np.log(np.abs(background_removed_data))
background_removed_log_ave = np.mean(background_removed_data_log, axis=1)
background_removed_log_std = np.std(background_removed_data_log, axis=1)
```
- 同様のlog変換・統計処理を実装
- プロット形式の統一（横軸: log強度、縦軸: 時間[ns]）

### フォント・表示設定統一
```python
font_large = 20      # タイトル・ラベル
font_medium = 18     # 軸ラベル
font_small = 16      # 目盛りラベル
```

## 詳細仕様確認・更新

### 1. GUI操作・グループ管理
- **グループ事前選択方式**: ポイント選択前にグループを選択し、そのグループに追加
- **初期状態**: Group 1が自動選択済み
- **新グループ作成**: 'n'キーでGroup 2, Group 3...を順次作成・選択
- **操作継続**: 選択中のグループにポイントを連続追加可能

### 2. 背景データ仕様
- **データ形式**: 1次元データ（各時刻における空間方向平均強度）
- **ファイル形式**: 1列のテキストファイル
- **読み込み**: B-scanファイルと同ディレクトリから自動読み込み

### 3. 強度値取得・時間変換
- **取得方法**: クリック位置の最近傍サンプル点の値を取得
- **時間分解能**: 0.312500e-9秒（CLAUDE.mdより）
- **変換式**: `time_ns = sample_index * sample_interval * 1e9`

### 4. プロット生成仕様
- **全時間範囲プロット**: データ全体の時間範囲を表示
- **制限時間範囲プロット**: 0-200nsに限定した表示
- **両方自動生成**: 1回の実行で2種類のプロットを生成・保存

## 追加機能案

### オプション機能
- undo機能（最後のポイント削除）
- グループ名の手動設定
- CSVエクスポート機能
- プロット上でのグループ色分け表示

## 実装準備完了

**確認された詳細仕様：**
1. ✅ グループ事前選択方式によるポイント管理
2. ✅ 背景データの1次元形式（空間方向平均強度）
3. ✅ 最近傍サンプル点による強度値取得
4. ✅ 0.312500e-9秒の時間分解能
5. ✅ 全範囲・制限範囲（0-200ns）の2つのプロット生成

これらの仕様に基づいて実装を進める準備が整いました。追加の仕様変更や不明な点がございましたらお知らせください。