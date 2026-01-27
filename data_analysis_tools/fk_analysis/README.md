# Frequency-Wavenumber (F-K) Analysis Tools

周波数-波数（F-K）ドメインでのLPRデータ解析ツール群です。

## 主要ツール

### `fk_transformation.py`
- **機能**: F-K変換の実行（全体変換・窓関数変換・テストモード）
- **特徴**: 
  - 全体データのF-K変換
  - 窓関数を使用した局所F-K変換
  - テストモード: 代表点での区切りF-K変換
  - B-scanとF-K変換の併記プロット
  - NaN含有データの正確な時間インデックス処理

### `fk_filtering.py`
- **機能**: F-Kドメインでのフィルタリング処理（全体・窓関数・テストモード）
- **フィルタタイプ**:
  - 三角形フィルタ: (250,0), (750,4), (750,-4) MHz の三角領域
  - 四角形フィルタ: (250,-4), (750,-4), (750,4), (250,4) MHz の矩形領域
- **特徴**: 
  - f=0軸対称フィルタリング
  - テストモード: 代表点での区切りF-Kフィルタリング
  - フィルタ前後の比較プロット出力
  - NaN含有データの正確な時間インデックス処理

### `fk_migration.py`
- **機能**: F-Kマイグレーション処理
- **用途**: 地下構造の空間的再構成

## 使用例

```bash
# F-K変換の実行
python fk_analysis/fk_transformation.py
# モード選択: 1=全体, 2=区切り, 3=テスト

# F-Kフィルタリング
python fk_analysis/fk_filtering.py
# フィルタ選択: 1=三角形, 2=四角形

# F-Kマイグレーション
python fk_analysis/fk_migration.py
```

## 出力ディレクトリ構造

```
fk_transformation/
├── full_fk_transform.png              # 全体F-K変換
└── windowed_results_x{m}_t{ns}/        # 区切り処理結果
    ├── window_x{m}_t{ns}_combined.png  # 各窓の結果
    └── test/                           # テストモード
        ├── x={m}_t={ns}.png           # 代表点結果(PNG)
        └── x={m}_t={ns}.pdf           # 代表点結果(PDF)

fk_filtering/
├── full_fk_filtering_{filter}.png      # 全体フィルタリング
└── windowed_results_x{m}_t{ns}_{filter}/
    ├── window_x{m}_t{ns}_filtered_{filter}.png
    ├── window_x{m}_t{ns}_filtered_{filter}.txt
    └── test/
        ├── x={m}_t={ns}.png
        └── x={m}_t={ns}.pdf
```

## 処理モード

### 1. 全体処理モード
- データ全体でのF-K変換・フィルタリング
- 全範囲の解析に適用

### 2. 区切り処理モード
- 任意の窓サイズでデータを分割して処理
- 局所的な解析が可能
- 時間・空間方向の窓サイズを設定可能

### 3. テストモード
- 代表的な地点での区切り処理
- terrain_corrected データ用の固定座標:
  - (322,65), (364,70), (247,65), (600,90), (1180,90)
  - (710,130), (850,80), (325,190), (1050,380) [x:m, t:ns]
- PNG/PDF両形式での出力

## 技術仕様

- **窓関数処理**: ハニング窓、ガウシアン窓対応
- **対数スケール表示**: dB単位での振幅表示
- **高解像度出力**: PNG (150 DPI) + PDF (300 DPI) 保存
- **NaN値処理**: 堅牢なNaN値ハンドリング
- **時間インデックス補正**: x=0での最初の有効データをt=0として処理