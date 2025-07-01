# Frequency-Wavenumber (F-K) Analysis Tools

周波数-波数（F-K）ドメインでのLPRデータ解析ツール群です。

## 主要ツール

### `fk_transformation.py`
- **機能**: F-K変換の実行（全体変換・窓関数変換）
- **特徴**: 
  - 全体データのF-K変換
  - 窓関数を使用した局所F-K変換
  - B-scanとF-K変換の併記プロット

### `fk_filtering.py`
- **機能**: F-Kドメインでのフィルタリング処理
- **フィルタタイプ**:
  - 三角形フィルタ: (250,0), (750,4), (750,-4) MHz の三角領域
  - 四角形フィルタ: (250,-4), (750,-4), (750,4), (250,4) MHz の矩形領域
- **特徴**: f=0軸対称フィルタリング

### `fk_migration.py`
- **機能**: F-Kマイグレーション処理
- **用途**: 地下構造の空間的再構成

## 使用例

```bash
# F-K変換の実行
python fk_analysis/fk_transformation.py

# F-Kフィルタリング
python fk_analysis/fk_filtering.py

# F-Kマイグレーション
python fk_analysis/fk_migration.py
```

## 技術仕様

- **窓関数処理**: ハニング窓、ガウシアン窓対応
- **対数スケール表示**: dB単位での振幅表示
- **高解像度出力**: PNG (150 DPI) + PDF (300 DPI) 保存
- **NaN値処理**: 堅牢なNaN値ハンドリング