# Statistical and Mathematical Analysis Tools

LPRデータの統計解析・数理解析ツール群です。

## 主要ツール

### 統計解析
- **`plot_rock_statistic.py`**: 岩石ラベルの統計的性質をヒストグラム表示
- **`make_RSFD_plot.py`**: 岩石サイズ頻度分布（RSFD）の作成・プロット
- **`make_RSFD_from_csv.py`**: CSVからRSFDプロットを生成


### その他の解析
- **`calc_RCS.py`**: レーダー散乱断面積（RCS）計算

## 使用例

```bash
# 岩石統計解析
python analysis/plot_rock_statistic.py

# RSFD解析
python analysis/make_RSFD_plot.py

# RCS計算
python analysis/calc_RCS.py
```

## 解析対象

- **岩石分布**: サイズ・深度・水平分布の統計
- **散乱特性**: 電磁波散乱の定量評価
- **サイズ頻度分布**: 岩石サイズの統計的性質

## 統計手法

- **ヒストグラム解析**: 分布の可視化
- **べき則フィッティング**: 累積サイズ分布のモデリング
- **散乱断面積計算**: レーダー反射強度の定量化