# Detection Algorithms

LPRデータからの特徴検出・物体検出アルゴリズム群です。

## 主要ツール

### 双曲線検出 (`detect_hyperbola/`)
- **`detect_hyperbola.py`**: 基本的な双曲線検出アルゴリズム
- **`detect_hyperbola_Hough.py`**: Hough変換による双曲線検出
- **`robust_hyperbola_fitting.py`**: ロバスト双曲線フィッティング
- **`robust_hyperbola_main.py`**: ロバスト検出のメインスクリプト

### 類似度・相関解析 (`local_similarity/`)
- **`calc_local_similarity.py`**: 局所類似度計算
- **`calc_cross_corr.py`**: 相互相関解析
- **`calc_local_normalized_crosscorr.py`**: 正規化相互相関
- **`calc_cross.py`**: 基本的な相互相関処理

### その他の検出アルゴリズム
- **`detect_peak.py`**: ピーク検出アルゴリズム
- **`hyperbole_detection_plot_boundingbox.py`**: バウンディングボックス表示

## 使用例

```bash
# 双曲線検出
python detection/detect_hyperbola/detect_hyperbola.py

# Hough変換による双曲線検出
python detection/detect_hyperbola/detect_hyperbola_Hough.py

# 局所類似度解析
python detection/local_similarity/calc_local_similarity.py

# 相互相関解析
python detection/local_similarity/calc_cross_corr.py

# ピーク検出
python detection/detect_peak.py
```

## 検出対象

- **地下岩石**: 双曲線状の反射パターン
- **層構造**: 水平方向の連続性
- **異常構造**: 通常パターンからの逸脱
- **信号類似性**: トレース間の相関・類似度パターン

## アルゴリズム特徴

- **マルチスケール処理**: 複数の空間スケールでの検出
- **ノイズ耐性**: メディアンフィルタ等による前処理
- **統計的検証**: 検出結果の信頼性評価
- **相関解析**: 空間的・時間的相関による特徴抽出