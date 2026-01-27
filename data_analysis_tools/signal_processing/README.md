# Signal Processing Tools

LPRデータの信号処理アルゴリズムを実装したツール群です。

## 主要ツール

### フィルタリング・前処理
- **`pulse_compression.py`**: パルス圧縮処理（マッチドフィルタ、ウィーナーフィルタ）
- **`sobel.py`**: Sobelフィルタによる空間微分処理
- **`gradient.py`**: 勾配解析（従来手法と除算手法）
- **`median_filter.py`**: メディアンフィルタによるノイズ除去
- **`envelope_median_filter.py`**: エンベロープ計算とメディアンフィルタの組み合わせ処理

### 信号解析
- **`hilbert.py`**: ヒルベルト変換によるエンベロープ抽出
- **`calc_acorr.py`**: 自己相関解析

## 使用例

```bash
# パルス圧縮処理
python signal_processing/pulse_compression.py

# 勾配解析
python signal_processing/gradient.py

# エンベロープ計算とメディアンフィルタ
python signal_processing/envelope_median_filter.py

# 自己相関解析  
python signal_processing/calc_acorr.py
```

## 技術的特徴

- **NaN値対応**: 全ツールでNaN値を適切に処理
- **進捗表示**: tqdmによる処理進捗の可視化
- **複数フォーマット出力**: PNG/PDF形式でのプロット保存
- **標準化されたパラメータ**: 共通の物理定数を使用