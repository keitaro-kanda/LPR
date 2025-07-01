# Testing and Experimental Tools

テスト・実験的なコード群です。

## テストツール

### 機能テスト (`test/`)
- **`hyperbola_detection_test.py`**: 双曲線検出アルゴリズムのテスト
- **`plot_hyperbola_shape_test.py`**: 双曲線形状プロットのテスト
- **`extract_rock_Hu2019.py`**: Hu2019手法による岩石抽出テスト

### 汎用テスト
- **`tool_test.py`**: 各種ツールの動作テスト

## 使用方法

```bash
# 双曲線検出テスト
python testing/test/hyperbola_detection_test.py

# 汎用ツールテスト
python testing/tool_test.py
```

## テスト対象

- **アルゴリズム検証**: 検出精度・処理速度の評価
- **データ整合性**: 入出力データの整合性確認
- **エラーハンドリング**: 異常入力に対する動作確認

## 注意事項

このディレクトリのコードは実験的・テスト目的であり、本格的な解析には使用しないでください。