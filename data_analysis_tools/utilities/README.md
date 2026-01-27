# Utility Tools

共通ユーティリティ・ヘルパー関数群です。

## 概要

このディレクトリには、複数のツールで共有される共通機能を配置する予定です。

## 今後追加予定の機能

### 共通関数
- **データ読み込み関数**: 標準化されたデータ読み込み
- **パラメータ管理**: 設定ファイル・定数の一元管理
- **エラーハンドリング**: 共通エラー処理関数

### 設定管理
- **`config.py`**: 標準パラメータの定義
- **`constants.py`**: 物理定数・システム定数
- **`path_utils.py`**: パス管理・検証ユーティリティ

### データ処理共通関数
- **`data_utils.py`**: データ前処理・後処理
- **`plot_utils.py`**: プロット共通設定・関数
- **`file_utils.py`**: ファイル操作・検証

## 使用方法

```python
# 将来的な使用例
from utilities.config import SAMPLE_INTERVAL, TRACE_INTERVAL
from utilities.data_utils import load_bscan_data, handle_nan_values
from utilities.plot_utils import setup_plot_style, save_plots
```

現在は空のディレクトリですが、今後のリファクタリングで共通機能を移行していく予定です。