# Core Pipeline Tools

メインのLPRデータ処理パイプラインを構成するコアツール群です。

## 主要ツール

### `run_data_processing.py`
- **機能**: LPRデータ処理パイプラインのメインコントローラー
- **使用方法**: `python core_pipeline/run_data_processing.py`
- **説明**: 6段階の処理（データ統合、バンドパスフィルタ、時間ゼロ補正、背景除去、ゲイン関数、地形補正）を実行

### データ入出力・変換ツール

- **`read_binary_xml.py`**: バイナリデータ（.2B/.2BL）の読み込み
- **`resampling.py`**: データのリサンプリングと品質フィルタリング
- **`convert_terrain_corrected_labels.py`**: 地形補正ラベルの双方向変換

### 処理アルゴリズム

- **`bandpass.py`**: バンドパスフィルタリング (250-750 MHz)
- **`time_zero_correction.py`**: 時間ゼロ補正
- **`remove_background.py`**: 背景信号除去
- **`gain_function.py`**: 距離依存ゲイン補正
- **`terrain_correction.py`**: 地形補正処理

## パラメータ設定

標準的な物理パラメータ:
- `sample_interval = 0.312500e-9` [s] - 時間サンプリング間隔
- `trace_interval = 3.6e-2` [m] - 空間トレース間隔
- `epsilon_r = 4.5` - 月面レゴリスの比誘電率（Zhang et al., 2024）
- `c = 299792458` [m/s] - 光速

## 処理フロー
1. バイナリデータの読み取り（`read_binary_xml.py`）
2. リサンプリング（`resampbling.py`）
3. 各種データ処理（処理手順や使う処理方法は任意に変更可能）
    -  データ統合 (`data_integration`)
    -  バンドパスフィルタ (`bandpass_filter`)
    -  時間ゼロ補正 (`time_zero_correction`)
    -  背景除去 (`background_removal`)
    -  ゲイン関数 (`gain_function`)
    -  地形補正 (`terrain_correction`)