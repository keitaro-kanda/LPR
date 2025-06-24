# ロバスト双曲線検出システム

断裂・段差・低S/N比データに対応した高精度双曲線検出システム

## 概要

このシステムは以下の特徴を持ちます：

### 主要機能
- **Hough変換 + フィッティング統合**: 初期検出と精密フィッティングの組み合わせ
- **セグメント化処理**: 断裂を含む双曲線の分割検出
- **ギャップ許容フィッティング**: 欠損データに対するロバストフィッティング
- **多基準品質評価**: フィッティング品質、物理的妥当性、データ整合性の総合評価
- **詳細可視化**: 検出結果の多角的分析プロット

### 技術的特徴
- **RANSAC アルゴリズム**: 外れ値に対するロバスト性
- **DBSCAN クラスタリング**: 連続セグメントの自動識別
- **異方性拡散フィルタ**: エッジ保存ノイズ除去
- **適応的閾値処理**: データ特性に応じた動的調整

## ファイル構成

```
tools/detect_hyperbola/
├── robust_hyperbola_fitting.py  # 核となるアルゴリズム実装
├── robust_hyperbola_main.py     # メイン実行プログラム
├── detect_hyperbola_Hough.py    # 既存Hough変換（依存）
├── detect_hyperbola.py          # 既存基本検出（参考）
└── README_robust_detection.md   # 本ドキュメント
```

## 使用方法

### 基本実行
```bash
cd tools/detect_hyperbola/
python robust_hyperbola_main.py
```

**注意**: 現在のバージョンでは、データパスが以下に固定されています：
`/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Processed_Data/4_Gain_function/4_Bscan_gain.txt`

異なるデータファイルを使用する場合は、`robust_hyperbola_main.py`の`bscan_file`変数を変更してください。

### プログラム内での使用
```python
from robust_hyperbola_fitting import RobustHyperbolaDetector

# 検出器初期化（デフォルト設定）
detector = RobustHyperbolaDetector(
    dt_ns=0.3125,           # 時間サンプリング間隔
    dx_m=0.036,             # 空間サンプリング間隔
    epsilon_r=3.0,          # 比誘電率
    max_gap_samples=10,     # 最大ギャップサンプル数（高速化）
    min_segment_points=6,   # セグメント最小点数（高速化）
    snr_threshold=3.0       # S/N比閾値（ノイズ除去厳格化）
)

# 検出実行
results = detector.detect_hyperbolas(
    bscan_data,            # B-scanデータ (2D numpy array)
    hough_peaks=None,      # Houghピーク（オプション）
    max_hyperbolas=10      # 最大検出数（高速化のため削減）
)

# 結果利用
for result in results:
    print(f"品質スコア: {result.quality_score:.3f}")
    print(f"パラメータ: t0={result.params.t0:.1f}, x0={result.params.x0:.1f}")
    physical = result.params.to_physical(0.3125, 0.036)
    print(f"深度: {physical['depth_m']:.3f}m")
```

## 出力結果

### ファイル出力
- `robust_detection_results.json`: 検出結果の詳細データ
- `robust_detection_overview.png/pdf`: 検出結果概要プロット
- `detailed_fitting_quality_*.png`: 個別フィッティング詳細プロット

### JSON結果構造
```json
{
  "detection_summary": {
    "total_detected": 検出総数,
    "high_quality_count": 高品質検出数,
    "average_quality": 平均品質スコア
  },
  "detections": [
    {
      "parameters": {
        "t0_pix": ピクセル単位時間,
        "x0_pix": ピクセル単位位置,
        "v_pix_per_pix": ピクセル単位速度,
        "time_ns": 物理時間[ns],
        "position_m": 物理位置[m],
        "depth_m": 深度[m],
        "velocity_m_ns": 物理速度[m/ns]
      },
      "quality": {
        "total_score": 総合品質スコア,
        "confidence": 信頼度,
        "has_gaps": ギャップ有無,
        "rmse": 二乗平均平方根誤差,
        "inlier_ratio": インライア比率
      },
      "segments": {
        "count": セグメント数,
        "sizes": セグメント別点数
      }
    }
  ]
}
```

## パラメータ調整指針

### 断裂が多い場合
```python
detector = RobustHyperbolaDetector(
    max_gap_samples=20,      # ギャップ許容度を上げる
    min_segment_points=5,    # セグメント最小点数を下げる
    snr_threshold=2.0        # S/N比閾値を下げる
)
```

### ノイズが多い場合
```python
detector = RobustHyperbolaDetector(
    max_gap_samples=10,      # ギャップ許容度を下げる
    min_segment_points=10,   # セグメント最小点数を上げる
    snr_threshold=4.0        # S/N比閾値を上げる
)
```

### 高精度が必要な場合
```python
detector = RobustHyperbolaDetector(
    max_gap_samples=8,       # 厳格なギャップ判定
    min_segment_points=12,   # 十分な点数を要求
    snr_threshold=3.5        # 高S/N比データのみ
)
```

## アルゴリズム詳細

### 処理フロー
1. **信号強調**: エンベロープ抽出 → ウィーナーフィルタ → 異方性拡散
2. **候補抽出**: セグメント化 → 高振幅点抽出 → S/N比フィルタリング
3. **セグメント識別**: DBSCAN クラスタリング → 連続性評価
4. **ロバストフィッティング**: RANSAC → 非線形最小二乗 → インライア評価
5. **品質評価**: 多基準スコアリング → 物理制約チェック → 信頼度算出

### 品質評価基準
- **フィッティング品質**: RMSE逆数ベース
- **データ被覆率**: インライア比率
- **物理的妥当性**: 時間・速度・位置制約
- **セグメント一貫性**: 複数セグメント間の整合性

### 物理制約
- 時間範囲: 10-500 ns (0.8-40m深度相当)
- 速度範囲: 0.1-10 pix/pix
- 位置制約: 非負値

## トラブルシューティング

### 検出数が少ない場合
1. S/N比閾値を下げる (`snr_threshold=2.0`)
2. セグメント最小点数を下げる (`min_segment_points=5`)
3. 品質スコア閾値確認（0.5以上で検出）

### 誤検出が多い場合
1. S/N比閾値を上げる (`snr_threshold=4.0`)
2. セグメント最小点数を上げる (`min_segment_points=12`)
3. 物理制約の強化（時間・速度範囲の調整）

### 処理が遅い場合
1. 最大検出数を制限 (`max_hyperbolas=5`)
2. **データの自動サブサンプリング**: 大規模データは自動的にサブサンプリングされます
   - 6944トレース以上 → factor=2でサブサンプリング  
   - 2000トレース以上 → factor調整でサブサンプリング
3. セグメント最小点数を上げる (`min_segment_points=10`)
4. S/N比閾値を上げる (`snr_threshold=4.0`)

### 大規模データでの高速化設定
```python
detector = RobustHyperbolaDetector(
    max_gap_samples=8,       # ギャップ判定を厳格化
    min_segment_points=8,    # セグメント最小点数を上げる
    snr_threshold=4.0        # 高S/N比データのみ処理
)

# 検出数も制限
results = detector.detect_hyperbolas(bscan_data, max_hyperbolas=5)
```

## 依存関係

### 必須
- numpy
- scipy  
- matplotlib
- scikit-learn

### オプション（Hough変換用）
- tqdm (進捗表示用)
- detect_hyperbola_Hough.py内の関数

### 実装上の注意
- `robust_hyperbola_fitting.py`: 未使用インポートを削除済み
- JSON シリアライゼーション: numpy型を自動的にPython標準型に変換

### インストール
```bash
pip install numpy scipy matplotlib scikit-learn tqdm
```

## パフォーマンス特性

### 処理時間の目安
- **小規模データ** (1600×2000): 約1-2分
- **中規模データ** (1600×6000): 約3-5分（自動サブサンプリング適用）
- **大規模データ** (1600×42000): 約5-10分（factor=7でサブサンプリング）

### メモリ使用量
- Hough変換: データサイズに比例（O(N×M)）
- フィッティング: セグメント数とエッジ点数に依存
- 可視化: matplotlib による画像生成

### 最適化済み機能
- **ベクトル化Hough変換**: バッチ処理で高速化
- **適応的サブサンプリング**: 大規模データの自動最適化
- **エッジ点フィルタリング**: ノイズ除去による処理点数削減
- **JSONシリアライゼーション**: numpy型の自動変換

## 既知の制限事項

1. **処理時間**: 大規模データでは数分かかる場合があります
2. **メモリ消費**: Hough変換でメモリを多く使用します
3. **依存関係**: 既存のHough変換モジュールに依存しています
4. **固定パス**: 現在はデータパスが固定されています

## 今後の拡張可能性

1. **深層学習統合**: CNNベース候補抽出
2. **ベイジアンフィッティング**: 不確実性定量化
3. **リアルタイム処理**: ストリーミングデータ対応
4. **マルチスケール検出**: 異なる解像度での統合検出
5. **インタラクティブ調整**: GUI での parameter tuning
6. **並列処理**: マルチプロセシングによる高速化
7. **GPU加速**: CUDA対応による大幅な高速化