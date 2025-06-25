# ロバスト双曲線検出システム

小スケール岩石双曲線検出に特化したロバスト検出システム

**対象サイズ**: 横幅1-2m、縦幅10-20ns程度の小さな双曲線

## 概要

このシステムは以下の特徴を持ちます：

### 主要機能
- **小スケール特化検出**: 1-2m横幅の小さな双曲線に最適化
- **Hough変換 + フィッティング統合**: 初期検出と精密フィッティングの組み合わせ
- **セグメント化処理**: 断裂を含む双曲線の分割検出
- **ギャップ許容フィッティング**: 欠損データに対するロバストフィッティング
- **多基準品質評価**: フィッティング品質、物理的妥当性、データ整合性の総合評価
- **スケールフィルタリング**: 大きすぎる双曲線の自動除外
- **詳細可視化**: 検出結果の多角的分析プロット（英語ラベル対応）

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

**小スケール検出の特徴**:
- データ範囲: 100ns（時間） × 50m（水平）に制限
- サブサンプリング: 2000トレース以上で適用
- 高い検出閾値: 99パーセンタイル使用

異なるデータファイルを使用する場合は、`robust_hyperbola_main.py`の`bscan_file`変数を変更してください。

### プログラム内での使用
```python
from robust_hyperbola_fitting import RobustHyperbolaDetector

# 検出器初期化（小スケール双曲線用設定）
detector = RobustHyperbolaDetector(
    dt_ns=0.3125,           # 時間サンプリング間隔
    dx_m=0.036,             # 空間サンプリング間隔
    epsilon_r=3.0,          # 比誘電率
    max_gap_samples=5,      # 小さな双曲線用に縮小
    min_segment_points=8,   # より厳格に
    snr_threshold=4.0       # ノイズ除去を厳格化
)

# 検出実行（小スケール用）
results = detector.detect_hyperbolas(
    bscan_data,            # B-scanデータ (2D numpy array)
    hough_peaks=None,      # Houghピーク（オプション）
    max_hyperbolas=50      # 小スケール検出用に増加
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
- `robust_detection_overview.png/pdf`: 検出結果概要プロット（英語ラベル）
- `detailed_fitting_quality_*.png`: 個別フィッティング詳細プロット（小さな双曲線のみ）

### 概要プロットの構成
1. **B-scan + Detection Results**: 原データ + 検出された双曲線（スケールフィルタ適用）
2. **Fitting Quality Distribution**: 品質スコア分布
3. **Depth-Position Distribution**: 検出された岩石の空間分布
4. **Residual Analysis**: フィッティング精度評価

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

### 小スケール双曲線検出（推奨設定）
```python
detector = RobustHyperbolaDetector(
    max_gap_samples=5,       # 小さな双曲線用
    min_segment_points=8,    # 適度な点数要求
    snr_threshold=4.0        # 厳格なノイズ除去
)

# データ範囲制限（100ns × 50m）
bscan_data = load_bscan_data(file, 100.0, 50.0, dt_ns, dx_m)
```

### より小さな双曲線検出
```python
detector = RobustHyperbolaDetector(
    max_gap_samples=3,       # より厳格
    min_segment_points=6,    # 最小限の点数
    snr_threshold=5.0        # 非常に厳格
)
```

### 断裂が多い小双曲線
```python
detector = RobustHyperbolaDetector(
    max_gap_samples=8,       # ギャップ許容
    min_segment_points=6,    # 最小点数を下げる
    snr_threshold=3.5        # 閾値を少し緩める
)
```

### 高ノイズ環境での小双曲線
```python
detector = RobustHyperbolaDetector(
    max_gap_samples=4,       # 厳格なギャップ判定
    min_segment_points=10,   # 十分な点数要求
    snr_threshold=5.0        # 非常に高いS/N比
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
- 時間範囲: 10-100 ns (小スケール用に制限)
- 速度範囲: 0.1-10 pix/pix
- 位置制約: 非負値
- **双曲線サイズフィルタ**: 80トレース（約2.8m）以下の幅のみ表示
- **描画範囲制限**: 頂点±80トレース、頂点+30ns範囲

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
1. 最大検出数を制限 (`max_hyperbolas=20`)
2. **データの自動サブサンプリング**: 大規模データは自動的にサブサンプリングされます
   - 2000トレース以上 → factor調整でサブサンプリング
   - 小スケール検出では控えめなサブサンプリング
3. セグメント最小点数を上げる (`min_segment_points=10`)
4. S/N比閾値を上げる (`snr_threshold=5.0`)
5. **データ範囲制限**: 処理範囲を100ns × 50mに制限

### 小スケール用高速化設定
```python
detector = RobustHyperbolaDetector(
    max_gap_samples=4,       # 厳格なギャップ判定
    min_segment_points=10,   # セグメント最小点数を上げる
    snr_threshold=5.0        # 非常に高S/N比データのみ
)

# データ範囲制限 + 検出数制限
bscan_data = load_bscan_data(file, 50.0, 25.0, dt_ns, dx_m)  # より小さな範囲
results = detector.detect_hyperbolas(bscan_data, max_hyperbolas=20)
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

### 処理時間の目安（小スケール検出）
- **小スケールデータ** (320×1388): 約3-4分
- **中規模データ** (320×2000): 約4-6分（自動サブサンプリング適用）
- **制限範囲** 100ns × 50m: Hough変換とフィッティングの高速化
- **検出結果**: 通常50個程度の双曲線（高品質29個程度）

### メモリ使用量
- Hough変換: データサイズに比例（O(N×M)）
- フィッティング: セグメント数とエッジ点数に依存
- 可視化: matplotlib による画像生成

### 最適化済み機能
- **ベクトル化Hough変換**: バッチ処理で高速化
- **適応的サブサンプリング**: 大規模データの自動最適化
- **エッジ点フィルタリング**: ノイズ除去による処理点数削減
- **JSONシリアライゼーション**: numpy型の自動変換
- **スケール最適化**: 小双曲線用パラメータ調整
- **英語UI**: 可視化の文字化け解決

## 既知の制限事項

1. **処理時間**: 小スケール用でも3-4分程度かかります
2. **メモリ消費**: Hough変換でメモリを多く使用します
3. **依存関係**: 既存のHough変換モジュールに依存しています
4. **固定パス**: 現在はデータパスが固定されています
5. **スケール制限**: 1-2m横幅の双曲線に特化（大きな双曲線は除外）
6. **データ範囲制限**: 100ns × 50m範囲に限定

## 小スケール検出の技術詳細

### スケールフィルタリング機能
- **幅推定**: 頂点から30ns下での双曲線幅を計算
- **サイズ制限**: 80トレース（約2.8m）以下のみ表示
- **描画最適化**: 小さな双曲線用の限定範囲描画

### Hough変換最適化
- **厳格閾値**: 99パーセンタイル使用
- **高ピーク閾値**: 15以上のピークのみ検出
- **データ範囲制限**: 100ns × 50m範囲

### 可視化改善
- **英語ラベル**: 文字化け問題を解決
- **太い線描画**: 小さな双曲線の視認性向上
- **スケールフィルタ**: 大きな双曲線の自動除外

## 今後の拡張可能性

1. **適応的スケール検出**: サイズに応じた自動パラメータ調整
2. **マルチスケール統合**: 小・中・大スケールの統合検出
3. **深層学習統合**: CNNベース小双曲線候補抽出
4. **リアルタイム処理**: ストリーミングデータ対応
5. **インタラクティブ調整**: GUI でのスケール選択
6. **並列処理**: マルチプロセシングによる高速化
7. **GPU加速**: CUDA対応による大幅な高速化
8. **可変範囲設定**: ユーザー指定可能なデータ範囲