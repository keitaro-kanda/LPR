#!/usr/bin/env python3
"""
ロバスト双曲線検出メインプログラム
Hough変換 + フィッティング統合版
"""

import numpy as np
import os
import time
import json
from typing import List, Dict
import matplotlib.pyplot as plt
import warnings

# ローカルモジュール
from robust_hyperbola_fitting import RobustHyperbolaDetector, DetectionResult, HyperbolaParams

# 既存のHough変換関数をインポート（detect_hyperbola_Hough.pyから）
try:
    from detect_hyperbola_Hough import (
        calculate_velocity_params, load_bscan_data, 
        perform_hough_transform, find_peaks_in_accumulator
    )
    HOUGH_AVAILABLE = True
except ImportError:
    print("警告: Hough変換モジュールが見つかりません。フィッティングのみで実行します。")
    HOUGH_AVAILABLE = False

warnings.filterwarnings('ignore')

class HyperbolaVisualization:
    """双曲線検出結果の可視化クラス"""
    
    def __init__(self, dt_ns: float = 0.3125, dx_m: float = 0.036):
        self.dt_ns = dt_ns
        self.dx_m = dx_m
    
    def plot_detection_overview(self, bscan_data: np.ndarray,
                              results: List[DetectionResult],
                              hough_peaks: List[Dict] = None,
                              output_dir: str = ".") -> None:
        """検出結果概要プロット"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ロバスト双曲線検出結果', fontsize=16)
        
        # B-scanデータ
        ax1 = axes[0, 0]
        self._plot_bscan_with_detections(ax1, bscan_data, results, hough_peaks)
        ax1.set_title('B-scan + 検出結果')
        
        # フィッティング品質分布
        ax2 = axes[0, 1]
        self._plot_quality_distribution(ax2, results)
        ax2.set_title('フィッティング品質分布')
        
        # 深度-位置分布
        ax3 = axes[1, 0]
        self._plot_depth_position_distribution(ax3, results)
        ax3.set_title('深度-位置分布')
        
        # 残差解析
        ax4 = axes[1, 1]
        self._plot_residual_analysis(ax4, results)
        ax4.set_title('残差解析')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "robust_detection_overview.png"), dpi=150)
        plt.savefig(os.path.join(output_dir, "robust_detection_overview.pdf"), dpi=300)
        plt.show()
    
    def _plot_bscan_with_detections(self, ax, bscan_data, results, hough_peaks):
        """B-scan + 検出結果"""
        height, width = bscan_data.shape
        
        # B-scan表示
        ax.imshow(bscan_data, aspect='auto', cmap='viridis',
                      extent=[0, width * self.dx_m, height * self.dt_ns, 0],
                      vmin=-np.max(np.abs(bscan_data))/15,
                      vmax=np.max(np.abs(bscan_data))/15)
        
        # Hough検出結果（があれば）
        if hough_peaks:
            hough_x = [p['x0_pix'] * self.dx_m for p in hough_peaks]
            hough_t = [p['t0_pix'] * self.dt_ns for p in hough_peaks]
            ax.scatter(hough_x, hough_t, c='cyan', marker='x', s=50, 
                      label=f'Hough peaks ({len(hough_peaks)})', alpha=0.7)
        
        # フィッティング結果
        if results:
            fit_x = [r.params.x0 * self.dx_m for r in results]
            fit_t = [r.params.t0 * self.dt_ns for r in results]
            qualities = [r.quality_score for r in results]
            
            scatter = ax.scatter(fit_x, fit_t, c=qualities, cmap='Reds', 
                               marker='o', s=100, edgecolors='black', linewidth=1,
                               label=f'Fitted hyperbolas ({len(results)})')
            plt.colorbar(scatter, ax=ax, label='Quality Score')
            
            # 双曲線描画
            for result in results[:5]:  # 上位5個のみ
                if result.quality_score > 0.6:
                    self._draw_hyperbola(ax, result.params, width, height, 
                                       alpha=0.6, color=plt.cm.Reds(result.quality_score))
        
        ax.set_xlabel('距離 (m)')
        ax.set_ylabel('時間 (ns)')
        ax.legend()
    
    def _draw_hyperbola(self, ax, params: HyperbolaParams, width: int, height: int,
                       alpha: float = 0.6, color: str = 'red'):
        """双曲線描画"""
        x_range = np.linspace(0, width, 200)
        try:
            t_curve = np.sqrt(params.t0**2 + (x_range - params.x0)**2 / params.v**2)
            
            # 物理単位に変換
            x_physical = x_range * self.dx_m
            t_physical = t_curve * self.dt_ns
            
            # 有効範囲内のみプロット
            valid_mask = (t_curve < height) & (t_curve > 0)
            if np.any(valid_mask):
                ax.plot(x_physical[valid_mask], t_physical[valid_mask], 
                       color=color, alpha=alpha, linewidth=2)
        except Exception:
            pass
    
    def _plot_quality_distribution(self, ax, results):
        """品質分布プロット"""
        if not results:
            ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
            return
        
        qualities = [r.quality_score for r in results]
        
        ax.hist(qualities, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(qualities), color='red', linestyle='--', 
                  label=f'平均: {np.mean(qualities):.3f}')
        ax.set_xlabel('品質スコア')
        ax.set_ylabel('度数')
        ax.legend()
    
    def _plot_depth_position_distribution(self, ax, results):
        """深度-位置分布"""
        if not results:
            ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
            return
        
        positions = [r.params.x0 * self.dx_m for r in results]
        depths = [r.params.to_physical(self.dt_ns, self.dx_m)['depth_m'] for r in results]
        qualities = [r.quality_score for r in results]
        
        scatter = ax.scatter(positions, depths, c=qualities, cmap='viridis', 
                           s=60, alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, ax=ax, label='Quality Score')
        
        ax.set_xlabel('位置 (m)')
        ax.set_ylabel('深度 (m)')
        ax.invert_yaxis()
    
    def _plot_residual_analysis(self, ax, results):
        """残差解析"""
        if not results:
            ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
            return
        
        all_residuals = []
        for result in results:
            if len(result.fit_residuals) > 0:
                all_residuals.extend(result.fit_residuals)
        
        if all_residuals:
            ax.hist(all_residuals, bins=30, alpha=0.7, density=True, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', label='理想値')
            ax.axvline(np.mean(all_residuals), color='orange', linestyle='--', 
                      label=f'平均: {np.mean(all_residuals):.3f}')
            ax.set_xlabel('残差 (samples)')
            ax.set_ylabel('密度')
            ax.legend()
    
    def plot_detailed_fitting(self, bscan_data: np.ndarray, 
                            result: DetectionResult,
                            output_dir: str = ".") -> None:
        """詳細フィッティング結果"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'詳細フィッティング結果 (品質: {result.quality_score:.3f})', fontsize=14)
        
        # 元データ + フィット
        ax1 = axes[0, 0]
        self._plot_fit_on_bscan(ax1, bscan_data, result)
        ax1.set_title('B-scan + フィット結果')
        
        # セグメント分析
        ax2 = axes[0, 1]
        self._plot_segment_analysis(ax2, result)
        ax2.set_title('セグメント分析')
        
        # 残差分布
        ax3 = axes[1, 0]
        self._plot_residual_distribution(ax3, result)
        ax3.set_title('残差分布')
        
        # パラメータ情報
        ax4 = axes[1, 1]
        self._plot_parameter_info(ax4, result)
        ax4.set_title('パラメータ情報')
        
        plt.tight_layout()
        
        filename = f"detailed_fitting_quality_{result.quality_score:.3f}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.show()
    
    def _plot_fit_on_bscan(self, ax, bscan_data, result):
        """B-scan上のフィット結果"""
        height, width = bscan_data.shape
        
        # B-scan表示
        ax.imshow(bscan_data, aspect='auto', cmap='gray',
                 extent=[0, width * self.dx_m, height * self.dt_ns, 0],
                 alpha=0.7)
        
        # フィット双曲線
        self._draw_hyperbola(ax, result.params, width, height, 
                           alpha=1.0, color='red')
        
        # インライア点
        if hasattr(result, 'segments') and result.segments:
            colors = plt.cm.Set1(np.linspace(0, 1, len(result.segments)))
            for i, segment in enumerate(result.segments):
                seg_x = segment['x'] * self.dx_m
                seg_t = segment['t'] * self.dt_ns
                ax.scatter(seg_x, seg_t, c=[colors[i]], s=20, alpha=0.7,
                          label=f'Segment {i+1} ({segment["size"]} pts)')
        
        ax.set_xlabel('距離 (m)')
        ax.set_ylabel('時間 (ns)')
        ax.legend()
    
    def _plot_segment_analysis(self, ax, result):
        """セグメント分析"""
        if not hasattr(result, 'segments') or not result.segments:
            ax.text(0.5, 0.5, 'セグメント情報なし', ha='center', va='center', transform=ax.transAxes)
            return
        
        segment_sizes = [s['size'] for s in result.segments]
        segment_labels = [f'Seg {i+1}' for i in range(len(result.segments))]
        
        bars = ax.bar(segment_labels, segment_sizes, alpha=0.7)
        ax.set_ylabel('点数')
        ax.set_title('セグメント別点数')
        
        # 値をバーの上に表示
        for bar, size in zip(bars, segment_sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{size}', ha='center', va='bottom')
    
    def _plot_residual_distribution(self, ax, result):
        """残差分布"""
        if len(result.fit_residuals) == 0:
            ax.text(0.5, 0.5, '残差データなし', ha='center', va='center', transform=ax.transAxes)
            return
        
        residuals = result.fit_residuals
        ax.hist(residuals, bins=20, alpha=0.7, density=True, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', label='理想値')
        ax.axvline(np.mean(residuals), color='orange', linestyle='--',
                  label=f'平均: {np.mean(residuals):.3f}')
        
        # 統計情報
        std_res = np.std(residuals)
        ax.axvline(std_res, color='green', linestyle=':', label=f'±1σ: {std_res:.3f}')
        ax.axvline(-std_res, color='green', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('残差 (samples)')
        ax.set_ylabel('密度')
        ax.legend()
    
    def _plot_parameter_info(self, ax, result):
        """パラメータ情報表示"""
        ax.axis('off')
        
        # 物理パラメータ計算
        physical = result.params.to_physical(self.dt_ns, self.dx_m)
        
        info_text = f"""
パラメータ情報:
─────────────────
ピクセル単位:
  t0: {result.params.t0:.1f} samples
  x0: {result.params.x0:.1f} traces  
  v:  {result.params.v:.4f} pix/pix

物理単位:
  時間: {physical['time_ns']:.1f} ns
  位置: {physical['position_m']:.2f} m
  深度: {physical['depth_m']:.3f} m
  速度: {physical['velocity_m_ns']:.4f} m/ns

品質情報:
─────────────────
  総合スコア: {result.quality_score:.3f}
  信頼度: {result.confidence:.3f}
  ギャップ有無: {'有' if result.has_gaps else '無'}
  セグメント数: {len(result.segments) if hasattr(result, 'segments') else 'N/A'}
  
残差統計:
  RMSE: {np.sqrt(np.mean(result.fit_residuals**2)):.3f}
  インライア率: {np.mean(result.inlier_mask):.1%}
        """
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace')

def save_detection_results(results: List[DetectionResult], 
                         output_dir: str,
                         filename: str = "robust_detection_results.json") -> None:
    """検出結果をJSONで保存"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # シリアライズ可能な形式に変換
    serializable_results = []
    for result in results:
        physical = result.params.to_physical(0.3125, 0.036)
        
        result_dict = {
            'parameters': {
                't0_pix': float(result.params.t0),
                'x0_pix': float(result.params.x0),
                'v_pix_per_pix': float(result.params.v),
                'time_ns': physical['time_ns'],
                'position_m': physical['position_m'],
                'depth_m': physical['depth_m'],
                'velocity_m_ns': physical['velocity_m_ns']
            },
            'quality': {
                'total_score': float(result.quality_score),
                'confidence': float(result.confidence),
                'has_gaps': bool(result.has_gaps),
                'rmse': float(np.sqrt(np.mean(result.fit_residuals**2))),
                'inlier_ratio': float(np.mean(result.inlier_mask))
            },
            'segments': {
                'count': int(len(result.segments)) if hasattr(result, 'segments') else 0,
                'sizes': [int(s['size']) for s in result.segments] if hasattr(result, 'segments') else []
            }
        }
        serializable_results.append(result_dict)
    
    # JSON保存
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'detection_summary': {
                'total_detected': int(len(results)),
                'high_quality_count': int(sum(1 for r in results if r.quality_score > 0.7)),
                'average_quality': float(np.mean([r.quality_score for r in results])) if results else 0.0
            },
            'detections': serializable_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"検出結果を保存しました: {filepath}")

def main():
    """メイン処理"""
    print("=== ロバスト双曲線検出システム ===")
    
    # パラメータ設定
    DT_NS = 0.3125
    DX_M = 0.036
    EPSILON_R = 3.0
    
    # データファイル入力（固定パス）
    bscan_file = "/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Processed_Data/4_Gain_function/4_Bscan_gain.txt"
    print(f"データファイル: {bscan_file}")
    
    if not os.path.isfile(bscan_file):
        print(f"エラー: ファイルが見つかりません: {bscan_file}")
        return
    
    # 出力ディレクトリ設定
    output_dir = os.path.join(os.path.dirname(bscan_file), "robust_hyperbola_detection")
    os.makedirs(output_dir, exist_ok=True)
    
    # データ読み込み
    print("データを読み込み中...")
    try:
        if HOUGH_AVAILABLE:
            bscan_data = load_bscan_data(bscan_file, 500.0, 500.0, DT_NS, DX_M)  # 水平方向も制限
        else:
            bscan_data = np.loadtxt(bscan_file)
            max_samples = int(500.0 / DT_NS)
            max_traces = int(500.0 / DX_M)
            bscan_data = bscan_data[:max_samples, :max_traces]
        
        if bscan_data is None or bscan_data.size == 0:
            print("データ読み込みエラー")
            return
        
        # さらにサブサンプリング（高速化のため）
        if bscan_data.shape[1] > 5000:
            subsample_factor = max(1, bscan_data.shape[1] // 5000)
            print(f"データサイズが大きいため、サブサンプリング（factor={subsample_factor}）を実行...")
            bscan_data = bscan_data[:, ::subsample_factor]
            
        print(f"処理データサイズ: {bscan_data.shape}")
        
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return
    
    # Hough変換（可能な場合）
    hough_peaks = []
    if HOUGH_AVAILABLE:
        print("Hough変換を実行中...")
        try:
            _, v_hough = calculate_velocity_params(EPSILON_R, DT_NS, DX_M)
            accumulator = perform_hough_transform(bscan_data, v_hough, 95)
            hough_peaks = find_peaks_in_accumulator(accumulator, None, 7)
            print(f"Hough変換で検出されたピーク数: {len(hough_peaks)}")
        except Exception as e:
            print(f"Hough変換エラー: {e}")
    
    # ロバストフィッティング
    print("ロバストフィッティングを実行中...")
    detector = RobustHyperbolaDetector(
        dt_ns=DT_NS, 
        dx_m=DX_M, 
        epsilon_r=EPSILON_R,
        max_gap_samples=10,     # 高速化のため縮小
        min_segment_points=6,   # 高速化のため縮小
        snr_threshold=3.0       # ノイズ除去を厳格化
    )
    
    start_time = time.time()
    results = detector.detect_hyperbolas(bscan_data, hough_peaks, max_hyperbolas=10)
    end_time = time.time()
    
    print(f"処理時間: {end_time - start_time:.2f}秒")
    print(f"検出された双曲線数: {len(results)}")
    
    if results:
        high_quality = [r for r in results if r.quality_score > 0.7]
        print(f"高品質検出数 (>0.7): {len(high_quality)}")
        print(f"平均品質スコア: {np.mean([r.quality_score for r in results]):.3f}")
    
    # 結果保存
    save_detection_results(results, output_dir)
    
    # 可視化
    print("可視化を作成中...")
    viz = HyperbolaVisualization(DT_NS, DX_M)
    
    # 概要プロット
    viz.plot_detection_overview(bscan_data, results, hough_peaks, output_dir)
    
    # 詳細プロット（上位3個）
    for result in results[:3]:
        if result.quality_score > 0.6:
            viz.plot_detailed_fitting(bscan_data, result, output_dir)
    
    print(f"処理完了。結果は {output_dir} に保存されました。")

if __name__ == "__main__":
    main()