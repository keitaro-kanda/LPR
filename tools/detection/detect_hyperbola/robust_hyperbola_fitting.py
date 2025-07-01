#!/usr/bin/env python3
"""
ロバスト双曲線フィッティングによる自動検出システム
断裂・段差・低S/N比データ対応版
"""

import numpy as np
import warnings
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.optimize import least_squares
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter
from sklearn.cluster import DBSCAN
from sklearn.base import BaseEstimator, RegressorMixin

# 警告を抑制
warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class HyperbolaParams:
    """双曲線パラメータクラス"""
    t0: float  # 頂点時間 [samples]
    x0: float  # 頂点位置 [traces]
    v: float   # 実効速度 [pix/pix]
    
    def to_physical(self, dt_ns: float, dx_m: float, epsilon_r: float = 3.0) -> Dict[str, float]:
        """物理単位への変換"""
        c_m_ns = 0.299792458
        v_medium = c_m_ns / np.sqrt(epsilon_r)
        
        return {
            'time_ns': self.t0 * dt_ns,
            'position_m': self.x0 * dx_m,
            'depth_m': v_medium * (self.t0 * dt_ns) * 1e-9 / 2,
            'velocity_m_ns': self.v * dx_m / dt_ns / 2
        }

@dataclass
class DetectionResult:
    """検出結果クラス"""
    params: HyperbolaParams
    quality_score: float
    fit_residuals: np.ndarray
    inlier_mask: np.ndarray
    segments: List[Dict]
    has_gaps: bool
    confidence: float
    
class HyperbolaRegressor(BaseEstimator, RegressorMixin):
    """双曲線回帰器（scikit-learn互換）"""
    
    def __init__(self, bounds=None):
        self.bounds = bounds or ([0, -np.inf, 0.1], [np.inf, np.inf, 10])
        
    def fit(self, X, y):
        """フィッティング実行"""
        # X: (n_samples, 1) - x座標
        # y: (n_samples,) - t座標
        
        x_data = X.ravel()
        t_data = y
        
        # 初期値推定
        t0_init = np.min(t_data)
        x0_init = x_data[np.argmin(t_data)]
        v_init = self._estimate_velocity(x_data, t_data, x0_init, t0_init)
        
        initial_params = [t0_init, x0_init, v_init]
        
        try:
            result = least_squares(
                self._residuals,
                initial_params,
                args=(t_data, x_data),
                bounds=self.bounds,
                method='trf'
            )
            
            if result.success:
                self.params_ = HyperbolaParams(*result.x)
                self.cost_ = result.cost
                return self
            else:
                raise ValueError("Optimization failed")
                
        except Exception:
            # フォールバック：線形近似
            self.params_ = HyperbolaParams(t0_init, x0_init, v_init)
            self.cost_ = np.inf
            return self
    
    def predict(self, X):
        """予測値計算"""
        x_data = X.ravel()
        return self._hyperbola_model(x_data, self.params_.t0, self.params_.x0, self.params_.v)
    
    def score(self, X, y):
        """R²スコア計算"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-10))
    
    @staticmethod
    def _hyperbola_model(x, t0, x0, v):
        """双曲線モデル"""
        return np.sqrt(t0**2 + (x - x0)**2 / v**2)
    
    @staticmethod
    def _residuals(params, t_data, x_data):
        """残差計算"""
        t0, x0, v = params
        t_pred = HyperbolaRegressor._hyperbola_model(x_data, t0, x0, v)
        return t_data - t_pred
    
    @staticmethod
    def _estimate_velocity(x_data, t_data, x0, t0):
        """速度の初期推定"""
        far_mask = t_data > t0 * 1.2
        if np.sum(far_mask) < 3:
            return 1.0
        
        x_far = x_data[far_mask]
        t_far = t_data[far_mask]
        
        # 漸近線からの推定
        dx = np.abs(x_far - x0)
        dt_sq = t_far**2 - t0**2
        dt_sq = np.maximum(dt_sq, 1e-10)
        
        v_estimates = dx / np.sqrt(dt_sq)
        return np.median(v_estimates[v_estimates > 0.1])

class RobustHyperbolaDetector:
    """ロバスト双曲線検出器"""
    
    def __init__(self, 
                 dt_ns: float = 0.3125,
                 dx_m: float = 0.036,
                 epsilon_r: float = 3.0,
                 max_gap_samples: int = 10,
                 min_segment_points: int = 5,
                 snr_threshold: float = 3.0):
        
        self.dt_ns = dt_ns
        self.dx_m = dx_m
        self.epsilon_r = epsilon_r
        self.max_gap_samples = max_gap_samples
        self.min_segment_points = min_segment_points
        self.snr_threshold = snr_threshold
        
        # 物理パラメータ計算
        c_m_ns = 0.299792458
        self.v_medium = c_m_ns / np.sqrt(epsilon_r)
        self.v_hough_pix_per_pix = (self.v_medium * dt_ns) / (2 * dx_m)
        
        print(f"ロバスト検出器初期化:")
        print(f"  dt={dt_ns}ns, dx={dx_m}m, εr={epsilon_r}")
        print(f"  v_medium={self.v_medium:.4f}m/ns")
        print(f"  v_hough={self.v_hough_pix_per_pix:.6f}pix/pix")
    
    def robust_signal_enhancement(self, bscan_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ロバスト信号強調"""
        print("信号強調処理を開始...")
        
        # 1. エンベロープ計算
        envelope = np.abs(hilbert(bscan_data, axis=0))
        
        # 2. ノイズレベル推定
        noise_samples = min(50, bscan_data.shape[0] // 10)
        noise_var = np.var(envelope[:noise_samples, :])
        
        # 3. 適応的ウィーナーフィルタ
        signal_var = np.var(envelope, axis=0, keepdims=True)
        wiener_filter = signal_var / (signal_var + noise_var + 1e-10)
        enhanced = envelope * wiener_filter
        
        # 4. 異方性拡散フィルタ（エッジ保存）
        enhanced = self._anisotropic_diffusion(enhanced, iterations=3, kappa=50, gamma=0.1)
        
        # 5. 局所S/N比計算
        local_snr = self._estimate_local_snr(enhanced)
        
        print(f"  信号強調完了: ノイズ分散={noise_var:.4f}")
        return enhanced, local_snr
    
    def _anisotropic_diffusion(self, image: np.ndarray, iterations: int = 3, 
                              kappa: float = 50, gamma: float = 0.1) -> np.ndarray:
        """異方性拡散フィルタ"""
        img = image.copy().astype(np.float64)
        
        for _ in range(iterations):
            # 勾配計算
            grad_n = np.roll(img, -1, axis=0) - img  # 北
            grad_s = np.roll(img, 1, axis=0) - img   # 南
            grad_e = np.roll(img, -1, axis=1) - img  # 東
            grad_w = np.roll(img, 1, axis=1) - img   # 西
            
            # 拡散係数計算
            c_n = np.exp(-(grad_n / kappa)**2)
            c_s = np.exp(-(grad_s / kappa)**2)
            c_e = np.exp(-(grad_e / kappa)**2)
            c_w = np.exp(-(grad_w / kappa)**2)
            
            # 更新
            img += gamma * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)
        
        return img
    
    def _estimate_local_snr(self, data: np.ndarray, window_size: int = 32) -> np.ndarray:
        """局所S/N比推定"""
        signal_power = uniform_filter(data**2, size=window_size)
        noise_power = uniform_filter(
            (data - uniform_filter(data, size=window_size))**2,
            size=window_size
        )
        return signal_power / (noise_power + 1e-10)
    
    def extract_hyperbola_candidates(self, enhanced_data: np.ndarray, 
                                   local_snr: np.ndarray,
                                   hough_peaks: List[Dict] = None) -> List[Dict]:
        """双曲線候補の抽出"""
        print("双曲線候補を抽出中...")
        
        candidates = []
        
        if hough_peaks:
            # Hough変換結果がある場合
            for peak in hough_peaks:
                t0, x0 = peak['t0_pix'], peak['x0_pix']
                candidate_points = self._extract_region_points(
                    enhanced_data, local_snr, t0, x0, radius=20
                )
                if len(candidate_points['t']) >= self.min_segment_points:
                    candidates.append(candidate_points)
        else:
            # 全域探索（セグメント化）
            candidates = self._segmented_candidate_extraction(enhanced_data, local_snr)
        
        print(f"  抽出された候補数: {len(candidates)}")
        return candidates
    
    def _extract_region_points(self, data: np.ndarray, local_snr: np.ndarray,
                              t0: int, x0: int, radius: int = 20) -> Dict:
        """指定領域からの点抽出"""
        height, width = data.shape
        
        # 領域設定
        t_min = max(0, t0 - radius)
        t_max = min(height, t0 + radius)
        x_min = max(0, x0 - radius)
        x_max = min(width, x0 + radius)
        
        region_data = data[t_min:t_max, x_min:x_max]
        region_snr = local_snr[t_min:t_max, x_min:x_max]
        
        # 高振幅点の抽出
        threshold = np.percentile(region_data, 90)
        snr_mask = region_snr > self.snr_threshold
        amp_mask = region_data > threshold
        
        valid_points = amp_mask & snr_mask
        t_coords, x_coords = np.where(valid_points)
        
        # 絶対座標に変換
        t_coords += t_min
        x_coords += x_min
        
        return {
            't': t_coords,
            'x': x_coords,
            'amplitudes': data[t_coords, x_coords],
            'snr_values': local_snr[t_coords, x_coords],
            'center': (t0, x0)
        }
    
    def _segmented_candidate_extraction(self, data: np.ndarray, 
                                      local_snr: np.ndarray) -> List[Dict]:
        """セグメント化による候補抽出"""
        width = data.shape[1]
        segment_width = width // 4
        overlap = segment_width // 3
        
        candidates = []
        
        for i in range(0, width - segment_width + 1, segment_width - overlap):
            end_idx = min(i + segment_width, width)
            
            segment_data = data[:, i:end_idx]
            segment_snr = local_snr[:, i:end_idx]
            
            # セグメント内での高振幅点抽出
            threshold = np.percentile(segment_data, 95)
            snr_mask = segment_snr > self.snr_threshold
            amp_mask = segment_data > threshold
            
            valid_points = amp_mask & snr_mask
            t_coords, x_coords = np.where(valid_points)
            
            if len(t_coords) >= self.min_segment_points:
                # 絶対座標に変換
                x_coords += i
                
                candidates.append({
                    't': t_coords,
                    'x': x_coords,
                    'amplitudes': data[t_coords, x_coords],
                    'snr_values': local_snr[t_coords, x_coords],
                    'segment_range': (i, end_idx)
                })
        
        return candidates
    
    def identify_continuous_segments(self, points_t: np.ndarray, 
                                   points_x: np.ndarray) -> List[Dict]:
        """連続セグメントの識別"""
        if len(points_t) < 3:
            return []
        
        # 空間的クラスタリング
        points = np.column_stack([points_t, points_x])
        
        # DBSCAN
        clustering = DBSCAN(eps=self.max_gap_samples, min_samples=3).fit(points)
        
        segments = []
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # ノイズ
                continue
                
            mask = clustering.labels_ == cluster_id
            if np.sum(mask) >= self.min_segment_points:
                segments.append({
                    't': points_t[mask],
                    'x': points_x[mask],
                    'size': np.sum(mask),
                    'cluster_id': cluster_id
                })
        
        # サイズ順ソート
        segments.sort(key=lambda s: s['size'], reverse=True)
        return segments
    
    def fit_with_gap_tolerance(self, candidate: Dict) -> Optional[DetectionResult]:
        """ギャップ許容フィッティング"""
        points_t = candidate['t']
        points_x = candidate['x']
        
        if len(points_t) < self.min_segment_points:
            return None
        
        # セグメント識別
        segments = self.identify_continuous_segments(points_t, points_x)
        
        if len(segments) == 0:
            return None
        
        best_result = None
        best_score = 0
        
        for primary_segment in segments:
            if primary_segment['size'] < self.min_segment_points:
                continue
            
            # RANSAC フィッティング
            result = self._ransac_hyperbola_fit(primary_segment)
            
            if result is None:
                continue
            
            # 全点での評価
            all_residuals = self._calculate_residuals(
                result.params_, points_t, points_x
            )
            
            # 品質評価
            quality = self._evaluate_fit_quality(
                result.params_, points_t, points_x, all_residuals, segments
            )
            
            if quality > best_score:
                best_score = quality
                best_result = DetectionResult(
                    params=result.params_,
                    quality_score=quality,
                    fit_residuals=all_residuals,
                    inlier_mask=np.abs(all_residuals) < np.std(all_residuals) * 2,
                    segments=segments,
                    has_gaps=len(segments) > 1,
                    confidence=quality
                )
        
        return best_result
    
    def _ransac_hyperbola_fit(self, segment: Dict, 
                             max_trials: int = 100) -> Optional[Any]:
        """RANSAC双曲線フィッティング"""
        points_t = segment['t']
        points_x = segment['x']
        n_points = len(points_t)
        
        if n_points < 6:
            return None
        
        best_fit = None
        best_inlier_ratio = 0
        
        # RANSACループ
        for _ in range(max_trials):
            # ランダムサンプリング
            sample_size = max(6, min(n_points // 2, 20))
            sample_indices = np.random.choice(n_points, sample_size, replace=False)
            
            sample_t = points_t[sample_indices]
            sample_x = points_x[sample_indices]
            
            try:
                # フィッティング
                regressor = HyperbolaRegressor()
                regressor.fit(sample_x.reshape(-1, 1), sample_t)
                
                # 全点での残差評価
                residuals = self._calculate_residuals(
                    regressor.params_, points_t, points_x
                )
                
                # インライア判定
                threshold = np.percentile(np.abs(residuals), 75) * 1.5
                inliers = np.abs(residuals) < threshold
                inlier_ratio = np.sum(inliers) / n_points
                
                if inlier_ratio > best_inlier_ratio:
                    best_inlier_ratio = inlier_ratio
                    best_fit = regressor
                    
            except Exception:
                continue
        
        return best_fit
    
    def _calculate_residuals(self, params: HyperbolaParams, 
                           points_t: np.ndarray, points_x: np.ndarray) -> np.ndarray:
        """残差計算"""
        t_pred = np.sqrt(params.t0**2 + (points_x - params.x0)**2 / params.v**2)
        return points_t - t_pred
    
    def _evaluate_fit_quality(self, params: HyperbolaParams,
                            points_t: np.ndarray, points_x: np.ndarray,
                            residuals: np.ndarray, segments: List[Dict]) -> float:
        """フィット品質評価"""
        
        # 1. フィッティング品質
        rmse = np.sqrt(np.mean(residuals**2))
        fitting_quality = 1.0 / (1.0 + rmse / 5.0)  # 正規化
        
        # 2. データ被覆率
        inlier_mask = np.abs(residuals) < np.std(residuals) * 2
        coverage_score = np.sum(inlier_mask) / len(points_t)
        
        # 3. 物理的妥当性
        physical_score = self._evaluate_physical_constraints(params)
        
        # 4. セグメント一貫性
        consistency_score = self._evaluate_segment_consistency(params, segments)
        
        # 重み付き総合スコア
        weights = [0.3, 0.3, 0.2, 0.2]
        scores = [fitting_quality, coverage_score, physical_score, consistency_score]
        
        return np.sum([w * s for w, s in zip(weights, scores)])
    
    def _evaluate_physical_constraints(self, params: HyperbolaParams) -> float:
        """物理制約評価"""
        score = 1.0
        
        # 時間制約
        time_ns = params.t0 * self.dt_ns
        if time_ns < 10 or time_ns > 500:
            score *= 0.1
        
        # 速度制約
        if params.v < 0.1 or params.v > 10:
            score *= 0.1
        
        # 位置制約（合理的範囲）
        if params.x0 < 0:
            score *= 0.5
        
        return score
    
    def _evaluate_segment_consistency(self, params: HyperbolaParams, 
                                    segments: List[Dict]) -> float:
        """セグメント一貫性評価"""
        if len(segments) <= 1:
            return 1.0
        
        consistencies = []
        for segment in segments:
            seg_residuals = self._calculate_residuals(
                params, segment['t'], segment['x']
            )
            seg_rmse = np.sqrt(np.mean(seg_residuals**2))
            consistency = 1.0 / (1.0 + seg_rmse / 5.0)
            consistencies.append(consistency)
        
        return np.mean(consistencies)
    
    def detect_hyperbolas(self, bscan_data: np.ndarray,
                         hough_peaks: List[Dict] = None,
                         max_hyperbolas: int = 10) -> List[DetectionResult]:
        """総合双曲線検出"""
        print(f"=== ロバスト双曲線検出開始 ===")
        
        # Phase 1: 信号強調
        enhanced_data, local_snr = self.robust_signal_enhancement(bscan_data)
        
        # Phase 2: 候補抽出
        candidates = self.extract_hyperbola_candidates(
            enhanced_data, local_snr, hough_peaks
        )
        
        # Phase 3: フィッティング
        results = []
        for i, candidate in enumerate(candidates):
            print(f"候補 {i+1}/{len(candidates)} を処理中...")
            result = self.fit_with_gap_tolerance(candidate)
            
            if result and result.quality_score > 0.5:
                results.append(result)
        
        # Phase 4: ランキング・フィルタリング
        results.sort(key=lambda r: r.quality_score, reverse=True)
        final_results = results[:max_hyperbolas]
        
        print(f"検出された信頼性の高い双曲線: {len(final_results)}")
        
        return final_results