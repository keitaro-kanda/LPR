import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=np.RankWarning)

def detect_hyperbolic_points(data, neighborhood_radius, min_points, r2_threshold):
    """
    2次元スキャッターデータから双曲線状に連続する点を検出する。

    Args:
        data (np.array): スキャッターデータ (x, y, intensity) の配列。
        neighborhood_radius (float): 近傍点を探索する半径。
        min_points (int): 連続点として検出するために必要な最小点数。
        r2_threshold (float): 双曲線フィッティングのR^2値の閾値。

    Returns:
        list: 検出された双曲線状の点のインデックスのリスト。
    """
    x = data[:, 0]
    y = data[:, 1] * 1e-9 * 3e8 / 2 # [m]に変換

    kdtree = KDTree(data[:, :2])
    candidate_indices = []

    for i in tqdm(range(len(data)), desc='Detecting Hyperbolic Points'):
        query_point = data[i, :2]
        neighbor_indices = kdtree.query_ball_point(query_point, r=neighborhood_radius)
        if len(neighbor_indices) < 5: # 最低限の点数
            continue

        neighbor_points = data[neighbor_indices][:, :2]

        # 双曲線フィッティング (簡易版: 二次関数で近似)
        try:
            # y = ax^2 + bx + c フィッティング
            p_y_x, _ = np.polyfit(neighbor_points[:, 0], neighbor_points[:, 1], 2, cov=True)
            y_fit = np.polyval(p_y_x, neighbor_points[:, 0])
            r2_y_x = r2_score(neighbor_points[:, 1], y_fit)

            # x = ay^2 + by + c フィッティング
            p_x_y, _ = np.polyfit(neighbor_points[:, 1], neighbor_points[:, 0], 2, cov=True)
            x_fit = np.polyval(p_x_y, neighbor_points[:, 1])
            r2_x_y = r2_score(neighbor_points[:, 0], x_fit)

            if r2_y_x > r2_threshold or r2_x_y > r2_threshold: # どちらかのフィッティングが良い場合
                candidate_indices.append(i)
        except: # フィッティングが失敗する場合の例外処理
            continue

    # 連続点の連結 (簡易版: 近傍の候補点を連結)
    connected_indices_list = []
    visited_indices = set()

    for index in tqdm(candidate_indices, desc='Connecting Hyperbolic Points'):
        if index in visited_indices:
            continue
        current_connected_indices = [index]
        visited_indices.add(index)
        
        query_point = data[index, :2]
        neighbor_candidate_indices = kdtree.query_ball_point(query_point, r=neighborhood_radius)
        
        queue = neighbor_candidate_indices
        while queue:
            neighbor_index = queue.pop(0)
            if neighbor_index in visited_indices or neighbor_index not in candidate_indices:
                continue
            if neighbor_index in candidate_indices:
                current_connected_indices.append(neighbor_index)
                visited_indices.add(neighbor_index)
                neighbor_point = data[neighbor_index, :2]
                next_neighbors = kdtree.query_ball_point(neighbor_point, r=neighborhood_radius)
                for next_neighbor_index in next_neighbors:
                    if next_neighbor_index not in visited_indices and next_neighbor_index not in queue:
                        queue.append(next_neighbor_index)
                        
        if len(current_connected_indices) >= min_points:
            connected_indices_list.append(current_connected_indices)

    # 重複を削除してインデックスをリストにまとめる
    detected_indices = []
    for indices in tqdm(connected_indices_list, desc='Merging Hyperbolic Points'):
        detected_indices.extend(indices)
    detected_indices = sorted(list(set(detected_indices))) # 重複削除とソート

    return detected_indices

if __name__ == '__main__':
    # ダミーデータの生成 (双曲線形状 + ランダムノイズ)
    #np.random.seed(0)
    #n_points = 300000
    #x_hyperbola = np.linspace(-10, 10, 200)
    #y_hyperbola = np.sqrt(1 + x_hyperbola**2 / 4) * 2 # 双曲線 y^2/4 - x^2/4 = 1 の上半分
    #hyperbola_points = np.column_stack([x_hyperbola, y_hyperbola])

    path = "/Volumes/SSD_Kanda_BUFFALO/LPR/LPR_2B/Processed_Data/4_Gain_function/detect_peak/peak_x_t_values.txt"
    data = np.loadtxt(path, delimiter=' ')
    hyperbola_points = data[:, :2]
    num_hyperbola_points = len(hyperbola_points)
    #num_noise_points = n_points - num_hyperbola_points
    
    #noise_x = np.random.uniform(-15, 15, num_noise_points)
    #noise_y = np.random.uniform(-10, 15, num_noise_points)
    #noise_points = np.column_stack([noise_x, noise_y])
    
    #dummy_data_points = np.vstack([hyperbola_points + np.random.normal(0, 0.05, hyperbola_points.shape), noise_points])
    #dummy_intensity = np.random.rand(n_points) # 強度はランダム
    #dummy_data = np.column_stack([dummy_data_points, dummy_intensity])

    # 双曲線状の点群を検出
    #detected_indices = detect_hyperbolic_points(dummy_data, neighborhood_radius=0.5, min_points=10, r2_threshold=0.5)
    detected_indices = detect_hyperbolic_points(data, neighborhood_radius=0.5, min_points=10, r2_threshold=0.5)
    detected_data = data[detected_indices]
    print(f"検出された点の数: {len(detected_indices)}")


    # 検出された双曲線データ点を保存
    output_path = "/Volumes/SSD_Kanda_BUFFALO/LPR/LPR_2B/Processed_Data/4_Gain_function/detect_peak/detect_hyperbola"
    np.savetxt(
        output_path,
        detected_data,
        delimiter=' ',
        fmt='%.3f'
    )
    print(f"検出された双曲線データを {output_path} に保存しました")
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], s=1, label='All Data') # 全データ
    plt.scatter(detected_data[:, 0], detected_data[:, 1], color='red', s=5, label='Detected Hyperbolic Points') # 検出された点
    plt.title('Detection of Hyperbolic Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(np.max(data[:, 1]), np.min(data[:, 1])) # y軸を反転
    plt.legend()
    plt.grid(True)
    plt.show()

    