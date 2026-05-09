"""
Shihab and Al-Nuaimy (2005)で提案された、岩石サイズを考慮した双曲線方程式のデモンストレーション。
岩石深さ、岩石半径による双曲線形状への影響を調査する。
"""

import numpy as np
import matplotlib.pyplot as plt

# 物理量の定義
radii = np.arange(0.1, 0.6, 0.1)       # 岩石半径 [m]
depths = np.arange(0.5, 5.1, 0.5)     # 岩石深さ [m]
er = 3.0
c = 3e8  # 光速 [m/ns]
v = c / np.sqrt(er)  # 電磁波の速度 [m/s]
rx_positions = np.arange(-2.5, 2.6, 0.1)  # 受信機位置 [m]
t0 = depths * 2 / v  # 時間遅延 [s]

# 双曲線方程式の定義
def hyperbola(distance, r, t0):
    """_summary_
    Args:
        distance : distance between transmitter and receiver
        r : radius of the rock
        t0 (_type_): _description_
    """
    t = np.sqrt((t0 + 2 * r / v)**2 + (2*distance/v)**2) - 2 * r / v

    return t

# 計算
hyperbolas = np.zeros((len(radii), len(t0), len(rx_positions)))
for i, r in enumerate(radii):
    for j, d in enumerate(t0):
        for k, rx in enumerate(rx_positions):
            hyperbolas[i, j, k] = hyperbola(np.abs(rx), r, t0[j])

# プロット
colors = plt.cm.viridis(np.linspace(0, 1, len(radii)))
fig, ax = plt.subplots(figsize=(12, 8))
for i, r in enumerate(radii):
    for j, d in enumerate(t0):
        # j == 0 の時（各半径の最初の深さ）だけラベルを設定し、それ以外は None にする
        current_label = f'Radius: {r:.1f} m' if j == 0 else None
        ax.plot(rx_positions, hyperbolas[i, j, :] / 1e-9, color=colors[i], label=current_label)

ax.set_xlabel('Receiver Position (m)', fontsize=16)
ax.set_ylabel('Time Delay (ns)', fontsize=16)
ax.set_xlim(-2.6, 2.6) # [m]
ax.set_ylim(0, np.amax(hyperbolas) / 1e-9 * 1.05) # [ns]
ax.invert_yaxis()
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(True, which='both', linestyle='--', linewidth=0.7)

# 第2縦軸で深さに対応するラベルを追加
ax2 = ax.twinx()
ax2.set_ylabel(r'Depth (\varepsilon_r = 3.0)', fontsize=16)
ax2.set_ylim(0, np.amax(hyperbolas) * v / 2) # [m]
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.invert_yaxis()

ax.legend(fontsize=14)

output_dir = '/Volumes/SSD_Kanda_SAMSUNG/modeling_tools_output/demonstrate_radiu_variation'
plt.savefig(f'{output_dir}/hyperbola_radius_variation.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/hyperbola_radius_variation.pdf', dpi=300, bbox_inches='tight')
plt.show()