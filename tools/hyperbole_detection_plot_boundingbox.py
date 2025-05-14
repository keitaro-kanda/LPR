import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm



def load_bscan_data(file_path):
    """
    B-scanデータをtxtファイルから読み込み、指定範囲を切り出す。
    """
    print(f"B-scanデータを読み込んでいます: {file_path}")
    try:
        data = np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"エラー: ファイルの読み込みに失敗しました。 {e}")
        return None
    
    print(f"読み込み成功。データ形状: {data.shape} (時間サンプル数, トレース数)")

    return data


def load_bounding_boxes(file_path):
    """
    バウンディングボックスの情報をtxtファイルから読み込み、2次元配列に変換する。
    """
    print(f"バウンディングボックスデータを読み込んでいます: {file_path}")
    try:
        boxes = np.loadtxt(file_path, delimiter=' ')
    except Exception as e:
        print(f"エラー: ファイルの読み込みに失敗しました。 {e}")
        return None
    
    print(f"読み込み成功。バウンディングボックス形状: {boxes.shape} (数, 4)")

    return boxes


def plot(bscan_data, bounding_boxes, output_dir):
    """
    B-scanデータとバウンディングボックスをプロットする。
    """
    vmax = np.max(np.abs(bscan_data)) / 15
    fig, ax = plt.subplots(figsize=(18, 6))
    # imshow の戻り値（mappable）を受け取る
    im = ax.imshow(
        bscan_data,
        aspect='auto',
        cmap='viridis',
        extent=[0, bscan_data.shape[1], bscan_data.shape[0], 0],
        vmin=-vmax,
        vmax=vmax
    )

    # バウンディングボックス描画
    for i in tqdm(range(bounding_boxes.shape[0]), desc="バウンディングボックスの描画"):
        x_start, y_start, x_end, y_end = bounding_boxes[i, :]
        rect = plt.Rectangle(
            (x_start, y_start),
            x_end - x_start,
            y_end - y_start,
            linewidth=2,
            edgecolor='w',
            facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect)

    ax.set_xlabel("Trace Number", fontsize=20)
    ax.set_ylabel("Time Sample Number", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # カラーバー用の軸を作成
    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    # mappable（im）を指定してカラーバーを作成
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Amplitude', fontsize=20)
    cax.tick_params(labelsize=16)

    # ファイル名を指定して保存
    fig.savefig(output_dir + '/Bscan_with_boundingbox.png', bbox_inches='tight', dpi=300)
    fig.savefig(output_dir + '/Bscan_with_boundingbox.pdf', bbox_inches='tight', dpi=300)
    print(f"プロットを保存しました: {output_dir})")

    plt.show()




def main():
    #* パラメータ設定
    delta_x = 0.312500  # [ns]
    delta_t = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]

    #* B-scanデータの読み込み
    bscan_file_path = input("B-scanデータのtxtファイルパスを入力してください: ")
    if not os.path.isfile(bscan_file_path):
        print(f"エラー: ファイルが見つかりません: {bscan_file_path}")
        return
    bscan_data = load_bscan_data(bscan_file_path)
    if bscan_data is None:
        return
    
    #* バウンディングボックスの読み込み
    bounding_boxes_file_path = input("バウンディングボックスのtxtファイルパスを入力してください: ")
    if not os.path.isfile(bounding_boxes_file_path):
        print(f"エラー: ファイルが見つかりません: {bounding_boxes_file_path}")
        return
    bounding_boxes = load_bounding_boxes(bounding_boxes_file_path)
    if bounding_boxes is None:
        return
    if bounding_boxes.shape[1] != 4:
        print(f"エラー: バウンディングボックスの形式が不正です。4列必要ですが、{bounding_boxes.shape[1]}列あります。")
        return
    if bounding_boxes.shape[0] == 0:
        print("エラー: バウンディングボックスが見つかりません。")
        return
    if bounding_boxes.shape[0] > 0:
        print(f"バウンディングボックスの数: {bounding_boxes.shape[0]}")
    else:
        print("エラー: バウンディングボックスが見つかりません。")
        return
    
    #* バウンディングボックス情報をx, tに変換
    # bounding_boxes[:, 0] = bounding_boxes[:, 0] * delta_x
    # bounding_boxes[:, 1] = bounding_boxes[:, 1] * delta_t
    # bounding_boxes[:, 2] = bounding_boxes[:, 2] * delta_x
    # bounding_boxes[:, 3] = bounding_boxes[:, 3] * delta_t
    
    #* output_dirの作成
    output_dir = os.path.dirname(bounding_boxes_file_path)

    #* プロット
    plot(bscan_data, bounding_boxes, output_dir)
    print(f"プロットを保存しました: {output_dir}")
    print("プロットが完了しました。")

if __name__ == "__main__":
    main()