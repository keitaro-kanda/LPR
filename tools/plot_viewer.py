import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm


# 軸範囲の設定
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]


def load_data(file_path):
    """データをテキストまたはCSVファイルから読み込む"""
    print("データを読み込んでいます...")
    data = np.loadtxt(file_path, delimiter=' ')
    print("データの読み込みが完了しました。")
    return data

def main():
    # B-scanデータの入力
    print('B-scanデータファイルのパスを入力してください:')
    bscan_path = input().strip()
    print('   ')

    # エコーピークデータの入力（オプション）
    print('エコーピークデータファイルのパスを入力してください（スキップする場合はEnterを押してください）:')
    peak_path = input().strip()
    print('   ')

    # データの読み込み
    data = load_data(bscan_path)
    if peak_path:
        peak_data = load_data(peak_path)

    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(show=True, title=f"B-scan Viewer: {bscan_path}")

    win.resize(2400, 800)

    cmap_Bscan_mpl = mpl_cm.get_cmap('bwr') # matplotlibのカラーマップオブジェクトを取得 # 修正
    lut = (cmap_Bscan_mpl(np.linspace(0, 1, 256)) * 255).astype(np.uint8) # NumPy配列形式のLUTを作成 # 修正
    cmap_Bscan = pg.ColorMap(pos=np.linspace(0, 1, 256), color=lut) # pyqtgraph.ColorMapオブジェクトを作成 # 修正

    cmap_peaks_mpl = mpl_cm.get_cmap('seismic') # matplotlibのカラーマップオブジェクトを取得 # 修正

    # レイアウトの列幅比率を設定（プロット領域を広くする）
    win.ci.layout.setColumnStretchFactor(0, 8)  # B-scanプロット領域の列を8に設定
    win.ci.layout.setColumnStretchFactor(1, 2)  # カラーバーの列を2に設定
    win.ci.layout.setColumnStretchFactor(2, 4)  # A-scanプロット領域の列を4に設定 # 追記

    # B-scanプロットアイテムの作成
    bscan_plot_item = pg.PlotItem() # 変数名変更
    bscan_plot_item.setLabel('bottom', 'x [m]', **{'font-size': '28pt'})
    bscan_plot_item.setLabel('left', 'Time [ns]', **{'font-size': '28pt'})
    bscan_plot_item.getAxis('bottom').setStyle(tickFont=QtGui.QFont('', 16))
    bscan_plot_item.getAxis('left').setStyle(tickFont=QtGui.QFont('', 16))
    bscan_plot_item.showGrid(x=True, y=True, alpha=0.5)

    # 軸の範囲を計算
    x_start = 0
    x_end = data.shape[1] * trace_interval
    y_start = 0
    y_end = data.shape[0] * sample_interval / 1e-9  # 単位をnsに変換

    # 軸範囲の設定
    bscan_plot_item.setXRange(x_start, x_end)
    bscan_plot_item.setYRange(y_start, y_end)

    # Y軸を反転
    bscan_plot_item.invertY(True)

    # イメージアイテムの作成
    img = pg.ImageItem()
    img.setImage(data.T)
    img.setLookupTable(lut) # LUTはNumPy配列のまま使用 # 修正
    img.setLevels([np.min(data), np.max(data)])

    # 画像の位置とスケールを設定
    img_rect = QtCore.QRectF(x_start, y_start, x_end - x_start, y_end - y_start)
    img.setRect(img_rect)

    # イメージアイテムをB-scanプロットアイテムに追加
    bscan_plot_item.addItem(img)

    # エコーピークデータの表示（存在する場合）
    if peak_path:
        # matplotlibのカラーマップ 'turbo' を使用
        cmap_peaks = mpl_cm.get_cmap('seismic')

        # 固定された範囲 -3000～3000 を用いて、各点の強度値を正規化します
        min_val = -3000
        max_val =  3000

        brushes = []
        for value in peak_data[:, 2]:
            norm_value = (value - min_val) / (max_val - min_val) if max_val - min_val != 0 else 0.5
            rgba = cmap_peaks(norm_value)
            color = QtGui.QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), int(rgba[3]*255))
            brushes.append(pg.mkBrush(color))

        scatter = pg.ScatterPlotItem(
            x=peak_data[:, 0],  # x座標
            y=peak_data[:, 1],  # 時間
            brush=brushes,      # valueに応じた色のリスト
            pen=None,
            symbol='o',
            size=5
        )
        bscan_plot_item.addItem(scatter)

    # A-scanプロットアイテムの作成 # 追記
    ascan_plot_item = pg.PlotItem()
    ascan_plot_item.setLabel('bottom', 'Amplitude', **{'font-size': '28pt'}) # A-scanは振幅 vs Time
    ascan_plot_item.setLabel('left', 'Time [ns]', **{'font-size': '28pt'})
    ascan_plot_item.getAxis('bottom').setStyle(tickFont=QtGui.QFont('', 16))
    ascan_plot_item.getAxis('left').setStyle(tickFont=QtGui.QFont('', 16))
    ascan_plot_item.showGrid(x=True, y=True, alpha=0.5)
    ascan_plot_item.setYRange(y_start, y_end) # B-scanとY軸範囲を合わせる
    ascan_plot_item.invertY(True) # B-scanとY軸方向を合わせる

    # A-scanプロットデータ初期化用Pen # 追記
    ascan_pen = pg.mkPen(color='b', width=2) # 青色の線

    # A-scanプロットアイテムにPlotCurveItemを追加 # 追記
    ascan_curve = pg.PlotCurveItem(pen=ascan_pen)
    ascan_plot_item.addItem(ascan_curve)

    # B-scanクリック時のイベントハンドラ # 追記
    def on_bscan_clicked(event):
        try: # エラーハンドリング開始
            if bscan_plot_item is None: # 念のため bscan_plot_item が None でないかチェック
                print("Error: bscan_plot_item is None in on_bscan_clicked")
                return
            if bscan_plot_item.scene() is None: # scene() が None でないかチェック
                print("Error: bscan_plot_item.scene() is None in on_bscan_clicked")
                return

            pos = event.pos()
            x_pos = pos.x()

            # x座標をトレース番号に変換
            trace_index = int(x_pos / trace_interval)

            # データ範囲外クリック対策
            if 0 <= trace_index < data.shape[1]:
                # A-scanデータの抽出
                ascan_data = data[:, trace_index]

                # A-scanプロットを更新
                time_axis = np.arange(len(ascan_data)) * sample_interval / 1e-9 # 時間軸[ns]
                ascan_curve.setData(x=ascan_data, y=time_axis) # 振幅 vs 時間

                # A-scanプロットのタイトルを更新 (選択したx座標を表示)
                ascan_plot_item.setTitle(f"A-scan at x = {x_pos:.2f} [m]")
            else:
                print("クリック位置がデータ範囲外です。")
        except Exception as e: # 例外発生時の処理
            print(f"Error in on_bscan_clicked: {e}") # エラー内容を詳細に表示


    # B-scanプロットアイテムにクリックイベントハンドラを登録 # 追記
    try: # エラーハンドリング開始
        if bscan_plot_item is None: # 念のため bscan_plot_item が None でないかチェック
            print(f"bscan_plot_item is None: {bscan_plot_item}") # ← 【追加】bscan_plot_item の値を出力
            print("Error: bscan_plot_item is None at registration")
        if bscan_plot_item.scene() is None: # scene() が None でないかチェック
            print("Error: bscan_plot_item.scene() is None at registration")
        else:
            bscan_plot_item.scene().sigMouseClicked.connect(on_bscan_clicked)
    except Exception as e: # 例外発生時の処理
        print(f"Error registering click handler: {e}") # エラー内容を詳細に表示


    # プロットアイテムとカラーバーの配置
    max_value = np.abs(data).max()
    colorbar = pg.ColorBarItem(values=(-max_value/5, max_value/5), colorMap=cmap_Bscan) # pyqtgraph.ColorMapオブジェクトを渡す # 修正
    colorbar.setImageItem(img)
    win.addItem(bscan_plot_item, row=0, col=0) # B-scanを配置
    win.addItem(colorbar, row=0, col=1)
    win.addItem(ascan_plot_item, row=0, col=2) # A-scanを配置 # 追記

    # ウィンドウの表示
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()