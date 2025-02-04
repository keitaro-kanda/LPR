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

    cmap_Bscan = mpl_cm.get_cmap('bwr')
    lut = (cmap_Bscan(np.linspace(0, 1, 256)) * 255).astype(np.uint8)

    # レイアウトの列幅比率を設定（プロット領域を広くする）
    win.ci.layout.setColumnStretchFactor(0, 8)  # プロット領域の列を8に設定
    win.ci.layout.setColumnStretchFactor(1, 2)  # カラーバーの列を2に設定

    # プロットアイテムの作成
    plotItem = pg.PlotItem()
    plotItem.setLabel('bottom', 'x [m]', **{'font-size': '28pt'})
    plotItem.setLabel('left', 'Time [ns]', **{'font-size': '28pt'})
    plotItem.getAxis('bottom').setStyle(tickFont=QtGui.QFont('', 16))
    plotItem.getAxis('left').setStyle(tickFont=QtGui.QFont('', 16))
    plotItem.showGrid(x=True, y=True, alpha=0.5)

    # 軸の範囲を計算
    x_start = 0
    x_end = data.shape[1] * trace_interval
    y_start = 0
    y_end = data.shape[0] * sample_interval / 1e-9  # 単位をnsに変換

    # 軸範囲の設定
    plotItem.setXRange(x_start, x_end)
    plotItem.setYRange(y_start, y_end)

    # Y軸を反転
    plotItem.invertY(True)

    # イメージアイテムの作成
    img = pg.ImageItem()
    img.setImage(data.T)
    img.setLookupTable(lut)
    img.setLevels([np.min(data), np.max(data)])

    # 画像の位置とスケールを設定
    img_rect = QtCore.QRectF(x_start, y_start, x_end - x_start, y_end - y_start)
    img.setRect(img_rect)

    # イメージアイテムをプロットアイテムに追加
    plotItem.addItem(img)

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
        plotItem.addItem(scatter)

    # プロットアイテムとカラーバーの配置
    max_value = np.abs(data).max()
    win.addItem(plotItem, row=0, col=0)
    colorbar = pg.ColorBarItem(values=(-max_value/5, max_value/5), colorMap=cmap_Bscan)
    colorbar.setImageItem(img)
    win.addItem(colorbar, row=0, col=1)

    # ウィンドウの表示
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
