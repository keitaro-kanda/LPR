import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
import argparse
from pyqtgraph.Qt import QtCore
import matplotlib.pyplot as plt


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
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(
        description='Bスキャンプロットビューア（PyQtGraph版）',
        epilog='End of help message',
        usage='python plot_viewer.py [file_path]',
        prog='plot_viewer.py'
        )
    parser.add_argument('file_path', type=str, help='データファイルのパスを指定してください')
    args = parser.parse_args()

    # データの読み込み
    data = load_data(args.file_path)

    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(show=True, title=f"B-scan Viewer: {args.file_path}")

    win.resize(2400, 800)


    # カラーマップの設定
    cmap = pg.colormap.get('viridis')  # カラーマップ名を変更可能
    lut = cmap.getLookupTable(0.0, 1.0, 256)

    # レイアウトの列幅比率を設定（プロット領域を広くする）
    win.ci.layout.setColumnStretchFactor(0, 8)  # プロット領域の列を8に設定
    win.ci.layout.setColumnStretchFactor(1, 2)  # カラーバーの列を2に設定

    # プロットアイテムの作成
    plotItem = pg.PlotItem()
    plotItem.setLabel('bottom', 'x [m]', **{'font-size': '20pt'})
    plotItem.setLabel('left', 'Time [ns]', **{'font-size': '20pt'})
    plotItem.getAxis('bottom').setStyle(tickFont=pg.QtGui.QFont('', 16))
    plotItem.getAxis('left').setStyle(tickFont=pg.QtGui.QFont('', 16))
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

    # アスペクト比の設定
    #plotItem.getViewBox().setAspectLocked(lock=True, ratio=desired_ratio)

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

    # プロットアイテムとカラーバーの配置
    max_value = np.abs(data).max()
    win.addItem(plotItem, row=0, col=0)
    colorbar = pg.ColorBarItem(values=(-max_value/5, max_value/5), colorMap=cmap)
    colorbar.setImageItem(img)
    win.addItem(colorbar, row=0, col=1)

    # ウィンドウの表示
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
