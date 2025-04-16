import sys
import os
import json
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def load_data(file_path):
    """テキストまたはCSVファイルからデータを読み込み"""
    print("データを読み込んでいます...")
    data = np.loadtxt(file_path, delimiter=' ')
    print("データの読み込みが完了しました。")
    return data

def get_color_for_label(label):
    """ラベル番号に応じた色(R,G,B)を返す"""
    mapping = {
        1: (255, 0, 0),      # 赤
        2: (0, 255, 0),      # 緑
        3: (0, 0, 255),      # 青
        4: (255, 255, 0),    # 黄
        5: (255, 0, 255),    # マゼンタ
        6: (0, 255, 255)     # シアン
    }
    return mapping.get(label, (255, 255, 255))

class LabelMarker(QtWidgets.QGraphicsEllipseItem):
    """
    B-scan上に追加するマーカーアイテム。
    クリック位置を中心として、半径1の縁無しの円を描画し、
    ラベル番号に応じた色で塗りつぶします。
    ダブルクリックで編集・削除のコンテキストメニューが表示されます。
    """
    def __init__(self, x, y, label_value, key, on_edit, on_delete, radius=0.7, *args, **kwargs):
        # (x, y) を中心とするように位置補正
        super().__init__(x - radius, y - radius, 2 * radius, 2 * radius, *args, **kwargs)
        self.key = key
        self.label_value = label_value
        self.on_edit = on_edit
        self.on_delete = on_delete
        self.radius = radius
        r, g, b = get_color_for_label(label_value)
        self.setBrush(QtGui.QBrush(QtGui.QColor(r, g, b)))
        self.setPen(QtGui.QPen(QtCore.Qt.NoPen))  # 縁無し
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

    def mouseDoubleClickEvent(self, ev):
        menu = QtWidgets.QMenu()
        editAct = menu.addAction("Edit Label")
        deleteAct = menu.addAction("Delete Label")
        action = menu.exec_(ev.screenPos())
        if action == editAct:
            try:
                current_value = int(self.label_value)
            except:
                current_value = 1
            new_value, ok = QtWidgets.QInputDialog.getInt(
                None, "Edit Label", "Enter new label (1-6):",
                current_value, 1, 6)
            if ok:
                self.label_value = new_value
                r, g, b = get_color_for_label(new_value)
                self.setBrush(QtGui.QBrush(QtGui.QColor(r, g, b)))
                self.on_edit(self.key, new_value)
        elif action == deleteAct:
            self.scene().removeItem(self)
            self.on_delete(self.key)
        ev.accept()

class CustomViewBox(pg.ViewBox):
    """
    カスタム ViewBox:
    マウス左クリックイベントを捕捉し、クリック位置（View座標系）を指定のコールバックに渡す。
    すでに LabelMarker 上でクリックされた場合は、新たなラベル追加を行わない。
    """
    def __init__(self, on_click_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_click_callback = on_click_callback

    def mouseClickEvent(self, ev):
        # シーン上のアイテムを取得（クリック位置をシーン座標から取得）
        scene_items = self.scene().items(ev.scenePos())
        for it in scene_items:
            if isinstance(it, LabelMarker):
                ev.ignore()
                return
        if ev.button() == QtCore.Qt.LeftButton:
            # シーン座標から View 座標への変換
            viewPos = self.mapSceneToView(ev.scenePos())
            self.on_click_callback(viewPos)
            ev.accept()
        else:
            ev.ignore()

def main():
    # 入力プロンプトでファイルパスを取得
    bscan_path = input("B-scanデータファイルのパスを入力してください:").strip()
    json_input = input("JSONファイルのパスを入力してください（空の場合はB-scanファイルと同じディレクトリのlabels.jsonを使用）:").strip()
    if not json_input:
        json_input = bscan_path.replace('.txt', '_labels.json')
    print('   ')
    
    # B-scanデータとエンベロープの読み込み
    data = load_data(bscan_path)
    envelop = np.abs(hilbert(data, axis=0))

    # パラメータ設定
    sample_interval = 0.312500e-9    # [s]
    trace_interval = 3.6e-2          # [m]
    x_start = 0
    x_end = data.shape[1] * trace_interval
    y_start = 0
    y_end = data.shape[0] * sample_interval / 1e-9  # ns換算

    # Qtアプリケーションとウィンドウ生成
    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(show=True, title=f"B-scan Viewer: {bscan_path}")
    win.resize(2400, 800)

    # Matplotlibカラーマップ（viridis）設定
    cmap_Bscan_mpl = mpl.colormaps.get_cmap('viridis')
    lut = (cmap_Bscan_mpl(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
    cmap_Bscan = pg.ColorMap(pos=np.linspace(0, 1, 256), color=lut)

    # レイアウト設定（B-scan領域、カラーバー、A-scan領域）
    win.ci.layout.setColumnStretchFactor(0, 8)
    win.ci.layout.setColumnStretchFactor(1, 2)
    win.ci.layout.setColumnStretchFactor(2, 4)

    # 既存のラベル情報の読み込み（存在すれば）
    labels_dict = {}    # キー: "1", "2", …, 値: [x, y, ラベル番号]
    if os.path.exists(json_input):
        try:
            with open(json_input, 'r') as f:
                json_data = json.load(f)
                if "results" in json_data:
                    labels_dict = json_data["results"]
        except Exception as e:
            print("JSON読み込みエラー:", e)

    def save_labels():
        with open(json_input, "w") as f:
            json.dump({"results": labels_dict}, f, indent=4)

    # LabelMarker オブジェクト管理用辞書
    label_items = {}

    # 再インデックス用関数：現在のラベル辞書および marker のキーを 1, 2, 3, ... に再設定する。
    def reindex_labels():
        new_labels_dict = {}
        new_label_items = {}
        # 現在のキーを数値順にソート
        sorted_keys = sorted(labels_dict.keys(), key=lambda k: int(k))
        for i, old_key in enumerate(sorted_keys, start=1):
            new_key = str(i)
            new_labels_dict[new_key] = labels_dict[old_key]
            if old_key in label_items:
                marker = label_items[old_key]
                marker.key = new_key
                new_label_items[new_key] = marker
        labels_dict.clear()
        labels_dict.update(new_labels_dict)
        label_items.clear()
        label_items.update(new_label_items)
        save_labels()

    def on_label_edit(key, new_label):
        if key in labels_dict:
            x, y, _ = labels_dict[key]
            labels_dict[key] = [x, y, new_label]
            save_labels()

    def on_label_delete(key):
        if key in labels_dict:
            del labels_dict[key]
            # 再インデックスしてキーを連番に更新
            reindex_labels()

    # B-scan領域でのクリック時にラベル（マーカー）を追加
    def on_viewbox_click(viewPos):
        x = viewPos.x()
        y = viewPos.y()
        # 座標範囲チェック
        if not (x_start <= x <= x_end and y_start <= y <= y_end):
            return
        label_value, ok = QtWidgets.QInputDialog.getInt(
            None, "Add Label", "Enter label (1-6):", 1, 1, 6)
        if ok:
            key = str(len(labels_dict) + 1)
            labels_dict[key] = [x, y, label_value]
            save_labels()
            marker = LabelMarker(x, y, label_value, key, on_label_edit, on_label_delete, radius=1)
            bscan_plot_item.addItem(marker)
            label_items[key] = marker

    # CustomViewBox の作成
    custom_vb = CustomViewBox(on_viewbox_click)

    # PlotItem生成時に custom_vb を viewBox として指定
    bscan_plot_item = pg.PlotItem(viewBox=custom_vb, title="B-scan")
    bscan_plot_item.setLabel('bottom', 'x [m]')
    bscan_plot_item.setLabel('left', 'Time [ns]')
    bscan_plot_item.showGrid(x=True, y=True, alpha=0.5)
    bscan_plot_item.setXRange(x_start, x_end)
    bscan_plot_item.setYRange(y_start, y_end)
    bscan_plot_item.invertY(True)

    # B-scan画像の ImageItem 作成と設定
    img = pg.ImageItem()
    img.setImage(data.T)
    img.setLookupTable(lut)
    img.setLevels([np.min(data), np.max(data)])
    img_rect = QtCore.QRectF(x_start, y_start, x_end - x_start, y_end - y_start)
    img.setRect(img_rect)
    bscan_plot_item.addItem(img)

    # JSONから読み込んだ既存ラベルの再描画
    for key, value in labels_dict.items():
        x, y, label_value = value
        marker = LabelMarker(x, y, label_value, key, on_label_edit, on_label_delete, radius=1)
        bscan_plot_item.addItem(marker)
        label_items[key] = marker

    # 以下、A-scan 表示用の設定（元コードの内容を踏襲）
    ascan_plot_item = pg.PlotItem()
    ascan_plot_item.setLabel('bottom', 'Amplitude')
    ascan_plot_item.setLabel('left', 'Time [ns]')
    ascan_plot_item.showGrid(x=True, y=True, alpha=0.5)
    ascan_plot_item.setYRange(y_start, y_end)
    ascan_plot_item.invertY(True)
    ascan_pen = pg.mkPen(color='w', width=2)
    env_pen = pg.mkPen(color='r', width=2)
    ascan_curve = pg.PlotCurveItem(pen=ascan_pen)
    ascan_plot_item.addItem(ascan_curve)
    env_curve = pg.PlotCurveItem(pen=env_pen)
    ascan_plot_item.addItem(env_curve)
    
    # 縦線（InfiniteLine）で A-scan 表示位置を指定
    v_line = pg.InfiniteLine(angle=90, movable=True, label='A-scan plot position')
    bscan_plot_item.addItem(v_line)
    initial_x_pos = x_end / 2.0
    v_line.setPos(initial_x_pos)
    
    def update_ascan():
        x_pos = v_line.value()
        trace_index = int(x_pos / trace_interval)
        if 0 <= trace_index < data.shape[1]:
            ascan_data = data[:, trace_index]
            env_data = envelop[:, trace_index]
            time_axis = np.arange(len(ascan_data)) * sample_interval / 1e-9
            ascan_curve.setData(x=ascan_data, y=time_axis)
            env_curve.setData(x=env_data, y=time_axis)
            ascan_plot_item.setTitle(f"A-scan at x = {x_pos:.2f} [m]")
            ascan_plot_item.setYRange(*bscan_plot_item.viewRange()[1])
        else:
            ascan_curve.clear()
            ascan_plot_item.setTitle("A-scan (データ範囲外)")
            ascan_plot_item.setYRange(y_start, y_end)
    v_line.sigPositionChanged.connect(update_ascan)
    update_ascan()

    # カラーバー設定（B-scan画像の右側）
    max_value = np.abs(data).max()
    colorbar = pg.ColorBarItem(values=(-max_value/8, max_value/8), colorMap=cmap_Bscan)
    colorbar.setImageItem(img)

    # プロットアイテムの配置
    win.addItem(bscan_plot_item, row=0, col=0)
    win.addItem(colorbar, row=0, col=1)
    win.addItem(ascan_plot_item, row=0, col=2)

    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
