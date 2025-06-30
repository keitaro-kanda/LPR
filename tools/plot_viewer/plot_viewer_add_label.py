import sys
import os
import json
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib as mpl
from scipy.signal import hilbert

def load_data(file_path):
    print("データを読み込んでいます...")
    data = np.loadtxt(file_path, delimiter=' ')
    print("データの読み込みが完了しました。")
    
    # NaN値の統計を表示
    nan_count = np.sum(np.isnan(data))
    total_count = data.size
    if nan_count > 0:
        print(f"NaN値検出: {nan_count} / {total_count} ({nan_count/total_count*100:.2f}%)")
    else:
        print("NaN値は検出されませんでした。")
    
    return data

def find_time_zero_index(data):
    """
    x=0（最初のトレース）において最初にNaNではない値をとるインデックスを見つける
    
    Parameters:
    -----------
    data : np.ndarray
        B-scanデータ (time x traces)
    
    Returns:
    --------
    time_zero_index : int
        t=0として設定するインデックス番号
    """
    if data.shape[1] == 0:
        return 0
    
    first_trace = data[:, 0]  # x=0のトレース
    
    # NaNではない最初のインデックスを探す
    valid_indices = np.where(~np.isnan(first_trace))[0]
    
    if len(valid_indices) == 0:
        print("警告: x=0のトレースにNaNではない値が見つかりません。t=0をインデックス0に設定します。")
        return 0
    
    time_zero_index = valid_indices[0]
    print(f"t=0として設定: インデックス {time_zero_index}")
    
    return time_zero_index

def get_color_for_label(label):
    mapping = {
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (255, 255, 0),
        5: (255, 0, 255),
        6: (0, 255, 255),
    }
    return mapping.get(label, (255, 255, 255))

class LabelMarker(QtWidgets.QGraphicsEllipseItem):
    """
    半径1の縁無し点を描画し、ラベル番号に応じた色で塗りつぶすマーカー。
    ダブルクリックで編集／削除メニューを表示し、time_top/bottom入力も可能。
    """
    def __init__(self, x, y, info_dict, key, on_edit, on_delete, radius=1, *args, **kwargs):
        super().__init__(x - radius, y - radius, 2*radius, 2*radius, *args, **kwargs)
        self.key = key
        self.info = info_dict    # {"x":..., "y":..., "label":..., "time_top":..., "time_bottom":...}
        self.on_edit = on_edit
        self.on_delete = on_delete
        r, g, b = get_color_for_label(self.info["label"])
        self.setBrush(QtGui.QBrush(QtGui.QColor(r, g, b)))
        self.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

    def mouseDoubleClickEvent(self, ev):
        menu = QtWidgets.QMenu()
        editAct = menu.addAction("Edit Label")
        deleteAct = menu.addAction("Delete Label")
        action = menu.exec_(ev.screenPos())
        if action == editAct:
            # 1) ラベル番号編集
            current_label = int(self.info.get("label", 1))
            new_label, ok_label = QtWidgets.QInputDialog.getInt(
                None, "Edit Label", "Enter new label (1–6):",
                current_label, 1, 6)
            if not ok_label:
                ev.accept()
                return
            self.info["label"] = new_label

            # 2) ラベルが 3 または 6 の場合、time_top/bottom を入力
            if new_label in (3, 6):
                # デフォルト値を数値保証
                default_top = self.info.get("time_top")
                default_top = float(default_top) if isinstance(default_top, (int, float)) else 0.0
                t_top, ok_top = QtWidgets.QInputDialog.getDouble(
                    None, "time_top [ns]", "Enter time_top (ns):",
                    default_top, 0.0, 1e6, 1)
                if not ok_top:
                    ev.accept()
                    return
                default_bot = self.info.get("time_bottom")
                default_bot = float(default_bot) if isinstance(default_bot, (int, float)) else 0.0
                t_bot, ok_bot = QtWidgets.QInputDialog.getDouble(
                    None, "time_bottom [ns]", "Enter time_bottom (ns):",
                    default_bot, 0.0, 1e6, 1)
                if not ok_bot:
                    ev.accept()
                    return
                self.info["time_top"] = round(t_top, 1)
                self.info["time_bottom"] = round(t_bot, 1)
            else:
                # 3/6 以外なら時刻情報をクリア
                self.info["time_top"] = None
                self.info["time_bottom"] = None

            # 3) 色の更新 & コールバック
            r, g, b = get_color_for_label(new_label)
            self.setBrush(QtGui.QBrush(QtGui.QColor(r, g, b)))
            self.on_edit(self.key, self.info)

        elif action == deleteAct:
            self.scene().removeItem(self)
            self.on_delete(self.key)

        ev.accept()

class CustomViewBox(pg.ViewBox):
    def __init__(self, on_click_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_click_callback = on_click_callback

    def mouseClickEvent(self, ev):
        # 既存マーカー上のクリックはスキップ
        for it in self.scene().items(ev.scenePos()):
            if isinstance(it, LabelMarker):
                ev.ignore()
                return
        if ev.button() == QtCore.Qt.LeftButton:
            viewPos = self.mapSceneToView(ev.scenePos())
            self.on_click_callback(viewPos)
            ev.accept()
        else:
            ev.ignore()

def main():
    bscan_path = input("B-scanデータファイルのパスを入力してください:").strip()
    json_input = input("JSONファイルのパスを入力してください（空→同ディレクトリ _labels.json）:").strip()
    if not json_input:
        json_input = bscan_path.replace('.txt', '_labels.json')
    print()

    data = load_data(bscan_path)
    
    # NaN値対応: hilbert変換前にNaN値をゼロに置換
    data_for_hilbert = np.nan_to_num(data, nan=0.0)
    envelop = np.abs(hilbert(data_for_hilbert, axis=0))

    sample_interval = 0.312500e-9
    trace_interval = 3.6e-2
    
    # t=0補正: x=0で最初にNaNではない値をとるインデックスを見つける
    time_zero_index = find_time_zero_index(data)
    
    # 時間軸の計算（t=0補正を適用）
    x_start, x_end = 0, data.shape[1] * trace_interval
    y_start = -time_zero_index * sample_interval / 1e-9  # 負の時間も含む
    y_end = (data.shape[0] - time_zero_index) * sample_interval / 1e-9
    
    print(f"時間軸範囲: {y_start:.2f} ns ～ {y_end:.2f} ns")
    print(f"空間軸範囲: {x_start:.2f} m ～ {x_end:.2f} m")

    # JSON 読み込み & マイグレーション
    labels_dict = {}
    if os.path.exists(json_input):
        try:
            jd = json.load(open(json_input, 'r'))
            for k, v in jd.get("results", {}).items():
                # 旧形式リスト→新形式 dict に変換
                if isinstance(v, list) and len(v) >= 3:
                    labels_dict[k] = {
                        "x": v[0], "y": v[1], "label": v[2],
                        "time_top": (v[3] if len(v) > 3 else None),
                        "time_bottom": (v[4] if len(v) > 4 else None)
                    }
                else:
                    labels_dict[k] = v
        except Exception as e:
            print("JSON読み込みエラー:", e)

    def save():
        with open(json_input, 'w') as f:
            json.dump({"results": labels_dict}, f, indent=4)

    # 再インデックス（1,2,3…）
    items = {}
    def reindex():
        new_ld, new_items = {}, {}
        for i, oldk in enumerate(sorted(labels_dict, key=lambda x: int(x)), start=1):
            newk = str(i)
            new_ld[newk] = labels_dict[oldk]
            if oldk in items:
                marker = items[oldk]
                marker.key = newk
                new_items[newk] = marker
        labels_dict.clear()
        labels_dict.update(new_ld)
        items.clear()
        items.update(new_items)
        save()

    def on_edit(key, info):
        labels_dict[key] = info
        save()

    def on_delete(key):
        if key in labels_dict:
            del labels_dict[key]
        reindex()

    # マーカー追加コールバック
    def on_click(viewPos):
        x, y = viewPos.x(), viewPos.y()
        if not (x_start <= x <= x_end and y_start <= y <= y_end):
            return
        lab, ok = QtWidgets.QInputDialog.getInt(None, "Add Label", "Enter label (1–6):", 1, 1, 6)
        if not ok:
            return
        tt = tb = None
        if lab in (3, 6):
            tt, ok1 = QtWidgets.QInputDialog.getDouble(None, "time_top [ns]", "Enter time_top:", 0.0, 0.0, 1e6, 1)
            if not ok1:
                return
            tb, ok2 = QtWidgets.QInputDialog.getDouble(None, "time_bottom [ns]", "Enter time_bottom:", 0.0, 0.0, 1e6, 1)
            if not ok2:
                return
            tt, tb = round(tt, 1), round(tb, 1)
        key = str(len(labels_dict) + 1)
        info = {"x": x, "y": y, "label": lab, "time_top": tt, "time_bottom": tb}
        labels_dict[key] = info
        save()
        m = LabelMarker(x, y, info, key, on_edit, on_delete, radius=1)
        plot.addItem(m)
        items[key] = m

    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(show=True, title=bscan_path)
    win.resize(2400, 800)

    # カラーマップ
    cmap = mpl.colormaps.get_cmap('viridis')
    lut = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
    cpg = pg.ColorMap(np.linspace(0, 1, 256), lut)

    win.ci.layout.setColumnStretchFactor(0, 8)
    win.ci.layout.setColumnStretchFactor(1, 2)
    win.ci.layout.setColumnStretchFactor(2, 4)

    # B-scan PlotItem
    vb = CustomViewBox(on_click)
    plot = pg.PlotItem(viewBox=vb, title="B-scan")
    plot.setLabel('bottom', 'x [m]')
    plot.setLabel('left', 'Time [ns]')
    plot.showGrid(True, True)
    plot.setXRange(x_start, x_end)
    plot.setYRange(y_start, y_end)
    plot.invertY(True)

    # NaN値を0に置換してから表示用データを作成（描画問題の対処）
    data_for_display = np.nan_to_num(data, nan=0.0)
    
    img = pg.ImageItem(data_for_display.T)
    img.setLookupTable(lut)
    img.setRect(QtCore.QRectF(x_start, y_start, x_end-x_start, y_end-y_start))
    plot.addItem(img)

    # 既存マーカー復元
    for k, info in labels_dict.items():
        m = LabelMarker(info["x"], info["y"], info, k, on_edit, on_delete, radius=1)
        plot.addItem(m)
        items[k] = m

    # A-scan 領域
    ascan = pg.PlotItem()
    ascan.setLabel('bottom', 'Amplitude')
    ascan.setLabel('left', 'Time [ns]')
    ascan.showGrid(True, True)
    ascan.setYRange(y_start, y_end)
    ascan.invertY(True)
    pen = pg.mkPen('w', width=2)
    pen2 = pg.mkPen('r', width=2)
    c1 = pg.PlotCurveItem(pen=pen)
    c2 = pg.PlotCurveItem(pen=pen2)
    ascan.addItem(c1)
    ascan.addItem(c2)
    vline = pg.InfiniteLine(angle=90, movable=True)
    plot.addItem(vline)
    vline.setPos(x_end/2)

    def update_ascan():
        xpos = vline.value()
        idx = int(xpos / trace_interval)
        if 0 <= idx < data.shape[1]:
            a = data[:, idx]
            e = envelop[:, idx]
            
            # t=0補正を適用した時間軸
            t = (np.arange(len(a)) - time_zero_index) * sample_interval / 1e-9
            
            # NaN値対応: NaN値がある場合の表示処理
            valid_mask = ~np.isnan(a)
            if np.any(valid_mask):
                # 有効な値のみをプロット
                c1.setData(a[valid_mask], t[valid_mask])
                c2.setData(e[valid_mask], t[valid_mask])
                
                # NaN値の範囲を表示
                nan_count = np.sum(~valid_mask)
                if nan_count > 0:
                    ascan.setTitle(f"A-scan at x={xpos:.2f}m (NaN: {nan_count} points)")
                else:
                    ascan.setTitle(f"A-scan at x={xpos:.2f}m")
            else:
                # 全てNaNの場合
                c1.clear()
                c2.clear()
                ascan.setTitle(f"A-scan at x={xpos:.2f}m (All NaN)")
            
            ascan.setYRange(*plot.viewRange()[1])
        else:
            c1.clear()
            c2.clear()
            ascan.setTitle("out of range")
            ascan.setYRange(y_start, y_end)

    vline.sigPositionChanged.connect(update_ascan)
    update_ascan()

    # NaN値を考慮したカラーバー範囲設定
    data_max = np.nanmax(np.abs(data))
    if np.isnan(data_max) or data_max == 0:
        data_max = 1.0  # デフォルト値
    
    colorbar = pg.ColorBarItem(values=(-data_max/10, data_max/10), colorMap=cpg)
    colorbar.setImageItem(img)

    win.addItem(plot, 0, 0)
    win.addItem(colorbar, 0, 1)
    win.addItem(ascan, 0, 2)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
