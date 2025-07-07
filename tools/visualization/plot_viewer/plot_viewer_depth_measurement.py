import sys
import os
import json
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib as mpl
from scipy.signal import hilbert
import math

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

class MeasurementLine(QtWidgets.QGraphicsLineItem):
    """
    深さ計測用のライン。始点と終点を結ぶ直線と、中央のテキストラベルを表示
    """
    def __init__(self, start_pos, end_pos, measurement_id, time_diff, on_delete):
        super().__init__()
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.measurement_id = measurement_id
        self.time_diff = time_diff
        self.on_delete = on_delete
        
        # 直線を描画
        self.setLine(start_pos.x(), start_pos.y(), end_pos.x(), end_pos.y())
        self.setPen(QtGui.QPen(QtCore.Qt.red, 2))
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        
        # テキストラベル（時間差を表示）
        self.text_item = QtWidgets.QGraphicsTextItem(f"{time_diff:.1f}ns")
        self.text_item.setDefaultTextColor(QtCore.Qt.red)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.text_item.setFont(font)
        
        # テキストの位置を直線の中央に設定
        center_x = (start_pos.x() + end_pos.x()) / 2
        center_y = (start_pos.y() + end_pos.y()) / 2
        self.text_item.setPos(center_x, center_y)
    
    def mouseDoubleClickEvent(self, ev):
        """ダブルクリックで削除メニューを表示"""
        menu = QtWidgets.QMenu()
        deleteAct = menu.addAction("Delete Measurement")
        action = menu.exec_(ev.screenPos())
        if action == deleteAct:
            self.on_delete(self.measurement_id)
        ev.accept()

class CustomViewBox(pg.ViewBox):
    def __init__(self, on_click_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_click_callback = on_click_callback
        self.measuring = False
        self.first_click_pos = None

    def mouseClickEvent(self, ev):
        # 既存の測定ライン上のクリックは無視
        for item in self.scene().items(ev.scenePos()):
            if isinstance(item, MeasurementLine):
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
    json_output = input("JSONファイルの出力パスを入力してください（空→同ディレクトリ _measurements.json）:").strip()
    if not json_output:
        json_output = bscan_path.replace('.txt', '_measurements.json')
    print()

    data = load_data(bscan_path)
    
    # NaN値対応: hilbert変換前にNaN値をゼロに置換
    data_for_hilbert = np.nan_to_num(data, nan=0.0)
    envelop = np.abs(hilbert(data_for_hilbert, axis=0))

    # 物理定数
    sample_interval = 0.312500e-9  # [s]
    trace_interval = 3.6e-2        # [m]
    c = 299792458                  # [m/s]
    epsilon_r = 4.5                # 相対誘電率
    
    # t=0補正: x=0で最初にNaNではない値をとるインデックスを見つける
    time_zero_index = find_time_zero_index(data)
    
    # 時間軸の計算（t=0補正を適用）
    x_start, x_end = 0, data.shape[1] * trace_interval
    y_start = -time_zero_index * sample_interval / 1e-9  # 負の時間も含む
    y_end = (data.shape[0] - time_zero_index) * sample_interval / 1e-9
    
    print(f"時間軸範囲: {y_start:.2f} ns ～ {y_end:.2f} ns")
    print(f"空間軸範囲: {x_start:.2f} m ～ {x_end:.2f} m")

    # 測定データの管理
    measurements = []
    measurement_items = {}
    measurement_counter = 0
    
    # 既存のJSONファイルを読み込み
    if os.path.exists(json_output):
        try:
            with open(json_output, 'r') as f:
                existing_data = json.load(f)
                measurements = existing_data.get("measurements", [])
                if measurements:
                    measurement_counter = max(m["id"] for m in measurements)
                    print(f"既存の測定データを読み込みました: {len(measurements)}件")
        except Exception as e:
            print(f"JSONファイル読み込みエラー: {e}")
    
    # 測定状態の管理
    measuring_state = {"active": False, "first_pos": None}

    def save_measurements():
        """測定データをJSONファイルに保存"""
        data_to_save = {
            "measurements": [
                {
                    "id": m["id"],
                    "x_position": m["x_position"],
                    "start_time_ns": m["start_time_ns"],
                    "end_time_ns": m["end_time_ns"],
                    "time_diff_ns": m["time_diff_ns"],
                    "depth_m": m["depth_m"]
                }
                for m in measurements
            ]
        }
        with open(json_output, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"測定データを保存しました: {json_output}")

    def save_measurements_final():
        """アプリケーション終了時にx_positionでソートしてIDを振り直して保存"""
        if not measurements:
            return
            
        # x_positionで昇順ソート
        sorted_measurements = sorted(measurements, key=lambda m: m["x_position"])
        
        # IDを1から振り直し
        for i, measurement in enumerate(sorted_measurements, 1):
            measurement["id"] = i
        
        data_to_save = {
            "measurements": sorted_measurements
        }
        
        with open(json_output, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"測定データを最終保存しました（x_position昇順でID振り直し）: {json_output}")
        print(f"保存件数: {len(sorted_measurements)}件")

    def calculate_depth(time_diff_ns):
        """時間差から深さを計算"""
        time_diff_s = time_diff_ns * 1e-9
        depth_m = (time_diff_s * c) / (2 * math.sqrt(epsilon_r))
        return depth_m

    def on_delete_measurement(measurement_id):
        """測定を削除"""
        nonlocal measurements, measurement_items
        
        # データから削除
        measurements = [m for m in measurements if m["id"] != measurement_id]
        
        # 画面から削除
        if measurement_id in measurement_items:
            line_item, text_item = measurement_items[measurement_id]
            plot.removeItem(line_item)
            plot.removeItem(text_item)
            del measurement_items[measurement_id]
        
        save_measurements()
        print(f"測定 #{measurement_id} を削除しました")

    def on_click(viewPos):
        """クリック時の処理"""
        nonlocal measurement_counter
        
        x, y = viewPos.x(), viewPos.y()
        
        # 範囲外のクリックは無視
        if not (x_start <= x <= x_end and y_start <= y <= y_end):
            return
        
        if not measuring_state["active"]:
            # 1回目のクリック（始点）
            measuring_state["active"] = True
            measuring_state["first_pos"] = viewPos
            print(f"計測開始点: x={x:.2f}m, t={y:.1f}ns")
            
        else:
            # 2回目のクリック（終点）
            first_pos = measuring_state["first_pos"]
            
            # 同じx座標での計測のみ許可
            if abs(first_pos.x() - x) > trace_interval:
                reply = QtWidgets.QMessageBox.question(
                    None, "計測確認", 
                    f"異なるx座標での計測です。\n開始点: x={first_pos.x():.2f}m\n終了点: x={x:.2f}m\n続行しますか？",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )
                if reply == QtWidgets.QMessageBox.No:
                    measuring_state["active"] = False
                    measuring_state["first_pos"] = None
                    return
            
            # 時間差を計算
            time_diff_ns = abs(y - first_pos.y())
            depth_m = calculate_depth(time_diff_ns)
            
            # 測定データを作成
            measurement_counter += 1
            measurement_data = {
                "id": measurement_counter,
                "x_position": (first_pos.x() + x) / 2,  # 中点のx座標
                "start_time_ns": min(first_pos.y(), y),
                "end_time_ns": max(first_pos.y(), y),
                "time_diff_ns": time_diff_ns,
                "depth_m": depth_m
            }
            
            measurements.append(measurement_data)
            
            # 画面に描画
            line_item = MeasurementLine(
                first_pos, viewPos, measurement_counter, time_diff_ns, on_delete_measurement
            )
            plot.addItem(line_item)
            plot.addItem(line_item.text_item)
            
            measurement_items[measurement_counter] = (line_item, line_item.text_item)
            
            # 測定状態をリセット
            measuring_state["active"] = False
            measuring_state["first_pos"] = None
            
            save_measurements()
            print(f"計測完了 #{measurement_counter}: 時間差={time_diff_ns:.1f}ns, 深さ={depth_m:.3f}m")

    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(show=True, title=f"Depth Measurement - {bscan_path}")
    win.resize(2400, 800)
    
    # アプリケーション終了時のイベントハンドラ
    def on_close_event():
        save_measurements_final()
    
    # ウィンドウのクローズイベントをオーバーライド
    original_close_event = win.closeEvent
    def close_event_handler(event):
        on_close_event()
        if original_close_event:
            original_close_event(event)
        else:
            event.accept()
    win.closeEvent = close_event_handler

    # カラーマップ
    cmap = mpl.colormaps.get_cmap('viridis')
    lut = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
    cpg = pg.ColorMap(np.linspace(0, 1, 256), lut)

    win.ci.layout.setColumnStretchFactor(0, 8)
    win.ci.layout.setColumnStretchFactor(1, 2)
    win.ci.layout.setColumnStretchFactor(2, 4)

    # B-scan PlotItem
    vb = CustomViewBox(on_click)
    plot = pg.PlotItem(viewBox=vb, title="B-scan (Click twice to measure depth)")
    plot.setLabel('bottom', 'x [m]')
    plot.setLabel('left', 'Time [ns]')
    plot.showGrid(True, True)
    plot.setXRange(x_start, x_end)
    plot.setYRange(y_start, y_end)
    plot.invertY(True)

    # NaN値を0に置換してから表示用データを作成
    data_for_display = np.nan_to_num(data, nan=0.0)
    
    img = pg.ImageItem(data_for_display.T)
    img.setLookupTable(lut)
    img.setRect(QtCore.QRectF(x_start, y_start, x_end-x_start, y_end-y_start))
    plot.addItem(img)

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
    
    # 既存の測定データを画面に復元
    def restore_measurements():
        """既存の測定データを画面に復元"""
        for measurement in measurements:
            # 座標を復元
            start_pos = QtCore.QPointF(measurement["x_position"], measurement["start_time_ns"])
            end_pos = QtCore.QPointF(measurement["x_position"], measurement["end_time_ns"])
            
            # 測定ラインを作成
            line_item = MeasurementLine(
                start_pos, end_pos, measurement["id"], 
                measurement["time_diff_ns"], on_delete_measurement
            )
            plot.addItem(line_item)
            plot.addItem(line_item.text_item)
            
            measurement_items[measurement["id"]] = (line_item, line_item.text_item)
            
            print(f"測定 #{measurement['id']} を復元: x={measurement['x_position']:.2f}m, "
                  f"時間差={measurement['time_diff_ns']:.1f}ns, 深さ={measurement['depth_m']:.3f}m")
    
    # 既存データを画面に復元
    if measurements:
        restore_measurements()
    
    # 操作説明をウィンドウタイトルに表示
    win.setWindowTitle(f"Depth Measurement Tool - {os.path.basename(bscan_path)} | Click twice to measure, Double-click line to delete")
    
    win.show()
    
    print("\n=== 操作方法 ===")
    print("1. B-scan上で2回クリックして深さを計測")
    print("2. 測定線をダブルクリックで削除")
    print("3. 測定データは自動的にJSONファイルに保存されます")
    print(f"4. 保存先: {json_output}")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()