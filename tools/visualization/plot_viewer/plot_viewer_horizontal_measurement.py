import sys
import os
import json
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib as mpl
from scipy.signal import hilbert

# グループ定義（最大10グループ）
GROUP_DEFINITIONS = {
    "Group1": {"color": QtGui.QColor(255, 0, 0), "name": "Group1"},      # 赤
    "Group2": {"color": QtGui.QColor(0, 0, 255), "name": "Group2"},      # 青
    "Group3": {"color": QtGui.QColor(0, 255, 0), "name": "Group3"},      # 緑
    "Group4": {"color": QtGui.QColor(255, 255, 0), "name": "Group4"},    # 黄
    "Group5": {"color": QtGui.QColor(255, 0, 255), "name": "Group5"},    # マゼンタ
    "Group6": {"color": QtGui.QColor(0, 255, 255), "name": "Group6"},    # シアン
    "Group7": {"color": QtGui.QColor(255, 128, 0), "name": "Group7"},    # オレンジ
    "Group8": {"color": QtGui.QColor(128, 0, 255), "name": "Group8"},    # 紫
    "Group9": {"color": QtGui.QColor(0, 128, 128), "name": "Group9"},    # ティール
    "Group10": {"color": QtGui.QColor(128, 128, 0), "name": "Group10"},  # オリーブ
}

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
    水平距離計測用のライン。始点と終点を結ぶ直線と、中央のテキストラベルを表示
    """
    def __init__(self, start_pos, end_pos, measurement_id, distance_m, on_delete, on_group_change):
        super().__init__()
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.measurement_id = measurement_id
        self.distance_m = distance_m
        self.on_delete = on_delete
        self.on_group_change = on_group_change
        self.groups = set()  # 所属グループのセット

        # 直線を描画
        self.setLine(start_pos.x(), start_pos.y(), end_pos.x(), end_pos.y())
        self.setPen(QtGui.QPen(QtCore.Qt.red, 2))
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)

        # テキストラベル（距離を表示）
        self.text_item = QtWidgets.QGraphicsTextItem()
        self.text_item.setDefaultTextColor(QtCore.Qt.red)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.text_item.setFont(font)

        # テキストの位置を直線の中央に設定
        center_x = (start_pos.x() + end_pos.x()) / 2
        center_y = (start_pos.y() + end_pos.y()) / 2
        self.text_item.setPos(center_x, center_y)

        # ラベルを更新
        self.update_label()

    def update_label(self):
        """ラベルテキストを更新（距離とグループ情報）"""
        if self.groups:
            group_str = ",".join(sorted([g.replace("Group", "G") for g in self.groups]))
            label_text = f"{self.distance_m:.3f}m [{group_str}]"
        else:
            label_text = f"{self.distance_m:.3f}m"
        self.text_item.setPlainText(label_text)

    def update_appearance(self):
        """外観を更新（太さと色）"""
        if self.groups:
            # グループに所属している場合は太線
            self.setPen(QtGui.QPen(QtCore.Qt.red, 4))
        else:
            # グループに所属していない場合は通常線
            self.setPen(QtGui.QPen(QtCore.Qt.red, 2))

    def add_to_group(self, group_name):
        """グループに追加"""
        if group_name not in self.groups:
            self.groups.add(group_name)
            self.update_label()
            self.update_appearance()
            print(f"測定 #{self.measurement_id} を {group_name} に追加しました")
            self.on_group_change()

    def remove_from_group(self, group_name):
        """グループから削除"""
        if group_name in self.groups:
            self.groups.remove(group_name)
            self.update_label()
            self.update_appearance()
            print(f"測定 #{self.measurement_id} を {group_name} から削除しました")
            self.on_group_change()

    def mousePressEvent(self, ev):
        """右クリックでメニューを表示"""
        if ev.button() == QtCore.Qt.RightButton:
            menu = QtWidgets.QMenu()

            # グループに追加サブメニュー
            add_menu = menu.addMenu("グループに追加")
            for group_name in sorted(GROUP_DEFINITIONS.keys()):
                action = add_menu.addAction(group_name)
                action.setData(("add", group_name))
                # 既に所属している場合はチェックマークを付ける
                if group_name in self.groups:
                    action.setCheckable(True)
                    action.setChecked(True)

            # グループから削除サブメニュー（所属グループがある場合のみ）
            if self.groups:
                remove_menu = menu.addMenu("グループから削除")
                for group_name in sorted(self.groups):
                    action = remove_menu.addAction(group_name)
                    action.setData(("remove", group_name))

            menu.addSeparator()

            # 削除アクション
            deleteAct = menu.addAction("Delete Measurement")
            deleteAct.setData(("delete", None))

            action = menu.exec_(ev.screenPos())
            if action:
                action_type, group_name = action.data() if action.data() else (None, None)
                if action_type == "add":
                    if group_name not in self.groups:
                        self.add_to_group(group_name)
                elif action_type == "remove":
                    self.remove_from_group(group_name)
                elif action_type == "delete":
                    self.on_delete(self.measurement_id)
            ev.accept()
        else:
            super().mousePressEvent(ev)

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
    json_output = input("JSONファイルの出力パスを入力してください（空→同ディレクトリ _horizontal_measurements.json）:").strip()
    if not json_output:
        json_output = bscan_path.replace('.txt', '_horizontal_measurements.json')
    print()

    data = load_data(bscan_path)

    # NaN値対応: hilbert変換前にNaN値をゼロに置換
    data_for_hilbert = np.nan_to_num(data, nan=0.0)
    envelop = np.abs(hilbert(data_for_hilbert, axis=0))

    # 物理定数
    sample_interval = 0.312500e-9  # [s]
    trace_interval = 3.6e-2        # [m]

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
                # 古いフォーマット（groupsがない）の場合は空リストを設定
                for m in measurements:
                    if "groups" not in m:
                        m["groups"] = []
                    # 古いフォーマット（selected_for_sumがある）の場合は削除
                    if "selected_for_sum" in m:
                        del m["selected_for_sum"]
                if measurements:
                    measurement_counter = max(m["id"] for m in measurements)
                    print(f"既存の測定データを読み込みました: {len(measurements)}件")
        except Exception as e:
            print(f"JSONファイル読み込みエラー: {e}")

    # 測定状態の管理
    measuring_state = {"active": False, "first_pos": None}

    def calculate_group_statistics():
        """各グループの統計情報を計算"""
        group_stats = {}
        for group_name in GROUP_DEFINITIONS.keys():
            group_stats[group_name] = {"count": 0, "sum_distance_m": 0.0}

        # 各測定線のグループ情報を集計
        for m in measurements:
            for group_name in m.get("groups", []):
                if group_name in group_stats:
                    group_stats[group_name]["count"] += 1
                    group_stats[group_name]["sum_distance_m"] += m["distance_m"]

        return group_stats

    def update_group_display():
        """グループごとの合計距離を計算してコンソールに出力"""
        group_stats = calculate_group_statistics()

        print(f"\n{'='*50}")
        print(f"{'グループ統計':^48}")
        print(f"{'='*50}")
        print(f"{'グループ':<15} {'測定数':>8} {'合計距離 (m)':>15}")
        print(f"{'-'*50}")

        total_count = 0
        total_distance = 0.0
        for group_name in sorted(GROUP_DEFINITIONS.keys()):
            stats = group_stats[group_name]
            if stats["count"] > 0:
                print(f"{group_name:<15} {stats['count']:>8} {stats['sum_distance_m']:>15.3f}")
                total_count += stats["count"]
                total_distance += stats["sum_distance_m"]

        # グループに所属していない測定線の数
        ungrouped_count = sum(1 for m in measurements if not m.get("groups", []))

        print(f"{'-'*50}")
        print(f"{'グループなし':<15} {ungrouped_count:>8} {'-':>15}")
        print(f"{'='*50}")
        print(f"注: 1つの測定線が複数グループに所属している場合、重複してカウントされます")
        print(f"{'='*50}\n")

        # グループテーブルUIも更新
        update_group_table()

    def save_measurements():
        """測定データをJSONファイルに保存"""
        group_stats = calculate_group_statistics()

        data_to_save = {
            "groups": {
                group_name: {
                    "sum_distance_m": stats["sum_distance_m"],
                    "count": stats["count"]
                }
                for group_name, stats in group_stats.items()
                if stats["count"] > 0
            },
            "measurements": [
                {
                    "id": m["id"],
                    "x_start": m["x_start"],
                    "x_end": m["x_end"],
                    "y_position": m["y_position"],
                    "distance_m": m["distance_m"],
                    "groups": m.get("groups", [])
                }
                for m in measurements
            ]
        }
        with open(json_output, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"測定データを保存しました: {json_output}")

    def save_measurements_final():
        """アプリケーション終了時にx_startでソートしてIDを振り直して保存"""
        if not measurements:
            return

        # x_startで昇順ソート
        sorted_measurements = sorted(measurements, key=lambda m: m["x_start"])

        # IDを1から振り直し
        for i, measurement in enumerate(sorted_measurements, 1):
            measurement["id"] = i

        # measurements を更新
        measurements.clear()
        measurements.extend(sorted_measurements)

        # グループ統計を計算
        group_stats = calculate_group_statistics()

        # 保存
        data_to_save = {
            "groups": {
                group_name: {
                    "sum_distance_m": stats["sum_distance_m"],
                    "count": stats["count"]
                }
                for group_name, stats in group_stats.items()
                if stats["count"] > 0
            },
            "measurements": sorted_measurements
        }

        with open(json_output, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"測定データを最終保存しました（x_start昇順でID振り直し）: {json_output}")
        print(f"保存件数: {len(sorted_measurements)}件")

        # グループ統計を表示
        if any(stats["count"] > 0 for stats in group_stats.values()):
            print("\nグループ別合計距離:")
            for group_name in sorted(GROUP_DEFINITIONS.keys()):
                stats = group_stats[group_name]
                if stats["count"] > 0:
                    print(f"  {group_name}: {stats['sum_distance_m']:.3f} m ({stats['count']}件)")

    def on_delete_measurement(measurement_id):
        """測定を削除"""
        nonlocal measurements, measurement_items

        # データから削除
        measurements[:] = [m for m in measurements if m["id"] != measurement_id]

        # 画面から削除
        if measurement_id in measurement_items:
            line_item, text_item = measurement_items[measurement_id]
            plot.removeItem(line_item)
            plot.removeItem(text_item)
            del measurement_items[measurement_id]

        save_measurements()
        update_group_display()
        print(f"測定 #{measurement_id} を削除しました")

    def on_group_change():
        """測定線のグループ所属が変更された時の処理"""
        # measurement_items からグループ情報を measurements に同期
        for meas_id, (line_item, _) in measurement_items.items():
            for m in measurements:
                if m["id"] == meas_id:
                    m["groups"] = list(line_item.groups)
                    break

        save_measurements()
        update_group_display()

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

            # 同じy座標での計測を推奨するが、異なる場合は確認
            if abs(first_pos.y() - y) > sample_interval / 1e-9:
                reply = QtWidgets.QMessageBox.question(
                    None, "計測確認",
                    f"異なるy座標での計測です。\n開始点: t={first_pos.y():.1f}ns\n終了点: t={y:.1f}ns\n続行しますか？",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )
                if reply == QtWidgets.QMessageBox.No:
                    measuring_state["active"] = False
                    measuring_state["first_pos"] = None
                    return

            # 水平距離を計算
            distance_m = abs(x - first_pos.x())

            # 測定データを作成
            measurement_counter += 1
            measurement_data = {
                "id": measurement_counter,
                "x_start": min(first_pos.x(), x),
                "x_end": max(first_pos.x(), x),
                "y_position": (first_pos.y() + y) / 2,  # 中点のy座標
                "distance_m": distance_m,
                "groups": []  # デフォルトはグループなし
            }

            measurements.append(measurement_data)

            # 画面に描画
            line_item = MeasurementLine(
                first_pos, viewPos, measurement_counter, distance_m,
                on_delete_measurement, on_group_change
            )
            plot.addItem(line_item)
            plot.addItem(line_item.text_item)

            measurement_items[measurement_counter] = (line_item, line_item.text_item)

            # 測定状態をリセット
            measuring_state["active"] = False
            measuring_state["first_pos"] = None

            save_measurements()
            print(f"計測完了 #{measurement_counter}: 距離={distance_m:.3f}m")

    app = QtWidgets.QApplication(sys.argv)

    # メインウィンドウとレイアウトの作成
    main_widget = QtWidgets.QWidget()
    main_layout = QtWidgets.QVBoxLayout()
    main_widget.setLayout(main_layout)

    # グラフィックスウィジェット
    win = pg.GraphicsLayoutWidget(title=f"Horizontal Distance Measurement - {bscan_path}")

    # グループ一覧テーブル
    group_table = QtWidgets.QTableWidget()
    group_table.setColumnCount(3)
    group_table.setHorizontalHeaderLabels(["グループ", "測定数", "合計距離 (m)"])
    group_table.setMaximumHeight(250)
    group_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    group_table.horizontalHeader().setStretchLastSection(True)

    def update_group_table():
        """グループテーブルを更新"""
        group_stats = calculate_group_statistics()
        group_table.setRowCount(0)

        row = 0
        for group_name in sorted(GROUP_DEFINITIONS.keys()):
            stats = group_stats[group_name]
            if stats["count"] > 0:
                group_table.insertRow(row)
                group_table.setItem(row, 0, QtWidgets.QTableWidgetItem(group_name))
                group_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(stats["count"])))
                group_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{stats['sum_distance_m']:.3f}"))
                row += 1

    # レイアウトに追加
    main_layout.addWidget(win)
    main_layout.addWidget(group_table)

    # メインウィンドウの設定
    main_widget.setWindowTitle(f"Horizontal Distance Measurement - {os.path.basename(bscan_path)}")
    main_widget.resize(2400, 1000)
    main_widget.show()

    # アプリケーション終了時のイベントハンドラ
    def on_close_event(event):
        save_measurements_final()
        event.accept()

    # ウィンドウのクローズイベントをオーバーライド
    main_widget.closeEvent = on_close_event

    # カラーマップ
    cmap = mpl.colormaps.get_cmap('viridis')
    lut = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
    cpg = pg.ColorMap(np.linspace(0, 1, 256), lut)

    win.ci.layout.setColumnStretchFactor(0, 8)
    win.ci.layout.setColumnStretchFactor(1, 2)
    win.ci.layout.setColumnStretchFactor(2, 4)

    # B-scan PlotItem
    vb = CustomViewBox(on_click)
    plot = pg.PlotItem(viewBox=vb, title="B-scan (Click twice to measure horizontal distance)")
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
            start_pos = QtCore.QPointF(measurement["x_start"], measurement["y_position"])
            end_pos = QtCore.QPointF(measurement["x_end"], measurement["y_position"])

            # 測定ラインを作成
            line_item = MeasurementLine(
                start_pos, end_pos, measurement["id"],
                measurement["distance_m"], on_delete_measurement, on_group_change
            )

            # グループ情報を復元
            groups = measurement.get("groups", [])
            if groups:
                line_item.groups = set(groups)
                line_item.update_label()
                line_item.update_appearance()

            plot.addItem(line_item)
            plot.addItem(line_item.text_item)

            measurement_items[measurement["id"]] = (line_item, line_item.text_item)

            group_str = ", ".join(groups) if groups else "なし"
            print(f"測定 #{measurement['id']} を復元: x={measurement['x_start']:.2f}～{measurement['x_end']:.2f}m, "
                  f"距離={measurement['distance_m']:.3f}m (グループ: {group_str})")

    # 既存データを画面に復元
    if measurements:
        restore_measurements()
        update_group_display()

    print("\n=== 操作方法 ===")
    print("1. B-scan上で2回クリックして水平距離を計測")
    print("2. 測定線を右クリック → 「グループに追加」でグループ選択（複数選択可能）")
    print("3. 測定線を右クリック → 「グループから削除」でグループ解除")
    print("4. 測定線をダブルクリックで削除")
    print("5. グループごとの合計距離は画面下部のテーブルとコンソールに表示")
    print("6. 測定データは自動的にJSONファイルに保存されます")
    print(f"7. 保存先: {json_output}")
    print("================\n")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
