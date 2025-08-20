import sys
import os
import json
from datetime import datetime
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
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

class RangeSelector(QtWidgets.QRubberBand):
    """
    B-scanの範囲選択用のラバーバンド
    """
    def __init__(self, parent=None):
        super().__init__(QtWidgets.QRubberBand.Rectangle, parent)
        self.start_point = None
        self.end_point = None
        self.is_selecting = False

class CustomViewBox(pg.ViewBox):
    """
    範囲選択機能付きのカスタムViewBox
    """
    rangeSelected = QtCore.pyqtSignal(object, object)  # start_point, end_point
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rubber_band = None
        self.start_pos = None
        self.is_selecting = False
        # 標準のマウス操作を維持しつつ、範囲選択を優先
        self.setMouseEnabled(x=True, y=True)
        
    def mousePressEvent(self, ev):
        # Shift+左クリックで範囲選択開始（PyQtGraphとの競合を避ける）
        if ev.button() == QtCore.Qt.LeftButton and ev.modifiers() == QtCore.Qt.ShiftModifier:
            self.start_pos = ev.pos()
            self.is_selecting = True
            if self.rubber_band is None:
                self.rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
            self.rubber_band.setGeometry(QtCore.QRect(self.start_pos, QtCore.QSize()))
            self.rubber_band.show()
            # 標準のViewBox処理をスキップ
            ev.accept()
            return
        # その他の場合は標準処理
        super().mousePressEvent(ev)
            
    def mouseMoveEvent(self, ev):
        if self.is_selecting and self.rubber_band is not None:
            self.rubber_band.setGeometry(QtCore.QRect(self.start_pos, ev.pos()).normalized())
            ev.accept()
            return
        # その他の場合は標準処理
        super().mouseMoveEvent(ev)
            
    def mouseReleaseEvent(self, ev):
        if self.is_selecting and ev.button() == QtCore.Qt.LeftButton:
            end_pos = ev.pos()
            
            # ViewBox座標に変換
            start_scene = self.mapToScene(self.start_pos)
            end_scene = self.mapToScene(end_pos)
            start_view = self.mapSceneToView(start_scene)
            end_view = self.mapSceneToView(end_scene)
            
            # 範囲選択完了シグナルを発射
            self.rangeSelected.emit(start_view, end_view)
            
            # ラバーバンドを非表示
            if self.rubber_band is not None:
                self.rubber_band.hide()
            
            self.is_selecting = False
            ev.accept()
            return
        # その他の場合は標準処理
        super().mouseReleaseEvent(ev)

class BScanAScanViewer(QtWidgets.QMainWindow):
    """
    B-scanとA-scanを並べて表示し、matplotlib出力可能なビューア
    """
    def __init__(self, data_file_path):
        super().__init__()
        self.data = None
        self.envelop = None
        self.data_file_path = data_file_path
        self.sample_interval = 0.312500e-9
        self.trace_interval = 3.6e-2
        self.time_zero_index = 0
        self.x_start = 0
        self.x_end = 0
        self.y_start = 0
        self.y_end = 0
        self.current_ascan_pos = 0
        self.selected_x_range = None
        self.selected_y_range = None
        self.selection_rect = None
        
        self.initUI()
        self.load_data_file()
        
    def initUI(self):
        self.setWindowTitle('B-scan & A-scan Viewer')
        self.setGeometry(100, 100, 1400, 800)
        
        # 中央ウィジェット
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # レイアウト
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # コントロールパネル
        control_panel = QtWidgets.QHBoxLayout()
        
        # ファイル情報表示
        self.file_label = QtWidgets.QLabel(f'データファイル: {os.path.basename(self.data_file_path)}')
        control_panel.addWidget(self.file_label)
        
        # プロット生成ボタン
        self.plot_button = QtWidgets.QPushButton('プロット生成')
        self.plot_button.clicked.connect(self.generate_plot)
        self.plot_button.setEnabled(False)
        control_panel.addWidget(self.plot_button)
        
        # ステータスラベル
        self.status_label = QtWidgets.QLabel('データ読み込み中...')
        control_panel.addWidget(self.status_label)
        
        control_panel.addStretch()
        main_layout.addLayout(control_panel)
        
        # 説明ラベル
        info_label = QtWidgets.QLabel('操作方法: Shift+ドラッグでB-scan範囲選択、縦線でA-scan位置選択')
        main_layout.addWidget(info_label)
        
        # pyqtgraphウィジェット
        self.graphics_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.graphics_widget)
        
    def load_data_file(self):
        try:
            self.data = load_data(self.data_file_path)
            self.setup_display()
            self.status_label.setText(f'データ読み込み完了: {os.path.basename(self.data_file_path)}')
            self.plot_button.setEnabled(True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'エラー', f'データ読み込みエラー: {str(e)}')
            sys.exit(1)
                
    def setup_display(self):
        """
        データ表示の初期化
        """
        # NaN値対応: hilbert変換前にNaN値をゼロに置換
        data_for_hilbert = np.nan_to_num(self.data, nan=0.0)
        self.envelop = np.abs(hilbert(data_for_hilbert, axis=0))
        
        # t=0補正: x=0で最初にNaNではない値をとるインデックスを見つける
        self.time_zero_index = find_time_zero_index(self.data)
        
        # 時間軸の計算（t=0補正を適用）
        self.x_start, self.x_end = 0, self.data.shape[1] * self.trace_interval
        self.y_start = -self.time_zero_index * self.sample_interval / 1e-9  # 負の時間も含む
        self.y_end = (self.data.shape[0] - self.time_zero_index) * self.sample_interval / 1e-9
        
        print(f"時間軸範囲: {self.y_start:.2f} ns ～ {self.y_end:.2f} ns")
        print(f"空間軸範囲: {self.x_start:.2f} m ～ {self.x_end:.2f} m")
        
        # グラフィックスレイアウトをクリア
        self.graphics_widget.clear()
        
        # カラーマップ
        cmap = mpl.colormaps.get_cmap('viridis')
        lut = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
        cpg = pg.ColorMap(np.linspace(0, 1, 256), lut)
        
        # レイアウト設定
        self.graphics_widget.ci.layout.setColumnStretchFactor(0, 8)
        self.graphics_widget.ci.layout.setColumnStretchFactor(1, 2)
        self.graphics_widget.ci.layout.setColumnStretchFactor(2, 4)
        
        # B-scan PlotItem（カスタムViewBox使用）
        self.bscan_viewbox = CustomViewBox()
        self.bscan_viewbox.rangeSelected.connect(self.on_range_selected)
        self.bscan_plot = pg.PlotItem(viewBox=self.bscan_viewbox, title="B-scan")
        self.bscan_plot.setLabel('bottom', 'x [m]')
        self.bscan_plot.setLabel('left', 'Time [ns]')
        self.bscan_plot.showGrid(True, True)
        self.bscan_plot.setXRange(self.x_start, self.x_end)
        self.bscan_plot.setYRange(self.y_start, self.y_end)
        self.bscan_plot.invertY(True)
        
        # NaN値を0に置換してから表示用データを作成（描画問題の対処）
        data_for_display = np.nan_to_num(self.data, nan=0.0)
        
        self.bscan_img = pg.ImageItem(data_for_display.T)
        self.bscan_img.setLookupTable(lut)
        self.bscan_img.setRect(QtCore.QRectF(self.x_start, self.y_start, 
                                           self.x_end-self.x_start, self.y_end-self.y_start))
        self.bscan_plot.addItem(self.bscan_img)
        
        # A-scan 領域
        self.ascan_plot = pg.PlotItem()
        self.ascan_plot.setLabel('bottom', 'Amplitude')
        self.ascan_plot.setLabel('left', 'Time [ns]')
        self.ascan_plot.showGrid(True, True)
        self.ascan_plot.setYRange(self.y_start, self.y_end)
        self.ascan_plot.invertY(True)
        
        pen = pg.mkPen('w', width=2)
        pen2 = pg.mkPen('r', width=2)
        self.ascan_curve1 = pg.PlotCurveItem(pen=pen)
        self.ascan_curve2 = pg.PlotCurveItem(pen=pen2)
        self.ascan_plot.addItem(self.ascan_curve1)
        self.ascan_plot.addItem(self.ascan_curve2)
        
        # A-scan位置選択用の縦線
        self.vline = pg.InfiniteLine(angle=90, movable=True)
        self.bscan_plot.addItem(self.vline)
        self.vline.setPos(self.x_end/2)
        self.vline.sigPositionChanged.connect(self.update_ascan)
        
        # カラーバー
        data_max = np.nanmax(np.abs(self.data))
        if np.isnan(data_max) or data_max == 0:
            data_max = 1.0  # デフォルト値
        
        colorbar = pg.ColorBarItem(values=(-data_max/10, data_max/10), colorMap=cpg)
        colorbar.setImageItem(self.bscan_img)
        
        # レイアウトに追加
        self.graphics_widget.addItem(self.bscan_plot, 0, 0)
        self.graphics_widget.addItem(colorbar, 0, 1)
        self.graphics_widget.addItem(self.ascan_plot, 0, 2)
        
        # 初期A-scan更新
        self.update_ascan()
        
    def on_range_selected(self, start_point, end_point):
        """
        範囲選択時のコールバック
        """
        x1, y1 = start_point.x(), start_point.y()
        x2, y2 = end_point.x(), end_point.y()
        
        # 範囲を正規化
        self.selected_x_range = (min(x1, x2), max(x1, x2))
        self.selected_y_range = (min(y1, y2), max(y1, y2))
        
        # 既存の選択範囲を削除
        if self.selection_rect is not None:
            self.bscan_plot.removeItem(self.selection_rect)
        
        # 新しい選択範囲を表示
        self.selection_rect = pg.RectROI([self.selected_x_range[0], self.selected_y_range[0]], 
                                        [self.selected_x_range[1] - self.selected_x_range[0], 
                                         self.selected_y_range[1] - self.selected_y_range[0]], 
                                        pen='r', movable=False, removable=False)
        self.bscan_plot.addItem(self.selection_rect)
        
        print(f"選択範囲: x={self.selected_x_range[0]:.2f}-{self.selected_x_range[1]:.2f}m, "
              f"y={self.selected_y_range[0]:.2f}-{self.selected_y_range[1]:.2f}ns")
        
        self.status_label.setText(f'範囲選択完了 - プロット生成可能')
        
    def update_ascan(self):
        """
        A-scan表示の更新
        """
        if self.data is None:
            return
            
        xpos = self.vline.value()
        self.current_ascan_pos = xpos
        idx = int(xpos / self.trace_interval)
        
        if 0 <= idx < self.data.shape[1]:
            a = self.data[:, idx]
            e = self.envelop[:, idx]
            
            # t=0補正を適用した時間軸
            t = (np.arange(len(a)) - self.time_zero_index) * self.sample_interval / 1e-9
            
            # NaN値対応: NaN値がある場合の表示処理
            valid_mask = ~np.isnan(a)
            if np.any(valid_mask):
                # 有効な値のみをプロット
                self.ascan_curve1.setData(a[valid_mask], t[valid_mask])
                self.ascan_curve2.setData(e[valid_mask], t[valid_mask])
                
                # NaN値の範囲を表示
                nan_count = np.sum(~valid_mask)
                if nan_count > 0:
                    self.ascan_plot.setTitle(f"A-scan at x={xpos:.2f}m (NaN: {nan_count} points)")
                else:
                    self.ascan_plot.setTitle(f"A-scan at x={xpos:.2f}m")
            else:
                # 全てNaNの場合
                self.ascan_curve1.clear()
                self.ascan_curve2.clear()
                self.ascan_plot.setTitle(f"A-scan at x={xpos:.2f}m (All NaN)")
            
            self.ascan_plot.setYRange(*self.bscan_plot.viewRange()[1])
        else:
            self.ascan_curve1.clear()
            self.ascan_curve2.clear()
            self.ascan_plot.setTitle("out of range")
            self.ascan_plot.setYRange(self.y_start, self.y_end)
    
    def generate_plot(self):
        """
        matplotlib用のプロット生成
        """
        if self.data is None:
            QtWidgets.QMessageBox.warning(self, '警告', 'データが読み込まれていません')
            return
            
        # 表示範囲の決定
        if self.selected_x_range and self.selected_y_range:
            x_min, x_max = self.selected_x_range
            y_min, y_max = self.selected_y_range
        else:
            x_min, x_max = self.x_start, self.x_end
            y_min, y_max = self.y_start, self.y_end
            
        # 出力ディレクトリとファイル名の自動生成
        data_dir = os.path.dirname(self.data_file_path)
        output_dir = os.path.join(data_dir, 'Bscan_with_Ascan')
        os.makedirs(output_dir, exist_ok=True)
        
        # ファイル名をxmin_ymin_xmax_ymax形式で生成
        filename = f"{x_min:.2f}_{y_min:.1f}_{x_max:.2f}_{y_max:.1f}"
        output_path = os.path.join(output_dir, filename)
            
        try:
            self.create_matplotlib_plot(output_path, x_min, x_max, y_min, y_max)
            QtWidgets.QMessageBox.information(self, '完了', 
                f'プロットを保存しました:\n{output_path}.png\n{output_path}.pdf')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'エラー', f'プロット生成エラー: {str(e)}')
    
    def save_plot_record(self, output_path, x_min, x_max, y_min, y_max):
        """
        プロット情報をJSONファイルに記録
        """
        try:
            # JSONファイルパス
            data_dir = os.path.dirname(self.data_file_path)
            json_dir = os.path.join(data_dir, 'Bscan_with_Ascan')
            json_path = os.path.join(json_dir, 'plot_records.json')
            
            # 既存のJSONデータを読み込み
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        records = json.load(f)
                except (json.JSONDecodeError, IOError):
                    # ファイルが破損している場合は新規作成
                    records = {"data_file": self.data_file_path, "plots": []}
            else:
                # 新規作成
                records = {"data_file": self.data_file_path, "plots": []}
            
            # 新しいプロット記録を作成
            filename = os.path.basename(output_path)
            new_record = {
                "timestamp": datetime.now().isoformat(),
                "filename": filename,
                "bscan_range": {
                    "x_min": round(x_min, 2),
                    "x_max": round(x_max, 2),
                    "y_min": round(y_min, 1),
                    "y_max": round(y_max, 1)
                },
                "ascan_position": round(self.current_ascan_pos, 2),
                "output_files": [f"{filename}.png", f"{filename}.pdf"]
            }
            
            # 記録を追加
            records["plots"].append(new_record)
            
            # JSONファイルに保存
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            
            print(f"JSON記録保存完了: {json_path}")
            
        except Exception as e:
            print(f"JSON記録保存エラー: {str(e)}")
    
    def create_matplotlib_plot(self, output_path, x_min, x_max, y_min, y_max):
        """
        matplotlib用のプロット作成
        """
        # CLAUDE.md準拠のフォントサイズ
        font_large = 20
        font_medium = 18
        font_small = 16
        
        # 図のセットアップ
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # B-scan用データ準備（転置を除去）
        data_for_display = np.nan_to_num(self.data, nan=0.0)
        
        # B-scan表示（転置なし）
        extent = [self.x_start, self.x_end, self.y_end, self.y_start]
        vmax = np.nanmax(np.abs(self.data)) / 10
        ax1.imshow(data_for_display, aspect='auto', extent=extent, 
                  cmap='seismic', vmin=-vmax, vmax=vmax)
        
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_max, y_min)  # Y軸反転
        ax1.set_xlabel('x [m]', fontsize=font_medium)
        ax1.set_ylabel('Time [ns]', fontsize=font_medium)
        ax1.set_title('B-scan', fontsize=font_large)
        ax1.tick_params(labelsize=font_small)
        ax1.grid(True, alpha=0.3)
        
        # A-scan位置を示す縦線
        ax1.axvline(x=self.current_ascan_pos, color='white', linewidth=2, alpha=0.7, linestyle='--')

        # A-scan表示（横軸=強度、縦軸=時間）
        idx = int(self.current_ascan_pos / self.trace_interval)
        if 0 <= idx < self.data.shape[1]:
            a = self.data[:, idx]
            e = self.envelop[:, idx]
            t = (np.arange(len(a)) - self.time_zero_index) * self.sample_interval / 1e-9
            
            # NaN値対応
            valid_mask = ~np.isnan(a)
            if np.any(valid_mask):
                ax2.plot(a[valid_mask], t[valid_mask], 'b-', linewidth=2, label='Amplitude')
                ax2.plot(e[valid_mask], t[valid_mask], 'r-', linewidth=2, label='Envelope')
        
        ax2.set_ylim(y_max, y_min)  # Y軸反転（時間軸）
        ax2.set_xlabel('Amplitude', fontsize=font_medium)
        ax2.set_ylabel('Time [ns]', fontsize=font_medium)
        ax2.set_title(f'A-scan at x={self.current_ascan_pos:.2f}m', fontsize=font_large)
        ax2.tick_params(labelsize=font_small)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=font_small)
        
        # レイアウト調整
        plt.tight_layout()
        
        # CLAUDE.md準拠の保存
        plt.savefig(f'{output_path}.png', dpi=120)   # Web quality
        plt.savefig(f'{output_path}.pdf', dpi=600)   # Publication quality
        plt.close()
        
        # JSON記録への保存
        self.save_plot_record(output_path, x_min, x_max, y_min, y_max)

def main():
    # CLAUDE.md準拠の入力プロンプト
    bscan_path = input('B-scanデータファイルのパスを入力してください: ').strip()
    
    # ファイル存在確認
    if not os.path.exists(bscan_path):
        print('エラー: 指定されたファイルが存在しません')
        sys.exit(1)
    
    print()
    
    app = QtWidgets.QApplication(sys.argv)
    viewer = BScanAScanViewer(bscan_path)
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()