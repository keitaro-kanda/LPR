import sys
import os
import json
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

#* Parameter definitions (CLAUDE.md standards)
sample_interval = 0.312500e-9    # [s] - Time sampling interval
trace_interval = 3.6e-2          # [m] - Spatial trace interval
c = 299792458                    # [m/s] - Speed of light
epsilon_r = 4.5                  # Relative permittivity of lunar regolith
receiver_time_delay = 28.203e-9  # [s] - Hardware delay

#* Font size standards (CLAUDE.md)
font_large = 20      # Titles and labels
font_medium = 18     # Axis labels  
font_small = 16      # Tick labels

def get_color_for_group(group_num):
    """Get color for group number"""
    colors = [
        (255, 0, 0),    # Red - Group 1
        (0, 255, 0),    # Green - Group 2
        (0, 0, 255),    # Blue - Group 3
        (255, 165, 0),  # Orange - Group 4
        (128, 0, 128),  # Purple - Group 5
        (165, 42, 42),  # Brown - Group 6
        (255, 192, 203), # Pink - Group 7
        (128, 128, 128)  # Gray - Group 8
    ]
    return colors[(group_num - 1) % len(colors)]

class GroupMarker(QtWidgets.QGraphicsEllipseItem):
    """Marker for group points with editing capabilities"""
    
    def __init__(self, x, y, point_data, group_num, key, on_edit, on_delete, radius=1):
        super().__init__(x - radius, y - radius, 2*radius, 2*radius)
        self.key = key
        self.point_data = point_data
        self.group_num = group_num
        self.on_edit = on_edit
        self.on_delete = on_delete
        
        # Set color based on group
        r, g, b = get_color_for_group(group_num)
        self.setBrush(QtGui.QBrush(QtGui.QColor(r, g, b)))
        self.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        
    def mouseDoubleClickEvent(self, event):
        """Handle double click for editing"""
        menu = QtWidgets.QMenu()
        deleteAct = menu.addAction("Delete Point")
        action = menu.exec_(event.screenPos())
        
        if action == deleteAct:
            self.scene().removeItem(self)
            self.on_delete(self.key)
        
        event.accept()

class CustomViewBox(pg.ViewBox):
    """Custom ViewBox for handling clicks"""
    
    def __init__(self, on_click_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_click_callback = on_click_callback
        
    def mouseClickEvent(self, event):
        # Skip clicks on existing markers
        for item in self.scene().items(event.scenePos()):
            if isinstance(item, GroupMarker):
                event.ignore()
                return
                
        if event.button() == QtCore.Qt.LeftButton:
            viewPos = self.mapSceneToView(event.scenePos())
            self.on_click_callback(viewPos)
            event.accept()
        else:
            event.ignore()

class SNComparisonToolPyQt(QtWidgets.QMainWindow):
    """PyQt5-based SN Comparison Tool"""
    
    def __init__(self):
        super().__init__()
        self.bscan_data = None
        self.background_data = None
        self.background_std = None
        self.data_path = None
        self.output_dir = None
        
        # Group management
        self.current_group = 1
        self.groups = {}  # {group_num: [point_data1, point_data2, ...]}
        self.markers = {}  # {key: marker_object}
        self.click_mode = False
        
        # UI setup
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle('SN Comparison Tool - PyQt5 Version')
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Control panel
        self.create_control_panel(layout)
        
        # Graphics layout widget
        self.graphics_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_widget)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.update_status()
        
    def create_control_panel(self, layout):
        """Create control panel with buttons"""
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QHBoxLayout(control_panel)
        
        # File selection
        self.load_btn = QtWidgets.QPushButton('Load B-scan File')
        self.load_btn.clicked.connect(self.load_file)
        control_layout.addWidget(self.load_btn)
        
        # Click mode toggle
        self.click_mode_btn = QtWidgets.QPushButton('Click Mode: OFF')
        self.click_mode_btn.clicked.connect(self.toggle_click_mode)
        self.click_mode_btn.setStyleSheet('background-color: red; color: white; font-weight: bold;')
        control_layout.addWidget(self.click_mode_btn)
        
        # Group controls
        group_label = QtWidgets.QLabel('Current Group:')
        control_layout.addWidget(group_label)
        
        self.group_spinbox = QtWidgets.QSpinBox()
        self.group_spinbox.setMinimum(1)
        self.group_spinbox.setMaximum(999)
        self.group_spinbox.setValue(1)
        self.group_spinbox.valueChanged.connect(self.change_group)
        control_layout.addWidget(self.group_spinbox)
        
        self.new_group_btn = QtWidgets.QPushButton('New Group')
        self.new_group_btn.clicked.connect(self.create_new_group)
        control_layout.addWidget(self.new_group_btn)
        
        # Analysis controls
        self.save_btn = QtWidgets.QPushButton('Save & Analyze')
        self.save_btn.clicked.connect(self.save_and_analyze)
        self.save_btn.setEnabled(False)
        control_layout.addWidget(self.save_btn)
        
        control_layout.addStretch()
        layout.addWidget(control_panel)
        
    def load_file(self):
        """Load B-scan file"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select B-scan File', '', 'Text Files (*.txt);;All Files (*)')
        
        if file_path:
            try:
                self.load_bscan_data(file_path)
                self.setup_output_directory()
                self.setup_plot()
                self.auto_load_existing_data()  # 既存データの自動読み込み
                self.save_btn.setEnabled(True)
                self.status_bar.showMessage(f'Loaded: {os.path.basename(file_path)}')
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to load file: {str(e)}')
    
    def load_bscan_data(self, file_path):
        """Load B-scan data with encoding handling"""
        try:
            # Try Shift-JIS first
            self.bscan_data = np.loadtxt(file_path, delimiter=' ', encoding='shift-jis')
        except UnicodeDecodeError:
            try:
                # Try UTF-16 as fallback
                self.bscan_data = np.loadtxt(file_path, delimiter=' ', encoding='utf-16')
            except Exception as e:
                raise Exception(f"Failed to load file with both encodings: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error loading file: {e}")
        
        self.data_path = file_path
        print(f'B-scan data loaded: {self.bscan_data.shape}')
        
    def setup_output_directory(self):
        """Setup output directory for saving results"""
        self.output_dir = os.path.join(os.path.dirname(self.data_path), 'SN_comparison')
        os.makedirs(self.output_dir, exist_ok=True)
        print(f'Output directory: {self.output_dir}')
        
    def setup_plot(self):
        """Setup B-scan plot"""
        # Clear previous plot
        self.graphics_widget.clear()
        
        # Calculate extent
        x_start, x_end = 0, self.bscan_data.shape[1] * trace_interval
        y_start = 0
        y_end = self.bscan_data.shape[0] * sample_interval * 1e9
        
        # Custom viewbox for click handling
        self.viewbox = CustomViewBox(self.on_click)
        self.plot_item = pg.PlotItem(viewBox=self.viewbox, title="B-scan Data")
        self.plot_item.setLabel('bottom', 'Moving distance [m]')
        self.plot_item.setLabel('left', 'Time [ns]')
        self.plot_item.showGrid(True, True)
        self.plot_item.setXRange(x_start, x_end)
        self.plot_item.setYRange(y_start, y_end)
        self.plot_item.invertY(True)
        
        # Image item
        # Handle NaN values and prepare data for display
        data_for_display = np.nan_to_num(self.bscan_data, nan=0.0)
        self.image_item = pg.ImageItem(data_for_display.T)
        
        # Set colormap (seismic-like)
        colors = [
            (0, 0, 255),    # Blue
            (255, 255, 255), # White
            (255, 0, 0)     # Red
        ]
        cmap = pg.ColorMap(pos=[0.0, 0.5, 1.0], color=colors)
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        self.image_item.setLookupTable(lut)
        
        # Set image position and scale
        self.image_item.setRect(QtCore.QRectF(x_start, y_start, x_end-x_start, y_end-y_start))
        
        # Set color scale
        vmax = 100  # Fixed scale as in original
        self.image_item.setLevels([-vmax, vmax])
        
        self.plot_item.addItem(self.image_item)
        
        # Add to layout
        self.graphics_widget.addItem(self.plot_item, 0, 0)
        
        # Add colorbar
        colorbar = pg.ColorBarItem(values=(-vmax, vmax), colorMap=cmap)
        colorbar.setImageItem(self.image_item)
        self.graphics_widget.addItem(colorbar, 0, 1)
        
        print('Plot setup complete')
        
    def auto_load_existing_data(self):
        """B-scanファイル読み込み後にJSONファイルを自動探索して既存データを読み込む"""
        if not self.output_dir:
            return
            
        json_path = os.path.join(self.output_dir, 'point_data.json')
        
        if os.path.exists(json_path):
            try:
                self.load_existing_data(json_path)
                self.status_bar.showMessage(
                    f'Loaded: {os.path.basename(self.data_path)} + existing data ({len(self.groups)} groups)'
                )
                print(f'既存データを読み込みました: {json_path}')
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, 'Warning', 
                    f'既存データファイルの読み込みに失敗しました: {str(e)}'
                )
                print(f'既存データ読み込みエラー: {str(e)}')
        else:
            print(f'既存データファイルが見つかりません: {json_path}')
    
    def load_existing_data(self, json_path):
        """JSONファイルから既存データを読み込んでマーカーを表示"""
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # groupsデータの復元
        for group_key, points in json_data.items():
            if not group_key.startswith('group_'):
                continue
                
            group_num = int(group_key.split('_')[1])
            self.groups[group_num] = points
            
            # 各ポイントに対してマーカーを作成
            for i, point_data in enumerate(points):
                x = point_data['x']
                y = point_data['time_ns']
                
                # マーカーキーを生成
                key = f"group_{group_num}_point_{i+1}"
                
                # マーカーを作成してプロットに追加
                marker = GroupMarker(x, y, point_data, group_num, key, 
                                   self.on_edit_marker, self.on_delete_marker, radius=1)
                
                if hasattr(self, 'plot_item') and self.plot_item:
                    self.plot_item.addItem(marker)
                    self.markers[key] = marker
        
        # グループスピンボックスの最大値を更新
        if self.groups:
            max_group = max(self.groups.keys())
            self.group_spinbox.setMaximum(max(max_group, 999))
            
        # ステータスを更新
        self.update_status()
        
        print(f'読み込み完了: {len(self.groups)}グループ, 合計{sum(len(points) for points in self.groups.values())}ポイント')
        
    def toggle_click_mode(self):
        """Toggle click mode on/off"""
        self.click_mode = not self.click_mode
        
        if self.click_mode:
            self.click_mode_btn.setText('Click Mode: ON')
            self.click_mode_btn.setStyleSheet('background-color: green; color: white; font-weight: bold;')
        else:
            self.click_mode_btn.setText('Click Mode: OFF')
            self.click_mode_btn.setStyleSheet('background-color: red; color: white; font-weight: bold;')
        
        self.update_status()
        print(f'Click mode: {"ON" if self.click_mode else "OFF"}')
        
    def change_group(self, value):
        """Change current group"""
        self.current_group = value
        self.update_status()
        print(f'Current group: {self.current_group}')
        
    def create_new_group(self):
        """Create new group and switch to it"""
        max_group = max(self.groups.keys()) if self.groups else 0
        new_group = max_group + 1
        self.group_spinbox.setValue(new_group)
        print(f'Created new group: {new_group}')
        
    def on_click(self, viewPos):
        """Handle click on plot"""
        if not self.click_mode:
            print('Click mode is OFF - ignoring click')
            return
            
        x, y = viewPos.x(), viewPos.y()
        print(f'Click detected: x={x:.3f}m, y={y:.3f}ns')
        
        # Convert to array indices
        x_idx = int(round(x / trace_interval))
        y_idx = int(round(y / (sample_interval * 1e9)))
        
        # Boundary check
        if (0 <= x_idx < self.bscan_data.shape[1] and 
            0 <= y_idx < self.bscan_data.shape[0]):
            
            # Get amplitude
            amplitude = self.bscan_data[y_idx, x_idx]
            
            # Create point data
            point_data = {
                "x": float(x),
                "time_ns": float(y),
                "amplitude": float(amplitude)
            }
            
            # Add to group
            if self.current_group not in self.groups:
                self.groups[self.current_group] = []
            
            self.groups[self.current_group].append(point_data)
            
            # Create marker
            key = f"group_{self.current_group}_point_{len(self.groups[self.current_group])}"
            marker = GroupMarker(x, y, point_data, self.current_group, key, 
                               self.on_edit_marker, self.on_delete_marker, radius=0.1)
            self.plot_item.addItem(marker)
            self.markers[key] = marker
            
            print(f'Point added to group {self.current_group}: x={x:.2f}m, t={y:.2f}ns, amp={amplitude:.3f}')
            print(f'Group {self.current_group} now has {len(self.groups[self.current_group])} points')
            
            self.update_status()
        else:
            print(f'Click outside bounds: x_idx={x_idx}, y_idx={y_idx}')
            
    def on_edit_marker(self, key, point_data):
        """Handle marker edit"""
        # This could be extended for editing point properties
        pass
        
    def on_delete_marker(self, key):
        """Handle marker deletion"""
        if key in self.markers:
            # Find and remove from groups
            for group_num, points in self.groups.items():
                # Find point with matching coordinates
                marker = self.markers[key]
                point_to_remove = None
                for point in points:
                    if (abs(point['x'] - marker.point_data['x']) < 0.001 and
                        abs(point['time_ns'] - marker.point_data['time_ns']) < 0.001):
                        point_to_remove = point
                        break
                
                if point_to_remove:
                    points.remove(point_to_remove)
                    print(f'Removed point from group {group_num}')
                    break
            
            del self.markers[key]
            self.update_status()
            
    def update_status(self):
        """Update status bar"""
        total_points = sum(len(points) for points in self.groups.values())
        mode_text = "ON" if self.click_mode else "OFF"
        self.status_bar.showMessage(
            f'Click Mode: {mode_text} | Current Group: {self.current_group} | '
            f'Total Groups: {len(self.groups)} | Total Points: {total_points}'
        )
        
    def save_and_analyze(self):
        """Save data and generate analysis"""
        if not self.groups:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'No points to save!')
            return
            
        try:
            # Save JSON data
            json_data = {}
            for group_num, points in self.groups.items():
                json_data[f"group_{group_num}"] = points
                
            json_path = os.path.join(self.output_dir, 'point_data.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f'Data saved to: {json_path}')
            
            # Generate analysis plots
            self.generate_analysis_plots()
            
            QtWidgets.QMessageBox.information(
                self, 'Success', 
                f'Data saved and analysis complete!\nCheck: {self.output_dir}'
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to save/analyze: {str(e)}')
       
    def generate_analysis_plots(self):
        """Generate background analysis and comparison plots"""
        # Calculate time array from bscan_data
        time_array = np.arange(0, self.bscan_data.shape[0] * sample_interval / 1e-9, 
                              sample_interval / 1e-9)
        
        # Generate plots: full range and 0-200ns
        self.create_comparison_plot(time_array, self.bscan_data, "full")
        self.create_comparison_plot(time_array, self.bscan_data, "0-200ns")

    def create_comparison_plot(self, time_array, bscan_data, range_type="full"):
        """Create comparison plot using matplotlib"""
        fig, ax = plt.subplots(figsize=(6, 10), tight_layout=True)
        
        # Convert to log scale, avoiding log(0)
        bscan_log = np.log(np.abs(bscan_data))
        bscan_log_mean = np.mean(bscan_log, axis=1)
        bscan_log_std = np.std(bscan_log, axis=1)
        
        # Plot background data
        ax.plot(bscan_log_mean, time_array, 'b-', linewidth=2, label='Background log amplitude')
        ax.fill_betweenx(time_array, bscan_log_mean - bscan_log_std,
                         bscan_log_mean + bscan_log_std, color='b', alpha=0.6)
        
        # Plot points from all groups
        for i, (group_num, points) in enumerate(self.groups.items()):
            if not points:
                continue
                
            color = 'k'
            
            # Extract data
            times = [point['time_ns'] for point in points]
            amplitudes = [np.log(abs(point['amplitude']) + 1e-10) for point in points]
            
            # Calculate statistics
            mean_time = np.mean(times)
            mean_amp = np.mean(amplitudes)
            std_amp = np.std(amplitudes)
            
            # Calculate time-shifted amplitude
            shifted_amp_array = np.zeros_like(time_array)
            shifted_amp_array = 4 * np.log(mean_time / time_array + 1e-10) + mean_amp
            shifted_amp_std_array = 4 * np.log(mean_time / time_array + 1e-10) + std_amp

            # Plot group mean with error bar
            ax.errorbar(mean_amp, mean_time, xerr=std_amp, 
                       fmt='o', color=color, markersize=8, 
                       capsize=5, capthick=2, linewidth=2,
                       label=f'Group {group_num} mean±std')

            # Plot time-shifted amplitude
            ax.plot(shifted_amp_array, time_array, color='gray', linestyle='--')
            # ax.fill_betweenx(time_array, shifted_amp_array - shifted_amp_std_array,
            #                  shifted_amp_array + shifted_amp_std_array, color='gray', alpha=0.6)

        # Set axis properties
        ax.set_xlabel('Log amplitude', fontsize=font_large)
        ax.set_ylabel('Time [ns]', fontsize=font_large)
        ax.tick_params(labelsize=font_medium)
        ax.grid(which='major', axis='both', linestyle='-.')
        ax.invert_yaxis()
        
        # Set time range
        if range_type == "0-200ns":
            ax.set_xlim(-5, 10)
            ax.set_ylim(200, 0)
            ax.set_title('SN Comparison (0-200ns)', fontsize=font_large)
        else:
            ax.set_ylim(np.max(time_array), 0)
            ax.set_title('SN Comparison (Full Range)', fontsize=font_large)
        
        # Add legend
        # ax.legend(fontsize=font_small, loc='lower right')
        
        # Save plot
        filename_png = f'SN_comparison_plot_{range_type.replace("-", "_")}.png'
        filename_pdf = f'SN_comparison_plot_{range_type.replace("-", "_")}.pdf'
        
        plt.savefig(os.path.join(self.output_dir, filename_png), format='png', dpi=120)
        plt.savefig(os.path.join(self.output_dir, filename_pdf), format='pdf', dpi=600)
        plt.close()
        
        print(f'Comparison plot saved: {filename_png}')

def main():
    """Main function"""
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName('SN Comparison Tool')
    app.setApplicationVersion('2.0 (PyQt5)')
    
    # Create and show main window
    window = SNComparisonToolPyQt()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()