import sys
import os
import json
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Button

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

class SNComparisonToolQt(QtWidgets.QMainWindow):
    """Pure PyQt5 + matplotlib based SN Comparison Tool"""
    
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
        self.click_mode = False
        
        # Matplotlib objects
        self.figure = None
        self.canvas = None
        self.ax = None
        self.click_cid = None
        self.key_cid = None
        
        # UI setup
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle('SN Comparison Tool - Pure Qt5 Version')
        self.setGeometry(100, 100, 1400, 800)
        
        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Control panel
        self.create_control_panel(main_layout)
        
        # Matplotlib canvas placeholder
        self.canvas_widget = QtWidgets.QWidget()
        self.canvas_widget.setMinimumHeight(600)
        main_layout.addWidget(self.canvas_widget)
        
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
        self.load_btn.setMinimumHeight(40)
        control_layout.addWidget(self.load_btn)
        
        # Click mode toggle
        self.click_mode_btn = QtWidgets.QPushButton('Click Mode: OFF')
        self.click_mode_btn.clicked.connect(self.toggle_click_mode)
        self.click_mode_btn.setStyleSheet('background-color: red; color: white; font-weight: bold; font-size: 14px;')
        self.click_mode_btn.setMinimumHeight(40)
        self.click_mode_btn.setEnabled(False)
        control_layout.addWidget(self.click_mode_btn)
        
        # Group controls
        group_label = QtWidgets.QLabel('Current Group:')
        group_label.setStyleSheet('font-size: 14px; font-weight: bold;')
        control_layout.addWidget(group_label)
        
        self.group_spinbox = QtWidgets.QSpinBox()
        self.group_spinbox.setMinimum(1)
        self.group_spinbox.setMaximum(999)
        self.group_spinbox.setValue(1)
        self.group_spinbox.valueChanged.connect(self.change_group)
        self.group_spinbox.setMinimumHeight(40)
        self.group_spinbox.setStyleSheet('font-size: 14px;')
        control_layout.addWidget(self.group_spinbox)
        
        self.new_group_btn = QtWidgets.QPushButton('New Group')
        self.new_group_btn.clicked.connect(self.create_new_group)
        self.new_group_btn.setMinimumHeight(40)
        self.new_group_btn.setEnabled(False)
        control_layout.addWidget(self.new_group_btn)
        
        # Clear last point
        self.clear_btn = QtWidgets.QPushButton('Remove Last Point')
        self.clear_btn.clicked.connect(self.remove_last_point)
        self.clear_btn.setMinimumHeight(40)
        self.clear_btn.setEnabled(False)
        control_layout.addWidget(self.clear_btn)
        
        # Analysis controls
        self.save_btn = QtWidgets.QPushButton('Save & Analyze')
        self.save_btn.clicked.connect(self.save_and_analyze)
        self.save_btn.setMinimumHeight(40)
        self.save_btn.setStyleSheet('background-color: blue; color: white; font-weight: bold; font-size: 14px;')
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
                self.calc_background_data()
                self.setup_matplotlib_plot()
                
                # Enable controls
                self.click_mode_btn.setEnabled(True)
                self.new_group_btn.setEnabled(True)
                self.clear_btn.setEnabled(True)
                self.save_btn.setEnabled(True)
                
                self.status_bar.showMessage(f'Loaded: {os.path.basename(file_path)}')
                
                # Show instructions
                QtWidgets.QMessageBox.information(
                    self, 'Instructions', 
                    'File loaded successfully!\n\n'
                    '1. Click "Click Mode: OFF" to enable clicking\n'
                    '2. Click on the B-scan to add points\n'
                    '3. Use "New Group" to create different groups\n'
                    '4. Use "Save & Analyze" when finished'
                )
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to load file: {str(e)}')
    
    def load_bscan_data(self, file_path):
        """Load B-scan data with encoding handling"""
        try:
            # Try Shift-JIS first
            self.bscan_data = np.loadtxt(file_path, delimiter=' ', encoding='shift-jis')
            print(f'Loaded with Shift-JIS: {self.bscan_data.shape}')
        except UnicodeDecodeError:
            try:
                # Try UTF-16 as fallback
                self.bscan_data = np.loadtxt(file_path, delimiter=' ', encoding='utf-16')
                print(f'Loaded with UTF-16: {self.bscan_data.shape}')
            except Exception as e:
                raise Exception(f"Failed to load file with both encodings: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error loading file: {e}")
        
        self.data_path = file_path
        print(f'B-scan data loaded: {self.bscan_data.shape}')
        
    def calc_background_data(self):
        """Calculate background mean amplitude and standard deviation"""
        bscan_mean = np.mean(self.bscan_data, axis=1)
        self.background_data = bscan_mean
        self.background_std = np.std(self.bscan_data, axis=1)
        print(f'Background data calculated: {self.background_data.shape}')
        print(f'Background range: min={np.min(self.background_data):.6f}, max={np.max(self.background_data):.6f}')
        
    def setup_output_directory(self):
        """Setup output directory for saving results"""
        self.output_dir = os.path.join(os.path.dirname(self.data_path), 'SN_comparison')
        os.makedirs(self.output_dir, exist_ok=True)
        print(f'Output directory: {self.output_dir}')
        
    def setup_matplotlib_plot(self):
        """Setup matplotlib plot embedded in Qt"""
        # Clear previous canvas if exists
        if self.canvas is not None:
            self.canvas.setParent(None)
            
        # Create new figure and canvas
        self.figure = Figure(figsize=(16, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # Set up canvas layout
        canvas_layout = QtWidgets.QVBoxLayout(self.canvas_widget)
        # Clear existing layout
        for i in reversed(range(canvas_layout.count())): 
            canvas_layout.itemAt(i).widget().setParent(None)
        canvas_layout.addWidget(self.canvas)
        
        # Create subplot
        self.ax = self.figure.add_subplot(111)
        
        # Calculate extent
        extent = [0, self.bscan_data.shape[1] * trace_interval,
                 self.bscan_data.shape[0] * sample_interval * 1e9, 0]
        
        # Display B-scan
        vmax = 100
        im = self.ax.imshow(self.bscan_data, aspect='auto', cmap='seismic',
                           extent=extent, vmin=-vmax, vmax=vmax)
        
        # Set labels and formatting
        self.ax.set_xlabel('Moving distance [m]', fontsize=font_medium)
        self.ax.set_ylabel('Time [ns]', fontsize=font_medium)
        self.ax.tick_params(axis='both', which='major', labelsize=font_small)
        self.ax.set_title('SN Comparison Tool - Click Mode OFF (Click button to enable)', fontsize=font_large)
        
        # Add colorbar
        self.figure.subplots_adjust(bottom=0.1, right=0.9, top=0.9, left=0.1)
        cbar_ax = self.figure.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = self.figure.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=font_small)
        
        # Connect event handlers
        self.click_cid = self.canvas.mpl_connect('button_press_event', self.on_click)
        self.key_cid = self.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Force canvas to be focusable
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        
        self.canvas.draw()
        print('Matplotlib plot setup complete')
        print(f'Event handlers connected: click_cid={self.click_cid}, key_cid={self.key_cid}')
        
    def on_click(self, event):
        """Handle mouse click events"""
        print(f'\n=== CLICK EVENT DEBUG ===')
        print(f'Event: {event}')
        print(f'Button: {event.button}')
        print(f'Inaxes: {event.inaxes}')
        print(f'Self.ax: {self.ax}')
        print(f'Same axes: {event.inaxes is self.ax}')
        print(f'Click mode: {self.click_mode}')
        print(f'Coordinates: x={event.xdata}, y={event.ydata}')
        
        if event.inaxes != self.ax:
            print('Click outside main axis - ignoring')
            return
            
        if not self.click_mode:
            print('Click mode is OFF - ignoring click')
            # Show reminder
            self.status_bar.showMessage('Click mode is OFF! Enable it first.', 3000)
            return
            
        if event.button == 1:  # Left click
            if event.xdata is None or event.ydata is None:
                print('Invalid coordinates - ignoring')
                return
                
            x_coord = float(event.xdata)
            time_ns = float(event.ydata)
            
            print(f'Processing click: x={x_coord:.3f}m, t={time_ns:.3f}ns')
            
            # Convert to array indices
            x_idx = int(round(x_coord / trace_interval))
            t_idx = int(round(time_ns / (sample_interval * 1e9)))
            
            print(f'Array indices: x_idx={x_idx}, t_idx={t_idx}')
            print(f'Array bounds: x_max={self.bscan_data.shape[1]-1}, t_max={self.bscan_data.shape[0]-1}')
            
            # Boundary check
            if (0 <= x_idx < self.bscan_data.shape[1] and 
                0 <= t_idx < self.bscan_data.shape[0]):
                
                # Get amplitude
                amplitude = float(self.bscan_data[t_idx, x_idx])
                print(f'Amplitude at [{t_idx}, {x_idx}]: {amplitude}')
                
                # Create point data
                point_data = {
                    "x": x_coord,
                    "time_ns": time_ns,
                    "amplitude": amplitude
                }
                
                # Add to current group
                if self.current_group not in self.groups:
                    self.groups[self.current_group] = []
                    print(f'Created new group: {self.current_group}')
                
                self.groups[self.current_group].append(point_data)
                print(f'Point added to group {self.current_group}')
                print(f'Group contents: {self.groups[self.current_group]}')
                
                # Plot the point
                colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
                color = colors[(self.current_group - 1) % len(colors)]
                
                self.ax.scatter(x_coord, time_ns, c=color, s=50, marker='o', 
                               edgecolors='black', linewidth=1, zorder=10, 
                               label=f'Group {self.current_group}' if len(self.groups[self.current_group]) == 1 else "")
                
                # Update legend if new group
                if len(self.groups[self.current_group]) == 1:
                    self.ax.legend(loc='upper right', fontsize=font_small-2)
                
                self.canvas.draw()
                print(f'Point plotted with color: {color}')
                
                self.update_status()
                self.status_bar.showMessage(
                    f'Point added: x={x_coord:.2f}m, t={time_ns:.2f}ns, amp={amplitude:.3f}', 2000)
                
            else:
                print(f'Click outside data bounds!')
                self.status_bar.showMessage('Click outside data area!', 2000)
        else:
            print(f'Non-left click: button={event.button}')
            
    def on_key(self, event):
        """Handle keyboard events"""
        print(f'Key pressed: {event.key}')
        
        if event.key == 'c':
            self.toggle_click_mode()
        elif event.key == 'n':
            self.create_new_group()
        elif event.key == 'r':
            self.remove_last_point()
        elif event.key == 'e':
            self.save_and_analyze()
            
    def toggle_click_mode(self):
        """Toggle click mode on/off"""
        self.click_mode = not self.click_mode
        
        if self.click_mode:
            self.click_mode_btn.setText('Click Mode: ON')
            self.click_mode_btn.setStyleSheet('background-color: green; color: white; font-weight: bold; font-size: 14px;')
            if self.ax:
                self.ax.set_title('SN Comparison Tool - Click Mode ON (Click to add points)', fontsize=font_large)
        else:
            self.click_mode_btn.setText('Click Mode: OFF')
            self.click_mode_btn.setStyleSheet('background-color: red; color: white; font-weight: bold; font-size: 14px;')
            if self.ax:
                self.ax.set_title('SN Comparison Tool - Click Mode OFF (Click button to enable)', fontsize=font_large)
        
        if self.canvas:
            self.canvas.draw()
        
        self.update_status()
        print(f'Click mode toggled: {"ON" if self.click_mode else "OFF"}')
        
    def change_group(self, value):
        """Change current group"""
        self.current_group = value
        self.update_status()
        print(f'Current group changed to: {self.current_group}')
        
    def create_new_group(self):
        """Create new group and switch to it"""
        max_group = max(self.groups.keys()) if self.groups else 0
        new_group = max_group + 1
        self.group_spinbox.setValue(new_group)
        self.status_bar.showMessage(f'Created new group: {new_group}', 2000)
        print(f'Created new group: {new_group}')
        
    def remove_last_point(self):
        """Remove last point from current group"""
        if (self.current_group in self.groups and 
            self.groups[self.current_group]):
            
            removed_point = self.groups[self.current_group].pop()
            print(f'Removed point: {removed_point}')
            
            # Redraw plot to remove the point
            self.redraw_points()
            
            self.update_status()
            self.status_bar.showMessage(
                f'Removed point from group {self.current_group}', 2000)
        else:
            self.status_bar.showMessage(
                f'No points to remove from group {self.current_group}', 2000)
            
    def redraw_points(self):
        """Redraw all points on the plot"""
        if not self.ax:
            return
            
        # Clear existing scatter plots
        for child in self.ax.get_children():
            if hasattr(child, 'get_offsets'):  # Scatter plot
                child.remove()
        
        # Redraw all points
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for group_num, points in self.groups.items():
            if points:
                color = colors[(group_num - 1) % len(colors)]
                x_coords = [point['x'] for point in points]
                y_coords = [point['time_ns'] for point in points]
                
                self.ax.scatter(x_coords, y_coords, c=color, s=50, marker='o',
                               edgecolors='black', linewidth=1, zorder=10,
                               label=f'Group {group_num}')
        
        # Update legend
        if self.groups:
            self.ax.legend(loc='upper right', fontsize=font_small-2)
        
        self.canvas.draw()
        print('Points redrawn')
        
    def update_status(self):
        """Update status bar"""
        total_points = sum(len(points) for points in self.groups.values())
        mode_text = "ON" if self.click_mode else "OFF"
        
        status_text = (f'Click Mode: {mode_text} | Current Group: {self.current_group} | '
                      f'Total Groups: {len(self.groups)} | Total Points: {total_points}')
        
        # Update window title as well
        self.setWindowTitle(f'SN Comparison Tool - {status_text}')
        
    def save_and_analyze(self):
        """Save data and generate analysis"""
        if not self.groups:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'No points to save!')
            return
            
        try:
            # Prepare data for saving
            json_data = {}
            total_points = 0
            
            for group_num, points in self.groups.items():
                if points:  # Only save non-empty groups
                    json_data[f"group_{group_num}"] = points
                    total_points += len(points)
            
            # Save JSON data
            json_path = os.path.join(self.output_dir, 'point_data.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f'Data saved to: {json_path}')
            print(f'Saved {len(json_data)} groups with {total_points} total points')
            
            # Generate analysis plots
            self.generate_analysis_plots(json_data)
            
            QtWidgets.QMessageBox.information(
                self, 'Success', 
                f'Analysis Complete!\n\n'
                f'Saved {len(json_data)} groups with {total_points} points\n'
                f'Results saved to: {self.output_dir}\n\n'
                f'Files created:\n'
                f'- point_data.json\n'
                f'- SN_comparison_plot_full.png/pdf\n'
                f'- SN_comparison_plot_0_200ns.png/pdf'
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to save/analyze: {str(e)}')
            print(f'Error in save_and_analyze: {e}')
            
    def generate_analysis_plots(self, json_data):
        """Generate background analysis and comparison plots"""
        print('Generating analysis plots...')
        
        # Calculate time array
        time_array = np.arange(0, len(self.background_data) * sample_interval / 1e-9, 
                              sample_interval / 1e-9)
        
        # Generate plots: full range and 0-200ns
        self.create_comparison_plot(time_array, self.background_data, self.background_std, 
                                   json_data, "full")
        self.create_comparison_plot(time_array, self.background_data, self.background_std, 
                                   json_data, "0-200ns")
        
    def create_comparison_plot(self, time_array, background_data, background_std, json_data, range_type="full"):
        """Create comparison plot using matplotlib"""
        print(f'Creating {range_type} comparison plot...')
        
        fig, ax = plt.subplots(figsize=(6, 10), tight_layout=True)
        
        # Convert to log scale, avoiding log(0)
        background_log = np.log(np.abs(background_data) + 1e-10)
        background_std_log = np.log(np.abs(background_std) + 1e-10)
        
        # Plot background data
        ax.plot(background_log, time_array, 'b-', linewidth=2, label='Background log amplitude')
        ax.fill_betweenx(time_array, background_log - background_std_log, 
                        background_log + background_std_log, color='b', alpha=0.6)
        
        # Plot points from all groups
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        print(f'Plotting data for {len(json_data)} groups')
        
        for i, (group_name, points) in enumerate(json_data.items()):
            if not points:
                continue
                
            color = colors[i % len(colors)]
            print(f'Processing {group_name}: {len(points)} points')
            
            # Extract data
            times = [point['time_ns'] for point in points]
            amplitudes = [np.log(abs(point['amplitude']) + 1e-10) for point in points]
            
            # Calculate statistics
            mean_time = np.mean(times)
            mean_amp = np.mean(amplitudes)
            std_amp = np.std(amplitudes) if len(amplitudes) > 1 else 0
            
            print(f'{group_name}: mean_time={mean_time:.2f}, mean_amp={mean_amp:.3f}, std_amp={std_amp:.3f}')
            
            # Plot individual points
            ax.scatter(amplitudes, times, c=color, s=30, alpha=0.7, 
                      label=f'{group_name} points')
            
            # Plot group mean with error bar
            ax.errorbar(mean_amp, mean_time, xerr=std_amp, 
                       fmt='o', color=color, markersize=8, 
                       capsize=5, capthick=2, linewidth=2,
                       label=f'{group_name} mean±std')
        
        # Set axis properties
        ax.set_xlabel('Log amplitude', fontsize=font_large)
        ax.set_ylabel('Time [ns]', fontsize=font_large)
        ax.tick_params(labelsize=font_medium)
        ax.grid(which='major', axis='both', linestyle='-.')
        ax.invert_yaxis()
        
        # Set time range
        if range_type == "0-200ns":
            ax.set_ylim(200, 0)
            ax.set_title('SN Comparison (0-200ns)', fontsize=font_large)
        else:
            ax.set_ylim(np.max(time_array), 0)
            ax.set_title('SN Comparison (Full Range)', fontsize=font_large)
        
        # Add legend
        ax.legend(fontsize=font_small, loc='lower right')
        
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
    app.setApplicationVersion('3.0 (Pure Qt5)')
    
    # Create and show main window
    window = SNComparisonToolQt()
    window.show()
    
    print('Application started')
    print('Instructions:')
    print('1. Click "Load B-scan File" to select data file')
    print('2. Click "Click Mode: OFF" to enable clicking')
    print('3. Click on B-scan to add points')
    print('4. Use "New Group" for different groups')
    print('5. Use "Save & Analyze" when finished')
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()