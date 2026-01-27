import numpy as np
import matplotlib.pyplot as plt
import os
import json

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

class SNComparisonTool:
    def __init__(self):
        self.bscan_data = None
        self.background_data = None
        self.background_std = None
        self.data_path = None
        self.current_group = 1
        self.groups = {}
        self.fig = None
        self.ax = None
        self.click_cid = None
        self.key_cid = None
        self.output_dir = None
        self.click_mode = False  # Click mode state
        self.mode_text = None    # Text object for mode display
        
    def load_bscan_data(self, file_path):
        """Load B-scan data with encoding handling"""
        try:
            # まずはShift-JISで読み込みを試行
            self.bscan_data = np.loadtxt(file_path, delimiter=' ', encoding='shift-jis')
            print(f'B-scanデータ読み込み完了: {self.bscan_data.shape}')
        except UnicodeDecodeError:
            # Shift-JISで失敗した場合、UTF-16で再試行
            try:
                self.bscan_data = np.loadtxt(file_path, delimiter=' ', encoding='utf-16')
                print(f'B-scanデータ読み込み完了: {self.bscan_data.shape}')
            except Exception as e:
                print(f"エラー: ファイル '{file_path}' の読み込みに失敗しました。")
                print(f"詳細: {e}")
                return False
        except Exception as e:
            print(f"予期せぬエラーがファイル '{file_path}' で発生しました。")
            print(f"詳細: {e}")
            return False
        
        self.data_path = file_path
        return True
    
    def calc_background_data(self):
        """Calculate background mean amplitude and its std"""
        bscan_mean = np.mean(self.bscan_data, axis=1)
        # Store original linear data
        self.background_data = bscan_mean
        self.background_std = np.std(self.bscan_data, axis=1)
        print(f'背景データ作成完了: {self.background_data.shape}')
        print(f'背景データ範囲: min={np.min(self.background_data):.3f}, max={np.max(self.background_data):.3f}')
        return True

    def setup_output_directory(self):
        """Setup output directory for saving results"""
        self.output_dir = os.path.join(os.path.dirname(self.data_path), 'SN_comparison')
        os.makedirs(self.output_dir, exist_ok=True)
        print(f'出力ディレクトリ: {self.output_dir}')
    
    def display_bscan(self):
        """Display B-scan data with interactive functionality"""
        self.fig, self.ax = plt.subplots(figsize=(20, 10))
        
        # Calculate extent for proper axis scaling
        extent = [0, self.bscan_data.shape[1] * trace_interval,
                 self.bscan_data.shape[0] * sample_interval * 1e9, 0]
        
        # Display B-scan
        # vmax = np.nanmax(np.abs(self.bscan_data)) / 10
        vmax = 100
        im = self.ax.imshow(self.bscan_data, aspect='auto', cmap='seismic',
                           extent=extent, vmin=-vmax, vmax=vmax)
        
        # Set labels and formatting
        self.ax.set_xlabel('Moving distance [m]', fontsize=font_medium)
        self.ax.set_ylabel('Time [ns]', fontsize=font_medium)
        self.ax.tick_params(axis='both', which='major', labelsize=font_small)
        
        # Add the second Y-axis for depth
        ax2 = self.ax.twinx()
        t_min, t_max = self.ax.get_ylim()
        depth_min = (t_min * 1e-9) * c / np.sqrt(epsilon_r) / 2
        depth_max = (t_max * 1e-9) * c / np.sqrt(epsilon_r) / 2
        ax2.set_ylim(depth_min, depth_max)
        ax2.set_ylabel(r'Depth [m] ($\varepsilon_r = 4.5$)', fontsize=font_medium)
        ax2.tick_params(axis='y', which='major', labelsize=font_small)
        
        # Add colorbar with more space
        self.fig.subplots_adjust(bottom=0.15, right=0.85, top=0.75, left=0.1)
        cbar_ax = self.fig.add_axes([0.65, 0.02, 0.15, 0.03])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=font_small-2)
        
        # Add instructions in figure coordinates for better positioning
        instruction_text = f"Current Group: {self.current_group}\n"
        instruction_text += "Controls:\n"
        instruction_text += "'c': Toggle click mode ON/OFF\n"
        instruction_text += "'n': Create new group\n"
        instruction_text += "'r': Remove last point\n"
        instruction_text += "'e': Exit and save"
        
        self.fig.text(0.02, 0.98, instruction_text, transform=self.fig.transFigure,
                     verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                     facecolor="white", alpha=0.9), fontsize=font_small-1)
        
        # Add mode status display
        self.update_mode_display()
        
        # Connect event handlers
        self.click_cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.key_cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.title('SN Comparison Tool - B-scan Data', fontsize=font_large)
        
        print(f"Event handlers connected:")
        print(f"Click handler ID: {self.click_cid}")
        print(f"Key handler ID: {self.key_cid}")
        print(f"Canvas backend: {self.fig.canvas.get_supported_filetypes()}")
        
        # Show plot and keep it interactive
        plt.show(block=True)
    
    def on_click(self, event):
        """Handle mouse click events"""
        print(f"\n=== CLICK DEBUG ===")
        print(f"Event received: {event}")
        print(f"Button: {event.button}")
        print(f"Event inaxes: {event.inaxes}")
        print(f"Self.ax: {self.ax}")
        print(f"Inaxes == self.ax: {event.inaxes == self.ax}")
        print(f"Click mode: {self.click_mode}")
        print(f"Event xdata: {event.xdata}")
        print(f"Event ydata: {event.ydata}")
        
        if event.inaxes != self.ax:
            print("Click outside main axis - ignoring")
            return
            
        if not self.click_mode:
            print("Click mode is OFF - ignoring click")
            return
            
        if event.button == 1:  # Left click
            print("Processing left click...")
            x_coord = event.xdata  # Moving distance [m]
            time_ns = event.ydata  # Time [ns]
            
            # Convert coordinates to array indices
            x_idx = int(round(x_coord / trace_interval))
            t_idx = int(round(time_ns / (sample_interval * 1e9)))
            
            print(f"Coordinates: x={x_coord:.3f}m, time={time_ns:.3f}ns")
            print(f"Array indices: x_idx={x_idx}, t_idx={t_idx}")
            print(f"Array shape: {self.bscan_data.shape}")
            
            # Boundary check
            if (0 <= x_idx < self.bscan_data.shape[1] and 
                0 <= t_idx < self.bscan_data.shape[0]):
                print("Coordinates within bounds - proceeding")
                
                # Get amplitude value
                amplitude = self.bscan_data[t_idx, x_idx]
                print(f"Amplitude at [{t_idx}, {x_idx}]: {amplitude}")
                
                # Add point to current group
                group_name = f"group_{self.current_group}"
                if group_name not in self.groups:
                    self.groups[group_name] = []
                    print(f"Created new group: {group_name}")
                
                point_data = {
                    "x": float(x_coord),
                    "time_ns": float(time_ns),
                    "amplitude": float(amplitude)
                }
                print(f"Point data created: {point_data}")
                
                self.groups[group_name].append(point_data)
                print(f"Point added to group. Group contents: {self.groups[group_name]}")
                
                # Plot the point
                colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
                color = colors[(self.current_group - 1) % len(colors)]
                self.ax.scatter(x_coord, time_ns, c=color, s=50, marker='o', 
                               edgecolors='black', linewidth=1, zorder=10)
                print(f"Point plotted with color: {color}")
                
                # Update the plot
                self.fig.canvas.draw()
                print("Plot updated")
                
                print(f"Point added to {group_name}: x={x_coord:.2f}m, t={time_ns:.2f}ns, amp={amplitude:.3f}")
                print(f"Total groups: {len(self.groups)}, Total points in {group_name}: {len(self.groups[group_name])}")
                print(f"All groups: {list(self.groups.keys())}")
            else:
                print(f"Coordinates out of bounds! x_idx={x_idx} (max={self.bscan_data.shape[1]-1}), t_idx={t_idx} (max={self.bscan_data.shape[0]-1})")
        else:
            print(f"Non-left click detected: button={event.button}")
    
    def on_key(self, event):
        """Handle keyboard events"""
        print(f"Key pressed: {event.key}")
        
        if event.key == 'c':  # Toggle click mode
            self.click_mode = not self.click_mode
            self.update_mode_display()
            status = "ON" if self.click_mode else "OFF"
            print(f"Click mode toggled: {status}")
            
        elif event.key == 'n':  # New group
            self.current_group += 1
            self.update_instruction_text()
            print(f"New group created: Group {self.current_group}")
            
        elif event.key == 'r':  # Remove last point
            self.remove_last_point()
            
        elif event.key == 'e':  # Exit
            self.save_data_and_exit()
    
    def update_instruction_text(self):
        """Update instruction text with current group"""
        # Clear previous text and add new one
        for text in self.fig.texts:
            if "Current Group:" in text.get_text():
                text.remove()
                break
        
        instruction_text = f"Current Group: {self.current_group}\n"
        instruction_text += "Controls:\n"
        instruction_text += "'c': Toggle click mode ON/OFF\n"
        instruction_text += "'n': Create new group\n"
        instruction_text += "'r': Remove last point\n"
        instruction_text += "'e': Exit and save"
        
        self.fig.text(0.02, 0.98, instruction_text, transform=self.fig.transFigure,
                     verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                     facecolor="white", alpha=0.9), fontsize=font_small-1)
        
        # Update mode display as well
        self.update_mode_display()
        
        self.fig.canvas.draw()
    
    def update_mode_display(self):
        """Update mode status display"""
        # Remove previous mode text if exists
        if self.mode_text is not None:
            self.mode_text.remove()
        
        # Create new mode text using figure coordinates
        mode_status = "CLICK MODE: ON" if self.click_mode else "CLICK MODE: OFF"
        color = "green" if self.click_mode else "red"
        
        self.mode_text = self.fig.text(0.82, 0.98, mode_status, transform=self.fig.transFigure,
                                      verticalalignment='top', fontsize=font_medium,
                                      bbox=dict(boxstyle="round,pad=0.5", 
                                              facecolor=color, alpha=0.8),
                                      color='white', weight='bold')
        
        self.fig.canvas.draw()
    
    def remove_last_point(self):
        """Remove the last point from current group"""
        group_name = f"group_{self.current_group}"
        if group_name in self.groups and self.groups[group_name]:
            removed_point = self.groups[group_name].pop()
            print(f"Removed point from {group_name}: x={removed_point['x']:.2f}m, t={removed_point['time_ns']:.2f}ns")
            
            # Clear and redraw all points
            self.redraw_points()
        else:
            print(f"No points to remove from {group_name}")
    
    def redraw_points(self):
        """Redraw all points on the plot"""
        # Clear previous scatter points (remove only scatter plots, keep the image)
        # This is a simple approach - we'll clear and redraw
        
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Remove all scatter plots by clearing and redisplaying image
        self.ax.clear()
        
        # Redisplay B-scan
        extent = [0, self.bscan_data.shape[1] * trace_interval,
                 self.bscan_data.shape[0] * sample_interval * 1e9, 0]
        vmax = 100
        self.ax.imshow(self.bscan_data, aspect='auto', cmap='seismic',
                      extent=extent, vmin=-vmax, vmax=vmax)
        
        # Reset labels
        self.ax.set_xlabel('Moving distance [m]', fontsize=font_medium)
        self.ax.set_ylabel('Time [ns]', fontsize=font_medium)
        self.ax.tick_params(axis='both', which='major', labelsize=font_small)
        
        # Restore title
        plt.title('SN Comparison Tool - B-scan Data', fontsize=font_large)
        
        # Redraw all points
        for i, (group_name, points) in enumerate(self.groups.items()):
            if points:
                color = colors[i % len(colors)]
                x_coords = [point['x'] for point in points]
                time_coords = [point['time_ns'] for point in points]
                self.ax.scatter(x_coords, time_coords, c=color, s=50, marker='o',
                               edgecolors='black', linewidth=1, zorder=10)
        
        self.fig.canvas.draw()
    
    def save_data_and_exit(self):
        """Save JSON data and generate analysis plots"""
        print(f"\n=== Saving Data ===")
        print(f"Total groups collected: {len(self.groups)}")
        for group_name, points in self.groups.items():
            print(f"{group_name}: {len(points)} points")
        
        # Save JSON data
        json_path = os.path.join(self.output_dir, 'point_data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.groups, f, indent=2, ensure_ascii=False)
        
        print(f"Point data saved to: {json_path}")
        
        # Close interactive plot
        plt.close(self.fig)
        
        # Generate analysis plots
        self.generate_analysis_plots()
        
        print("Analysis complete. Check SN_comparison directory for results.")
    
    def generate_analysis_plots(self):
        """Generate background analysis and comparison plots"""
        if self.background_data is None:
            print("Error: Background data not loaded")
            return
            
        # Calculate time array
        time_array = np.arange(0, len(self.background_data) * sample_interval / 1e-9, 
                              sample_interval / 1e-9)
        
        # Generate two plots: full range and 0-200ns
        self.create_comparison_plot(time_array, self.background_data, self.background_std, range_type="full")
        self.create_comparison_plot(time_array, self.background_data, self.background_std, range_type="0-200ns")

    def create_comparison_plot(self, time_array, background_data, background_std, range_type="full"):
        """Create comparison plot for specified time range"""
        fig, ax = plt.subplots(figsize=(6, 10), tight_layout=True)
        
        # Convert background data to log scale for plotting
        # Add small value to avoid log(0) errors
        background_log = np.log(np.abs(background_data) + 1e-10)
        background_std_log = np.log(np.abs(background_std) + 1e-10)
        
        # Plot background data
        ax.plot(background_log, time_array, 'b-', linewidth=2, label='Background log amplitude')
        ax.fill_betweenx(time_array, background_log - background_std_log, background_log + background_std_log, color='b', alpha=0.6)

        # Plot points from all groups
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
        group_stats = {}
        
        print(f"\n=== Creating {range_type} plot ===")
        print(f"Number of groups to plot: {len(self.groups)}")
        
        for i, (group_name, points) in enumerate(self.groups.items()):
            if not points:
                print(f"Skipping {group_name}: no points")
                continue
                
            print(f"Processing {group_name}: {len(points)} points")
            color = colors[i % len(colors)]
            
            # Extract data for this group
            times = [point['time_ns'] for point in points]
            amplitudes = [np.log(abs(point['amplitude'])) if point['amplitude'] != 0 else -10 for point in points]
            
            # Calculate statistics
            mean_time = np.mean(times)
            mean_amp = np.mean(amplitudes)
            std_amp = np.std(amplitudes)
            
            group_stats[group_name] = {
                'mean_time': mean_time,
                'mean_amp': mean_amp,
                'std_amp': std_amp,
                'times': times,
                'amplitudes': amplitudes
            }
            
            # Plot individual points
            ax.scatter(amplitudes, times, c=color, s=30, alpha=0.7, 
                      label=f'{group_name} points')
            print(f"Plotted {len(amplitudes)} points for {group_name}")
            
            # Plot group mean with error bar
            ax.errorbar(mean_amp, mean_time, xerr=std_amp, 
                       fmt='o', color=color, markersize=8, 
                       capsize=5, capthick=2, linewidth=2,
                       label=f'{group_name} mean±std')
            print(f"Plotted mean point for {group_name}: amp={mean_amp:.3f}, time={mean_time:.3f}")
        
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
        
        print(f"Comparison plot saved: {filename_png}")
    
    def run(self):
        """Main execution function"""
        # Get input file path
        data_path = input('B-scanファイルのパスを入力してください: ').strip()
        print(' ')
        print(f'B-scanファイル読み込み中: {data_path}')

        # Path validation
        if not os.path.exists(data_path):
            print('エラー: 指定されたファイルが存在しません')
            return False
        
        # Load B-scan data
        if not self.load_bscan_data(data_path):
            return False

        # Calculate background data
        if not self.calc_background_data():
            return False
        
        # Setup output directory
        self.setup_output_directory()
        
        # Initialize groups with Group 1
        self.groups = {}
        
        print("\n=== SN Comparison Tool ===")
        print("Instructions:")
        print("- Press 'c' to toggle click mode ON/OFF")
        print("- When click mode is ON, click on B-scan to add points")
        print("- Press 'n' to create a new group")
        print("- Press 'r' to remove last point from current group")
        print("- Press 'e' to exit and generate analysis")
        print("- Current group: Group 1")
        print("- Click mode: OFF (press 'c' to enable)")
        print("\nDisplaying B-scan data...")
        
        # Display interactive B-scan
        self.display_bscan()
        
        return True

def main():
    """Main function"""
    tool = SNComparisonTool()
    tool.run()

if __name__ == "__main__":
    main()