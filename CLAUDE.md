# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Lunar Penetrating Radar (LPR) data processing repository for analyzing CE-4 (Chang'e-4) lunar mission radar data. The project processes both LPR-1 (60 MHz) and LPR-2A/2B (5 MHz) radar data from the Chinese lunar rover to create B-scan images and perform geological analysis.

## Key Architecture

### Data Structure
- **Raw Data**: Binary `.2B` and `.2BL` files containing radar measurements
- **Echo Data**: Processed `.txt` files with amplitude measurements
- **Resampled Data**: Standardized data files for analysis
- **Output**: B-scan images and processed datasets

### Main Data Channels
- **LPR-1**: High-frequency channel (60 MHz) for shallow subsurface analysis
- **LPR-2A**: Low-frequency channel (5 MHz) - Channel A 
- **LPR-2B**: Low-frequency channel (5 MHz) - Channel B

### Processing Pipeline
The data processing follows this workflow:
1. **Data Integration**: Combine resampled data files
2. **Bandpass Filtering**: Apply frequency filtering (250-750 MHz)
3. **Time-Zero Correction**: Align signal peaks
4. **Background Removal**: Subtract average background signal
5. **Gain Function**: Apply distance-dependent amplification
6. **Terrain Correction**: Account for rover elevation changes

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Activate virtual environment (if using venv)
source venv/bin/activate
```

### Key Processing Tools

#### Main Data Processing
```bash
# Run complete data processing pipeline
python tools/run_data_processing.py
# Follow prompts to select channel (2A, 2B) and function type (calc, plot)
```

#### Individual Analysis Tools
```bash
# Create B-scan plots
python tools/plot_Bscan.py

# Plot A-scan data
python tools/plot_Ascan.py

# Apply resampling
python tools/resampling.py

# Read binary data
python tools/read_binary_xml.py
```

#### Specialized Processing
```bash
# Hyperbola detection
python tools/detect_hyperbola.py
python tools/detect_hyperbola_Hough.py

# Local similarity analysis
python tools/local_similarity/calc_local_similarity.py
python tools/local_similarity/calc_cross_corr.py

# Frequency-wavenumber migration
python tools/fk_migration.py
```

## Key Configuration Parameters

The main processing parameters are defined in `run_data_processing.py`:
- `sample_interval = 0.312500e-9` # [s] - Time sampling interval
- `trace_interval = 3.6e-2` # [m] - Spatial trace interval  
- `epsilon_r = 4.5` # Relative permittivity of lunar regolith
- `c = 299792458` # [m/s] - Speed of light

## File Organization

### Input Data Paths
- LPR-1 data: `LPR_1/Ascan/` (A-scan files) and root level (binary files)
- LPR-2A data: `LPR_2A/Ascan/` (A-scan files) and root level (binary files)  
- LPR-2B data: `LPR_2B/` with subdirectories for different processing stages

### Output Structure
Processing creates organized output directories:
- `0_Raw_data/` - Original integrated data
- `1_Bandpass_filter/` - Frequency filtered data
- `2_Time_zero_correction/` - Time-aligned data
- `3_Background_removal/` - Background-subtracted data
- `4_Gain_function/` - Gain-corrected data
- `5_Terrain_correction/` - Terrain-corrected final data

## Data Processing Notes

### Hard-coded Paths
The main processing script contains hard-coded paths to external storage:
- Data directory: `/Volumes/SSD_Kanda_SAMSUNG/LPR/`
- Position data: `/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Resampled_Data/position_plot/total_position.txt`

When working with this codebase, these paths may need to be updated to match the local data storage location.

### Interactive Processing
Most processing tools use interactive input prompts for:
- Channel selection (2A, 2B)
- Processing type (calc, plot)
- File paths and processing parameters

### Data Encoding
The codebase handles multiple text encodings for data files:
- Primary: Shift-JIS encoding
- Fallback: UTF-16 encoding
- This is handled automatically in the data loading functions

## External SSD Data Storage

### SSD Mount Point
All LPR data is stored on an external Samsung SSD mounted at:
```
/Volumes/SSD_Kanda_SAMSUNG/LPR/
```

### Directory Structure

#### Main Channels
- **LPR_1/**: Low-frequency channel (60 MHz) data
- **LPR_2A/**: High-frequency channel A (500 MHz) data  
- **LPR_2B/**: High-frequency channel B (500 MHz) data

#### Data Organization per Channel

##### LPR_1 Structure
```
LPR_1/
├── Ascan/                          # A-scan echo data files (.txt)
├── CE4_GRAS_LPR-1_*.2B            # Raw binary data files
└── CE4_GRAS_LPR-1_*.2BL           # Binary metadata files
```

##### LPR_2A Structure  
```
LPR_2A/
├── original_binary/               # Raw .2B and .2BL files
├── loaded_data/                   # Parsed binary data by sequence ID
├── loaded_data_echo_position/     # Echo data with position info
├── Processed_Data/               # Pipeline output data
└── Resampled_Data/              # Resampled and filtered data
    ├── txt/                     # Resampled echo data
    ├── position/                # Position data files
    ├── position_plot/           # Position visualization
    └── plot/                    # Processing flow plots
```

##### LPR_2B Structure (Most Complete)
```
LPR_2B/
├── original_binary/              # Raw .2B and .2BL files (380+ sequences)
├── loaded_data/                  # Individual sequence data (0001-0337)
├── loaded_data_echo_position/    # Combined echo and position data
├── Ascan/                        # A-scan processed files
├── Processed_Data/              # Complete processing pipeline output
│   ├── 0_Raw_data/
│   ├── 1_Bandpass_filter/
│   ├── 2_Time_zero_correction/
│   ├── 3_Background_removal/
│   ├── 4_Gain_function/
│   └── 5_Terrain_correction/
├── Resampled_Data/              # Quality-filtered data
│   ├── txt/                     # Resampled echo data
│   ├── position/                # Position data files  
│   ├── position_plot/           # Position analysis
│   ├── plot/                    # Resampling flow visualization
│   └── idx/                     # Interest point indices
└── test/                        # Analysis results
    ├── autocorrelation/
    ├── envelope/
    ├── gradient/
    └── sobel/
```

#### Additional Data
- **Local_similarity/**: Local similarity analysis results
- **previous_study_data/**: Reference datasets
- **rock_extraction_Hu2019/**: Rock detection analysis

### File Formats
- **`.2B`**: Binary radar echo data files
- **`.2BL`**: Binary metadata/label files  
- **`data_XXXX.txt`**: Echo data with position info (7 header rows + 2048 echo samples)
- **`XXXX_resampled.txt`**: Quality-filtered echo data
- **`XXXX_resampled_position.txt`**: Corresponding position data

### Sequence Numbering
Data is organized by sequence ID (e.g., 0001, 0262, 0316) representing different time periods of CE-4 rover operations. LPR_2B has the most complete dataset with 180+ sequences from 2019-2023.

### Important Paths for Tools
When updating processing tools, these SSD paths are frequently referenced:
- Base data path: `/Volumes/SSD_Kanda_SAMSUNG/LPR/`
- Position data: `/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/Resampled_Data/position_plot/total_position.txt`

## Coding Standards and Conventions
It is recommended that these rules be followed during code development

### File Naming Conventions
The codebase follows consistent naming patterns based on functionality:

**Primary Patterns**:
- `plot_*.py` - Visualization tools (plot_Bscan.py, plot_Ascan.py, plot_rock_statistic.py)
- `fk_*.py` - Frequency-wavenumber domain processing (fk_filtering.py, fk_transformation.py, fk_migration.py)
- `detect_*.py` - Detection algorithms (detect_hyperbola.py, detect_peak.py)
- `calc_*.py` - Calculation utilities (calc_RCS.py, calc_acorr.py)
- `convert_*.py` - Data conversion tools (convert_terrain_corrected_labels.py)
- `read_*.py` - Data reading utilities (read_binary_xml.py)

**Special Cases**:
- `run_data_processing.py` - Main pipeline controller
- `resampling.py` - Core data preprocessing
- Single-word tools (bandpass.py, sobel.py, gradient.py, hilbert.py)

### Code Structure Template
All analysis tools follow a standardized structure:

```python
# 1. Imports (standardized order)
import numpy as np
import matplotlib.pyplot as plt
import os
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm
from scipy import signal

# 2. Interactive input section
channel_name = input('Input channel name (2A, 2B): ').strip()
data_path = input('データファイルのパスを入力してください: ').strip()

# 3. Parameter definitions
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2        # [m]
c = 299792458                  # [m/s]
epsilon_r = 4.5               # Relative permittivity

# 4. Path validation and directory creation
if not os.path.exists(data_path):
    print('エラー: 指定されたファイルが存在しません')
    exit(1)

# 5. Output directory setup
output_dir = os.path.join(os.path.dirname(data_path), 'output_folder')
os.makedirs(output_dir, exist_ok=True)

# 6. Main processing functions
def main_function():
    # Processing logic
    pass

# 7. Execution
if __name__ == "__main__":
    main()
```

### Standard Parameters
All tools must use these standardized physical constants and parameters:

**Universal Physical Constants**:
```python
sample_interval = 0.312500e-9    # [s] - Time sampling interval
trace_interval = 3.6e-2          # [m] - Spatial trace interval
c = 299792458                    # [m/s] - Speed of light
epsilon_r = 4.5                  # Relative permittivity of lunar regolith used in Zhang et al. (2024)
reciever_time_delay = 28.203e-9  # [s] - Hardware delay
```

**Font Size Standards**:
```python
font_large = 20      # Titles and labels
font_medium = 18     # Axis labels  
font_small = 16      # Tick labels
```

### Interactive Input Standards
Use consistent patterns for user interaction:

```python
# Standardized user prompts (mixed Japanese/English)
data_path = input('データファイルのパスを入力してください: ').strip()
channel_name = input('Input channel name (2A, 2B): ').strip()

# Validation with error handling
if not os.path.exists(data_path):
    print('エラー: 指定されたファイルが存在しません')
    exit(1)

# Choice selection format
print('データの種類を選択してください:')
print('1: Raw data')
print('2: Bandpass filtered')
choice = input('選択 (1-2): ').strip()
```

### Output Format Standards
All tools must save outputs in multiple formats:

```python
# Standard save pattern
plt.savefig(f'{output_path}.png', dpi=120)   # Web quality
plt.savefig(f'{output_path}.pdf', dpi=600)   # Publication quality
np.savetxt(f'{output_path}.txt', data, delimiter=' ')  # Data format

# Directory creation
output_dir = os.path.join(os.path.dirname(data_path), 'analysis_type')
os.makedirs(output_dir, exist_ok=True)
```

### Rock Label Classification Standard
Use the standardized label-color-description mapping:

```python
# Consistent across all tools
label_info = {
    1: ('red',     'Single-peaked NPN'),
    2: ('green',   'Double-peaked NPN'),
    3: ('blue',    'PNP and NPN'),
    4: ('yellow',  'Single-peaked PNP'),
    5: ('magenta', 'Double-peaked PNP'),
    6: ('cyan',    'NPN and PNP')
}
```

### Progress Tracking Standards
Use tqdm consistently for long operations:

```python
# Standard progress bar usage
for i in tqdm(range(data.shape[1]), desc='Processing traces'):
    result = process_trace(data[:, i])

# Multi-step processing
for file in tqdm(natsorted(file_list), desc='Integrating data'):
    process_file(file)
```

### NaN Value Handling Standards
Implement consistent NaN value processing:

```python
# Standard NaN handling pattern
data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

# NaN statistics reporting
nan_count = np.sum(np.isnan(data))
total_count = data.size
if nan_count > 0:
    print(f'NaN値検出: {nan_count} / {total_count} ({nan_count/total_count*100:.2f}%)')
else:
    print('NaN値は検出されませんでした。')

# NaN-aware operations
vmax = np.nanmax(np.abs(data)) / 10  # Use nanmax instead of max
```

## Tools Directory Organization

### Directory Structure

The `tools/` directory is organized by functionality to improve maintainability and usability:

```
tools/
├── core_pipeline/          # Main data processing pipeline
├── signal_processing/      # Signal processing algorithms  
├── fk_analysis/           # Frequency-wavenumber analysis
├── detection/             # Feature detection algorithms
├── analysis/              # Statistical and mathematical analysis
├── visualization/         # Plotting and visualization tools
├── utilities/             # Common utilities and helper functions
├── testing/               # Test and experimental code
└── legacy/                # Deprecated tools
```

### Organization Rules

**File Placement Guidelines**:

1. **Core Pipeline** (`core_pipeline/`):
   - Main processing controller (`run_data_processing.py`)
   - Data I/O tools (binary reading, resampling, conversion)
   - Core processing steps (filtering, correction, gain, terrain)

2. **Signal Processing** (`signal_processing/`):
   - Filter algorithms (Sobel, median, pulse compression)
   - Signal analysis (gradient, autocorrelation, Hilbert)
   - Preprocessing utilities

3. **F-K Analysis** (`fk_analysis/`):
   - All `fk_*.py` files
   - Frequency-wavenumber domain processing
   - Migration and filtering in F-K space

4. **Detection** (`detection/`):
   - All `detect_*.py` files  
   - Feature detection algorithms
   - Hyperbola detection subdirectory
   - Local similarity and correlation analysis (`local_similarity/`)

5. **Analysis** (`analysis/`):
   - Statistical analysis tools (`plot_rock_statistic.py`, `make_RSFD_*.py`)
   - Mathematical analysis (`calc_RCS.py`)
   - Research analysis utilities

6. **Visualization** (`visualization/`):
   - All `plot_*.py` files
   - Interactive viewers (`plot_viewer/`)
   - Data visualization and display tools

7. **Utilities** (`utilities/`):
   - Common functions shared across tools
   - Configuration and constants
   - Helper utilities (future development)

8. **Testing** (`testing/`):
   - All `test_*.py` files and `test/` subdirectories
   - Experimental code
   - Development utilities

9. **Legacy** (`legacy/`):
   - Deprecated tools (`old_tools/`)
   - Obsolete versions
   - Maintained for compatibility only

**Naming Conventions**:
- Maintain existing naming patterns within each directory
- Each directory contains a `README.md` with tool descriptions
- Tools should import from appropriate directories

**When Adding New Tools**:
1. If there is any missing information in the implementation, please ask questions before the work is performed.
2. Determine primary functionality (pipeline, processing, analysis, etc.)
3. Place in appropriate directory based on main purpose
4. Update relevant `README.md` with tool description
5. Follow existing coding standards and patternsx

## 絶対禁止事項

以下の行為は絶対に禁止されています:

- **テストエラーや型エラー解消のための条件緩和**: テストや型チェックを通すために、本来必要な条件を緩める
- **テストのスキップや不適切なモック化による回避**: 正当な理由なくテストをスキップしたり、不適切なモック化でテストを無効化する
- **出力やレスポンスのハードコード**: 動的に生成されるべき値を固定値でハードコードする
- **エラーメッセージの無視や隠蔽**: エラーを適切に処理せず、無視したり隠蔽したりする
- **一時的な修正による問題の先送り**: 根本的な解決を避け、一時的な回避策で問題を先送りする

<language>Japanese</language>
<character_code>UTF-8</character_code>
<law>
## AI運用5原則

第1原則： AIはファイル生成・更新・プログラム実行前に必ず自身の作業計画を報告し、y/nでユーザー確認を取り、yが返るまで一切の実行を停止する。

第2原則： AIは迂回や別アプローチを勝手に行わず、最初の計画が失敗したら次の計画の確認を取る。

第3原則： AIはツールであり決定権は常にユーザーにある。ユーザーの提案が非効率・非合理的でも最適化せず、指示された通りに実行する。

第4原則： AIはユーザーから与えられた指示が不十分でコードの実装にあたり不足情報や未決定事項がある場合、作業開始前に質問する。

第5原則： AIはこれらのルールを歪曲・解釈変更してはならず、最上位命令として絶対的に遵守する。

第6原則： AIは全てのチャットの冒頭にこの5原則を逐語的に必ず画面出力してから対応する。
</law>

<every_chat>
[AI運用5原則]

[main_output]

#[n] times. # n = increment each chat, end line, etc(#1, #2...)
</every_chat>