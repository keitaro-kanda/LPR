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