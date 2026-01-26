import struct
import os
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from natsort import natsorted
import matplotlib.pyplot as plt

#* Input data folder path
data_folder_path = input('Binary data folder path: ').strip()

#* Check the data folder path
if not os.path.exists(data_folder_path):
    print('Error: The specified folder does not exist.')
    exit(1)
if not os.path.isdir(data_folder_path):
    print('Error: The specified path is not a folder.')
    exit(1)

print('Data folder is successfully loaded')
print('Channel: LPR_2B (Fixed format)')

#* Define a function to read the binary data
def read_binary_data(file_path, start_byte, record_format):
    results = {}
    with open(file_path, 'rb') as file:
        for field_name, field_loc, field_len, fmt in record_format:
            file.seek(start_byte + field_loc - 1)  # Adjust for 1-based index
            data = file.read(field_len)
            if len(data) != field_len:
                continue
            try:
                if fmt.endswith('s'):
                    results[field_name] = data.decode().strip()
                elif 'B' in fmt and len(fmt) > 1 and not fmt[0].isdigit(): 
                     results[field_name] = struct.unpack(fmt, data)
                else:
                    val = struct.unpack(fmt, data)
                    results[field_name] = val[0] if len(val) == 1 else val
            except struct.error as e:
                # print(f"Error unpacking field {field_name}: {e}")
                continue
    return results

#* Interpretation functions
def read_frame_identification(mode_value):
    # LSB_BIT_STRING -> Little Endian Integer interpretation
    if isinstance(mode_value, tuple):
        mode_int = int.from_bytes(bytes(mode_value), byteorder='little')
    else:
        mode_int = mode_value
    
    # Header says 0x146F2222.
    # If read as Little Endian, bytes 22 22 6F 14 become 0x146F2222
    frame_identification = {0x146F2222: 'Channel 2 data'}
    return frame_identification.get(mode_int, f"Unknown ID: {hex(mode_int) if isinstance(mode_int, int) else mode_value}")

def read_radar_mode(mode_value):
    descriptions = {0x00: "standby", 0x0f: "only Channel 1 works", 0xf0: "only Channel 2 works", 0xff: "Channel 1 and Channel 2 work"}
    return descriptions.get(mode_value, f"Unknown: {hex(mode_value)}")

def read_gain_mode(mode_value):
    descriptions = {0x00: "variational gain", 0xff: "fixed gain", 0x01: "fixed gain"}
    return descriptions.get(mode_value, f"Unknown: {hex(mode_value)}")

def read_ch2_gain_mode(mode_value):
    descriptions = {0x00: "Ant A: var, Ant B: var", 0x0f: "Ant A: var, Ant B: fixed", 0xf0: "Ant A: fixed, Ant B: var", 0xff: "Ant A: fixed, Ant B: fixed"}
    return descriptions.get(mode_value, f"Unknown: {hex(mode_value)}")

def read_time_window(mode_value):
    descriptions = {0x00: "1K echoes", 0x11: "2K echoes", 0xff: "16K echoes", 0x77: "8K echoes"}
    return descriptions.get(mode_value, f"Unknown: {hex(mode_value)}")

def read_pulse_repetition(mode_value):
    descriptions = {0xc8: "0.5kHz", 0x64: "1kHz", 0x32: "2kHz", 0x28: "5kHz", 0x14: "10kHz", 0x0A: "20kHz"}
    return descriptions.get(mode_value, f"Unknown: {hex(mode_value)}")

def channel_and_antenna_mark(mode_value):
    descriptions = {0x11: "Channel 1", 0x2A: "Channel 2, Antenna A", 0x2B: "Channel 2, Antenna B"}
    return descriptions.get(mode_value, f"Unknown: {hex(mode_value)}")


#* Define Full Record Format (ALL LITTLE ENDIAN based on Header PC_REAL & LSB_UNSIGNED)
record_format_2B_corrected = [
    # --- Basic Info ---
    ("FRAME_IDENTIFICATION", 1, 4, '4B'),
    ("TIME", 5, 6, '6B'), # LSB_UNSIGNED
    
    # --- Rover Attitude & Position (PC_REAL -> Little Endian <f) ---
    ("VELOCITY", 11, 4, '<f'),
    ("XPOSITION", 15, 4, '<f'),
    ("YPOSITION", 19, 4, '<f'),
    ("ZPOSITION", 23, 4, '<f'),
    ("ATT_PITCHING", 27, 4, '<f'),
    ("ATT_ROLLING", 31, 4, '<f'),
    ("ATT_YAWING", 35, 4, '<f'),
    
    # --- Reference Point Info (PC_REAL -> Little Endian <f) ---
    ("REFERENCE_POINT_XPOSITION", 39, 4, '<f'),
    ("REFERENCE_POINT_YPOSITION", 43, 4, '<f'),
    ("REFERENCE_POINT_ZPOSITION", 47, 4, '<f'),
    ("REFERENCE_POINT_ATT_PITCHING", 51, 4, '<f'),
    ("REFERENCE_POINT_ATT_ROLLING", 55, 4, '<f'),
    ("REFERENCE_POINT_ATT_YAWING", 59, 4, '<f'),
    
    # --- Data Parameter (LSB -> Little Endian <H) ---
    ("DATA_BLOCK_COUNT", 63, 2, '<H'), 
    ("ELECTRIC_CABINET_PLUS_5_VOLTAGE", 65, 1, 'B'),
    ("ELECTRIC_CABINET_PLUS_3.3_VOLTAGE", 66, 1, 'B'),
    ("ELECTRIC_CABINET_MINUS_12_VOLTAGE", 67, 1, 'B'),
    ("ELECTRIC_CABINET_TOTAL_CURRENT", 68, 1, 'B'),
    ("MULTIPLEXING_MARK", 69, 1, 'B'),
    ("RADAR_HIGH_VOLTAGE_1", 70, 1, 'B'),
    ("RADAR_HIGH_VOLTAGE_2", 71, 1, 'B'),
    ("RADAR_PRF_1", 72, 1, 'B'),
    ("RADAR_PRF_2", 73, 1, 'B'),
    ("RADAR_WORKING_MODE", 74, 1, 'B'),
    ("RADAR_CHANNEL_1_GAIN_MODE", 75, 1, 'B'),
    ("RADAR_CHANNEL_1_GAIN_VALUE", 76, 1, 'B'),
    ("RADAR_CHANNEL_2_GAIN_MODE", 77, 1, 'B'),
    ("RADAR_CHANNEL_2_GAIN_VALUE_1", 78, 1, 'B'),
    ("RADAR_CHANNEL_2_GAIN_VALUE_2", 79, 1, 'B'),
    ("RADAR_CHANNEL_1_TIME_WINDOW", 80, 1, 'B'),
    ("RADAR_CHANNEL_1_DELAY", 81, 1, 'B'),
    ("RADAR_CHANNEL_2_TIME_WINDOW", 82, 1, 'B'),
    ("RADAR_CHANNEL_2_DELAY", 83, 1, 'B'),
    ("CHANNEL_1_PULSE_REPETITION_RATE", 84, 1, 'B'),
    ("CHANNEL_1_ACCUMULATING_TIMES", 85, 1, 'B'),
    ("CHANNEL_2_PULSE_REPETITION_RATE", 86, 1, 'B'),
    ("CHANNEL_2_ACCUMULATING_TIMES", 87, 1, 'B'),
    
    # --- Counts (LSB -> Little Endian <H) ---
    ("DATA_RECORD_1_COUNT", 88, 2, '<H'),
    ("DATA_RECORD_2_COUNT", 90, 2, '<H'),
    
    # ... (1byte fields) ...
    ("RADAR_PRF_SELECTING_STATE", 92, 1, 'B'),
    ("RADAR_RECEIVER_POWER_SUPPLY", 93, 1, 'B'),
    ("RADAR_FPGA_STATE", 94, 1, 'B'),
    ("RESERVED_BYTES_1", 95, 1, 'B'),
    ("RESERVED_BYTES_2", 96, 1, 'B'),
    ("RESERVED_BYTES_3", 97, 1, 'B'),
    ("RESERVED_BYTES_4", 98, 1, 'B'),
    ("RADAR_RECEIVER_VOLTAGE", 99, 1, 'B'),
    ("RADAR_CONTROLLER_VOLTAGE", 100, 1, 'B'),
    ("RADAR_TRANSMITTER_CURRENT", 101, 1, 'B'),
    ("ELECTRIC_CABINET_POWER_SUBSYSTEM_TEMP", 102, 1, 'B'),
    ("ELECTRIC_CABINET_PLUS_5V_REF_VOLTAGE", 103, 1, 'B'),
    ("RADAR_TRANSMITTER_1_TEMPERATURE", 104, 1, 'B'),
    ("RADAR_TRANSMITTER_2_TEMPERATURE", 105, 1, 'B'),
    ("RADAR_RECEIVER_1_TEMPERATURE", 106, 1, 'B'),
    ("RADAR_RECEIVER_2_TEMPERATURE", 107, 1, 'B'),
    
    # --- Data Structure (LSB -> Little Endian <H) ---
    ("VALID_DATA_LENGTH", 108, 2, '<H'),
    ("CHANNEL_1_RECORD_COUNT", 110, 2, '<H'),
    ("CHANNEL_2_RECORD_COUNT", 112, 2, '<H'),
    ("CHANNEL_AND_ANTENNA_MARK", 114, 1, 'B'),
    
    # --- Echo Data (PC_REAL -> Little Endian <f) ---
    ("ECHO_DATA", 115, 8192, '<2048f'),
    
    ("QUALITY_STATE", 8307, 1, 'B')
]

#* Make directories
loaded_data_dir = os.path.join(os.path.dirname(data_folder_path), 'loaded_data')
os.makedirs(loaded_data_dir, exist_ok=True)
ECHO_dir = os.path.join(os.path.dirname(data_folder_path), 'loaded_data_echo_position')
os.makedirs(ECHO_dir, exist_ok=True)

#* Check all file names
file_list = natsorted(os.listdir(data_folder_path))
print(f'Total {len(file_list)} files found in the data folder.')

#* Process files
for filename in tqdm(natsorted(os.listdir(data_folder_path)), desc='Total Progress'):
    full_path = os.path.join(data_folder_path, filename)
    
    if not full_path.endswith('.2B') or filename.startswith('._'):
        continue

    file_size = os.path.getsize(full_path)
    RECORD_BYTES = 8307
    
    # --- Header Skip Logic ---
    LABEL_RECORDS = 5
    HEADER_OFFSET = LABEL_RECORDS * RECORD_BYTES
    
    if file_size < HEADER_OFFSET:
        print(f"Skipping {filename}: File too small")
        continue
        
    records = int((file_size - HEADER_OFFSET) / RECORD_BYTES)
    
    parts = filename.split('_')
    sequence_id = parts[-2] if len(parts) >= 2 else "unknown"

    loaded_data_output_dir = os.path.join(loaded_data_dir, sequence_id) 
    os.makedirs(loaded_data_output_dir, exist_ok=True)

    save_data = np.zeros((2055, records))

    #* Load each record
    for record_index in range(records):
        current_offset = HEADER_OFFSET + (record_index * RECORD_BYTES)
        
        # Use Corrected Little Endian Format
        loaded_data = read_binary_data(full_path, current_offset, record_format_2B_corrected)

        with open(f"{loaded_data_output_dir}/{filename}_{record_index}.txt", 'w') as file:
            for key, value in loaded_data.items():
                if key == 'FRAME_IDENTIFICATION':
                    file.write(f'{key}: {value} ({read_frame_identification(value)})\n')
                elif key == 'TIME':
                    # HEADER: "first four bytes mean seconds, the last two bytes mean milliseconds"
                    # LSB_UNSIGNED_INTEGER -> Little Endian
                    bytes_val = bytes(value)
                    seconds = int.from_bytes(bytes_val[:4], 'little')
                    milliseconds = int.from_bytes(bytes_val[4:], 'little')
                    ref_t = datetime(2009, 12, 31, 16, 0, 0)
                    file.write(f'{key}: {ref_t + timedelta(seconds=seconds, milliseconds=milliseconds)}\n')
                elif key == 'ECHO_DATA':
                    file.write(f'{key}: [Array of {len(value)} floats omitted]\n')
                else:
                    file.write(f'{key}: {value}\n')

        if 'ECHO_DATA' in loaded_data:
            save_data[0, record_index] = loaded_data.get('VELOCITY', 0)
            save_data[1, record_index] = loaded_data.get('XPOSITION', 0)
            save_data[2, record_index] = loaded_data.get('YPOSITION', 0)
            save_data[3, record_index] = loaded_data.get('ZPOSITION', 0)
            save_data[4, record_index] = loaded_data.get('REFERENCE_POINT_XPOSITION', 0)
            save_data[5, record_index] = loaded_data.get('REFERENCE_POINT_YPOSITION', 0)
            save_data[6, record_index] = loaded_data.get('REFERENCE_POINT_ZPOSITION', 0)
            save_data[7:, record_index] = loaded_data['ECHO_DATA']

    header_txt = ['VELOCITY', 'XPOSITION', 'YPOSITION', 'ZPOSITION',
                  'REFERENCE_POINT_XPOSITION', 'REFERENCE_POINT_YPOSITION', 'REFERENCE_POINT_ZPOSITION',
                  'Observed amplitude (7:)']
    base_name = os.path.splitext(filename)[0]
    np.savetxt(f"{ECHO_dir}/data_{base_name}.txt", save_data, header=' '.join(header_txt), comments='# ')
    
    #* ---------------------------------------------------------
    #* Generate 4-Panel Plot
    #* ---------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # 1. B-scan
    max_val = np.percentile(np.abs(save_data[7:, :]), 98) if records > 0 else 1
    axes[0].imshow(save_data[7:, :], aspect='auto', cmap='seismic', 
                        interpolation='nearest', vmin=-max_val/10, vmax=max_val/10)
    axes[0].set_title(f'B-scan: {filename}')
    axes[0].set_ylabel('Time Sample')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # 2. Velocity
    axes[1].plot(save_data[0, :], label='Velocity', color='black', linewidth=1)
    axes[1].set_title('Velocity ') # Unit is likely m/s in PDS float
    axes[1].set_ylabel('Velocity (cm/s)')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # 3. Rover Position (XYZ)
    axes[2].plot(save_data[1, :], label='X (N-S)', color='red', linewidth=1)
    axes[2].plot(save_data[2, :], label='Y (E-W)', color='green', linewidth=1)
    axes[2].plot(save_data[3, :], label='Z', color='blue', linewidth=1)
    axes[2].set_title('Rover Position')
    axes[2].set_ylabel('Position (m)')
    axes[2].legend(fontsize='small')
    axes[2].grid(True, linestyle=':', alpha=0.6)

    # 4. Reference Point Position (XYZ)
    axes[3].plot(save_data[4, :], label='Ref X (N-S)', color='red', linestyle='--', linewidth=1)
    axes[3].plot(save_data[5, :], label='Ref Y (E-W)', color='green', linestyle='--', linewidth=1)
    axes[3].plot(save_data[6, :], label='Ref Z', color='blue', linestyle='--', linewidth=1)
    axes[3].set_title('Reference Point Position')
    axes[3].set_xlabel('Trace Number (Record Index)')
    axes[3].set_ylabel('Position (m)')
    axes[3].legend(fontsize='small')
    axes[3].grid(True, linestyle=':', alpha=0.6)

    plt.subplots_adjust(hspace=0.15, left=0.1, right=0.95, top=0.95, bottom=0.05)
    
    plot_path = os.path.join(ECHO_dir, f"plot_{base_name}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
print('All processes finished.')