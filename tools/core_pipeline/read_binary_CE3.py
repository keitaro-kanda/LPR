import struct
import os
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse
from natsort import natsorted

#* Input data folder path
data_folder_path = input('Binary data folder path: ').strip()
# data_folder_path = '/Volumes/SSD_Kanda_BUFFALO/LPR/LPR_2B/original_binary' # デバッグ用

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
            file.seek(start_byte + field_loc - 1)  # Adjust for 1-based index from PDS header
            data = file.read(field_len)
            if len(data) != field_len:
                # 最後のレコードなどで数バイト足りない場合への対策（通常は発生しないはず）
                continue
            try:
                if fmt.endswith('s'):
                    results[field_name] = data.decode().strip()
                elif 'B' in fmt and len(fmt) > 1 and not fmt[0].isdigit(): 
                     # '4B'などのバイト列そのままの場合
                     results[field_name] = struct.unpack(fmt, data)
                else:
                    # 数値へ変換（タプルから値を取り出す）
                    val = struct.unpack(fmt, data)
                    results[field_name] = val[0] if len(val) == 1 else val
            except struct.error as e:
                print(f"Error unpacking field {field_name}: {e}")
                continue
    return results

#* Define a function to convert bytes to integer
def bytes_to_int(bytes_data):
    # バイト列がタプルで来る場合とbytesで来る場合に対応
    if isinstance(bytes_data, tuple):
        bytes_data = bytes(bytes_data)
    return int.from_bytes(bytes_data, byteorder='big') # TIME等の分割バイトはBigEndian扱いが多いが、個別の値はLittle

#* Define interpretation functions
def read_frame_identification(mode_value):
    # mode_valueはタプル(4 bytes)または整数で来る可能性があるため調整
    if isinstance(mode_value, tuple):
        mode_int = int.from_bytes(bytes(mode_value), byteorder='big') # 0x146F2222のようなIDはBigEndian順で読むのが一般的
    else:
        mode_int = mode_value
        
    frame_identification = {
        0x146F2222: 'Channel 2 data'
    }
    return frame_identification.get(mode_int, f"Unknown ID: {hex(mode_int) if isinstance(mode_int, int) else mode_value}")

def read_radar_mode(mode_value):
    descriptions = {
        0x00: "standby",
        0x0f: "only Channel 1 works",
        0xf0: "only Channel 2 works",
        0xff: "Channel 1 and Channel 2 work"
    }
    return descriptions.get(mode_value, f"Unknown: {hex(mode_value)}")

def read_gain_mode(mode_value):
    descriptions = {
        0x00: "variational gain",
        0xff: "fixed gain", # ヘッダー記述に基づく修正
        0x01: "fixed gain"  # 前コードにあった定義も維持
    }
    return descriptions.get(mode_value, f"Unknown: {hex(mode_value)}")

def read_ch2_gain_mode(mode_value):
    descriptions = {
        0x00: "Ant A: var, Ant B: var",
        0x0f: "Ant A: var, Ant B: fixed",
        0xf0: "Ant A: fixed, Ant B: var",
        0xff: "Ant A: fixed, Ant B: fixed"
    }
    return descriptions.get(mode_value, f"Unknown: {hex(mode_value)}")

def read_time_window(mode_value):
    descriptions = {
        0x00: "1K echoes",
        0x11: "2K echoes",
        0xff: "16K echoes", # Ch1 definition
        0x77: "8K echoes"   # Ch2 definition
    }
    return descriptions.get(mode_value, f"Unknown: {hex(mode_value)}")

def read_pulse_repetition(mode_value):
    descriptions = {
        0xc8: "0.5kHz", 0x64: "1kHz", 0x32: "2kHz", # Ch1
        0x28: "5kHz", 0x14: "10kHz", 0x0A: "20kHz"  # Ch2
    }
    return descriptions.get(mode_value, f"Unknown: {hex(mode_value)}")

def channel_and_antenna_mark(mode_value):
    descriptions = {
        0x11: "Channel 1",
        0x2A: "Channel 2, Antenna A", # 修正: ヘッダーでは2AがAntenna A
        0x2B: "Channel 2, Antenna B"
    }
    return descriptions.get(mode_value, f"Unknown: {hex(mode_value)}")


#* Define Full Record Format based on Header
#* 重要: PC_REAL -> '<f' (Little Endian float), LSB_UNSIGNED -> '<H'/'<I' (Little Endian int)
record_format_2B_full = [
    # --- Basic Info ---
    ("FRAME_IDENTIFICATION", 1, 4, '4B'),
    ("TIME", 5, 6, '6B'),
    
    # --- Rover Attitude & Position (PC_REAL -> <f) ---
    ("VELOCITY", 11, 4, '<f'),
    ("XPOSITION", 15, 4, '<f'),
    ("YPOSITION", 19, 4, '<f'),
    ("ZPOSITION", 23, 4, '<f'),
    ("ATT_PITCHING", 27, 4, '<f'),
    ("ATT_ROLLING", 31, 4, '<f'),
    ("ATT_YAWING", 35, 4, '<f'),
    
    # --- Reference Point Info (PC_REAL -> <f) ---
    ("REFERENCE_POINT_XPOSITION", 39, 4, '<f'),
    ("REFERENCE_POINT_YPOSITION", 43, 4, '<f'),
    ("REFERENCE_POINT_ZPOSITION", 47, 4, '<f'),
    ("REFERENCE_POINT_ATT_PITCHING", 51, 4, '<f'),
    ("REFERENCE_POINT_ATT_ROLLING", 55, 4, '<f'),
    ("REFERENCE_POINT_ATT_YAWING", 59, 4, '<f'),
    
    # --- Data Parameter 1 Expansion ---
    ("DATA_BLOCK_COUNT", 63, 2, '<H'), # LSB_UNSIGNED_INTEGER 2bytes
    ("ELECTRIC_CABINET_PLUS_5_VOLTAGE", 65, 1, 'B'),
    ("ELECTRIC_CABINET_PLUS_3.3_VOLTAGE", 66, 1, 'B'),
    ("ELECTRIC_CABINET_MINUS_12_VOLTAGE", 67, 1, 'B'), # 12 interpreted as -12
    ("ELECTRIC_CABINET_TOTAL_CURRENT", 68, 1, 'B'),
    ("MULTIPLEXING_MARK", 69, 1, 'B'),
    ("RADAR_HIGH_VOLTAGE_1", 70, 1, 'B'),
    ("RADAR_HIGH_VOLTAGE_2", 71, 1, 'B'),
    ("RADAR_PRF_1", 72, 1, 'B'),
    ("RADAR_PRF_2", 73, 1, 'B'),
    
    # --- Settings ---
    ("RADAR_WORKING_MODE", 74, 1, 'B'),
    ("RADAR_CHANNEL_1_GAIN_MODE", 75, 1, 'B'),
    ("RADAR_CHANNEL_1_GAIN_VALUE", 76, 1, 'B'),
    ("RADAR_CHANNEL_2_GAIN_MODE", 77, 1, 'B'),
    ("RADAR_CHANNEL_2_GAIN_VALUE_1", 78, 1, 'B'),
    ("RADAR_CHANNEL_2_GAIN_VALUE_2", 79, 1, 'B'),
    
    # --- Data Parameter 2 Expansion ---
    ("RADAR_CHANNEL_1_TIME_WINDOW", 80, 1, 'B'),
    ("RADAR_CHANNEL_1_DELAY", 81, 1, 'B'),
    ("RADAR_CHANNEL_2_TIME_WINDOW", 82, 1, 'B'),
    ("RADAR_CHANNEL_2_DELAY", 83, 1, 'B'),
    ("CHANNEL_1_PULSE_REPETITION_RATE", 84, 1, 'B'),
    ("CHANNEL_1_ACCUMULATING_TIMES", 85, 1, 'B'),
    ("CHANNEL_2_PULSE_REPETITION_RATE", 86, 1, 'B'),
    ("CHANNEL_2_ACCUMULATING_TIMES", 87, 1, 'B'),
    
    # --- Counts ---
    ("DATA_RECORD_1_COUNT", 88, 2, '<H'), # LSB 2bytes
    ("DATA_RECORD_2_COUNT", 90, 2, '<H'),
    
    # --- More Status ---
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
    
    # --- Data Structure ---
    ("VALID_DATA_LENGTH", 108, 2, '<H'),
    ("CHANNEL_1_RECORD_COUNT", 110, 2, '<H'),
    ("CHANNEL_2_RECORD_COUNT", 112, 2, '<H'),
    ("CHANNEL_AND_ANTENNA_MARK", 114, 1, 'B'),
    
    # --- Echo Data (2048 floats * 4 bytes = 8192 bytes) ---
    ("ECHO_DATA", 115, 8192, '<2048f'), # Little Endian Floats
    
    # --- Footer ---
    ("QUALITY_STATE", 8307, 1, 'B')
]


#* Make 'loaded_data' directory and 'ECHO' directory
loaded_data_dir = os.path.join(os.path.dirname(data_folder_path), 'loaded_data')
os.makedirs(loaded_data_dir, exist_ok=True)
ECHO_dir = os.path.join(os.path.dirname(data_folder_path), 'loaded_data_echo_position')
os.makedirs(ECHO_dir, exist_ok=True)


#* Check all file names in the data folder
file_list = natsorted(os.listdir(data_folder_path))
print(f'Total {len(file_list)} files found in the data folder.')

#* Output only the echo data as txt file
for filename in tqdm(natsorted(os.listdir(data_folder_path)), desc='Total Progress'):
    full_path = os.path.join(data_folder_path, filename)
    
    #* Skip non-2B files or hidden files
    if not full_path.endswith('.2B') or filename.startswith('._'):
        continue

    file_size = os.path.getsize(full_path)
    
    # Check if file size aligns with record bytes
    RECORD_BYTES = 8307
    if file_size % RECORD_BYTES != 0:
        print(f"Warning: {filename} size {file_size} is not a multiple of {RECORD_BYTES}")

    records = int(file_size / RECORD_BYTES)
    
    #* Get the sequence ID
    parts = filename.split('_')
    sequence_id = parts[-2] if len(parts) >= 2 else "unknown"

    #* Make output directory for loaded_data of each sequence
    loaded_data_output_dir = os.path.join(loaded_data_dir, sequence_id) 
    os.makedirs(loaded_data_output_dir, exist_ok=True)

    #* Make array for ECHO data (Rows: Velocity, XYZ, RefXYZ, Echo...)
    # 7 metadata fields + 2048 echo points = 2055 rows
    save_data = np.zeros((2055, records))

    #* Load each record
    for record_index in range(records):
        loaded_data = read_binary_data(full_path, record_index * RECORD_BYTES, record_format_2B_full)

        #* Write the detailed loaded data to a text file
        with open(f"{loaded_data_output_dir}/{filename}_{record_index}.txt", 'w') as file:
            for key, value in loaded_data.items():
                
                # --- Specific Formatting based on field name ---
                if key == 'FRAME_IDENTIFICATION':
                    description = read_frame_identification(value)
                    file.write(f'{key}: {value} ({description})\n')
                
                elif key == 'TIME':
                    # Value is a tuple of 6 bytes
                    bytes_val = bytes(value)
                    seconds = int.from_bytes(bytes_val[:4], 'big')
                    milliseconds = int.from_bytes(bytes_val[4:], 'big')
                    reference_time = datetime(2009, 12, 31, 16, 0, 0)
                    UTC_time = reference_time + timedelta(seconds=seconds, milliseconds=milliseconds)
                    file.write(f'{key}: {UTC_time} (Raw: {seconds}s {milliseconds}ms)\n')
                
                elif key == 'RADAR_WORKING_MODE':
                    file.write(f'{key}: {read_radar_mode(value)}\n')
                    
                elif 'GAIN_MODE' in key:
                    if 'CHANNEL_2' in key:
                        file.write(f'{key}: {read_ch2_gain_mode(value)}\n')
                    else:
                        file.write(f'{key}: {read_gain_mode(value)}\n')
                
                elif 'TIME_WINDOW' in key:
                    file.write(f'{key}: {read_time_window(value)}\n')
                    
                elif 'PULSE_REPETITION' in key:
                    file.write(f'{key}: {read_pulse_repetition(value)}\n')
                
                elif key == 'CHANNEL_AND_ANTENNA_MARK':
                    file.write(f'{key}: {channel_and_antenna_mark(value)}\n')
                
                elif key == 'ECHO_DATA':
                    file.write(f'{key}: [Array of {len(value)} floats omitted in txt]\n')
                
                else:
                    # Default write for all other fields (Voltages, Temps, Counts, Positions)
                    file.write(f'{key}: {value}\n')


        #* Save data for the B-scan summary (ECHO_DATA + Coordinates)
        # Ensure we have data before assigning to avoid errors if read failed
        if 'ECHO_DATA' in loaded_data:
            save_data[0, record_index] = loaded_data.get('VELOCITY', 0)
            save_data[1, record_index] = loaded_data.get('XPOSITION', 0)
            save_data[2, record_index] = loaded_data.get('YPOSITION', 0)
            save_data[3, record_index] = loaded_data.get('ZPOSITION', 0)
            save_data[4, record_index] = loaded_data.get('REFERENCE_POINT_XPOSITION', 0)
            save_data[5, record_index] = loaded_data.get('REFERENCE_POINT_YPOSITION', 0)
            save_data[6, record_index] = loaded_data.get('REFERENCE_POINT_ZPOSITION', 0)
            save_data[7:, record_index] = loaded_data['ECHO_DATA']

    #* Save the ECHO data matrix
    header_txt = ['VELOCITY', 'XPOSITION', 'YPOSITION', 'ZPOSITION',
                  'REFERENCE_POINT_XPOSITION', 'REFERENCE_POINT_YPOSITION', 'REFERENCE_POINT_ZPOSITION',
                  'Observed amplitude (7:)']

    np.savetxt(f"{ECHO_dir}/data_{sequence_id}.txt", save_data, header=' '.join(header_txt), comments='# ')
    
    # print(f'Finished sequence {sequence_id}')

print('All processes finished.')