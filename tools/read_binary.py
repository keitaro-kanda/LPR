import struct
import os
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt

data_path = 'LPR_2B/CE4_GRAS_LPR-2B_SCI_N_20231216075001_20231217065500_0316_A.2B'
data_folder_path = 'LPR_2B/original_data'
#data_folder_path = '/Volumes/SSD_kanda/LPR/LPR_2B/original_data'

#* check the data type whther it is LPR_2B, LPR_2A, or LPR_1
channel = 'LPR_2B'
print('Channnel:', channel)


#* Define a function to read the binary data
def read_binary_data(file_path, start_byte):
    # Define the structure of a single record based on the provided metadata
    if channel == 'LPR_2B' or channel == 'LPR_2A':
        record_format = [
            ("FRAME_IDENTIFICATION", start_byte+1, 4, '4B'),  # 4 UnsignedBytes
            ("TIME", start_byte+5, 6, '6B'),  # 6 UnsignedBytes
            ("VELOCITY", start_byte+11, 4, '>f'),  # 1 big-endian float
            ("XPOSITION", start_byte+15, 4, '>f'),
            ("YPOSITION", start_byte+19, 4, '>f'),
            ("ZPOSITION", start_byte+23, 4, '>f'),
            ("ATT_PITCHING", start_byte+27, 4, '>f'),
            ("ATT_ROLLING", start_byte+31, 4, '>f'),
            ("ATT_YAWING", start_byte+35, 4, '>f'),
            ("REFERENCE_POINT_XPOSITION", start_byte+39, 4, '>f'),
            ("REFERENCE_POINT_YPOSITION", start_byte+43, 4, '>f'),
            ("REFERENCE_POINT_ZPOSITION", start_byte+47, 4, '>f'),
            ("REFERENCE_POINT_ATT_PITCHING", start_byte+51, 4, '>f'),
            ("REFERENCE_POINT_ATT_ROLLING", start_byte+55, 4, '>f'),
            ("REFERENCE_POINT_ATT_YAWING", start_byte+59, 4, '>f'),
            ("DataParameter1", start_byte+63, 11, '11B'),
            ("RADAR_WORKING_MODE", start_byte+74, 1, 'B'),
            ("RADAR_CHANNEL_1_GAIN_MODE", start_byte+75, 1, 'B'),
            ("RADAR_CHANNEL_1_GAIN_VALUE", start_byte+76, 1, 'B'),
            ("RADAR_CHANNEL_2_GAIN_MODE", start_byte+77, 1, 'B'),
            ("RADAR_CHANNEL_2_GAIN_VALUE_1", start_byte+78, 1, 'B'),
            ("RADAR_CHANNEL_2_GAIN_VALUE_2", start_byte+79, 1, 'B'),
            ("DataParameter2", start_byte+80, 28, '28B'),
            ("VALID_DATA_LENGTH", start_byte+108, 2, '<H'),  # 2 bytes, little-endian unsigned short
            ("CHANNEL_1_RECORD_COUNT", start_byte+110, 2, '<H'),
            ("CHANNEL_2_RECORD_COUNT", start_byte+112, 2, '<H'),
            ("CHANNEL_AND_ANTENNA_MARK", start_byte+114, 1, 'B'),
            ("ECHO_DATA", start_byte+115, 8192, '2048f'),  # 2048 little-endian floats
            ("QUALITY_STATE", start_byte+8307, 1, 'B')
        ]
    elif channel == 'LPR_1':
        record_format = [
            ("FRAME_IDENTIFICATION", 1, 4, '4B'),  # 4 UnsignedBytes
            ("TIME", 5, 6, '6B'),  # 6 UnsignedBytes
            ("VELOCITY", 11, 4, '>f'),  # 1 big-endian float
            ("XPOSITION", 15, 4, '>f'),
            ("YPOSITION", 19, 4, '>f'),
            ("ZPOSITION", 23, 4, '>f'),
            ("ATT_PITCHING", 27, 4, '>f'),
            ("ATT_ROLLING", 31, 4, '>f'),
            ("ATT_YAWING", 35, 4, '>f'),
            ("REFERENCE_POINT_XPOSITION", 39, 4, '>f'),
            ("REFERENCE_POINT_YPOSITION", 43, 4, '>f'),
            ("REFERENCE_POINT_ZPOSITION", 47, 4, '>f'),
            ("REFERENCE_POINT_ATT_PITCHING", 51, 4, '>f'),
            ("REFERENCE_POINT_ATT_ROLLING", 55, 4, '>f'),
            ("REFERENCE_POINT_ATT_YAWING", 59, 4, '>f'),
            ("DataParameter1", 63, 11, '11B'),
            ("RADAR_WORKING_MODE", 74, 1, 'B'),
            ("RADAR_CHANNEL_1_GAIN_MODE", 75, 1, 'B'),
            ("RADAR_CHANNEL_1_GAIN_VALUE", 76, 1, 'B'),
            ("RADAR_CHANNEL_2_GAIN_MODE", 77, 1, 'B'),
            ("RADAR_CHANNEL_2_GAIN_VALUE_1", 78, 1, 'B'),
            ("RADAR_CHANNEL_2_GAIN_VALUE_2", 79, 1, 'B'),
            ("DataParameter2", 80, 28, '28B'),
            ("VALID_DATA_LENGTH", 108, 2, '<H'),  # 2 bytes, little-endian unsigned short
            ("CHANNEL_1_RECORD_COUNT", 110, 2, '<H'),
            ("CHANNEL_2_RECORD_COUNT", 112, 2, '<H'),
            ("CHANNEL_AND_ANTENNA_MARK", 114, 1, 'B'),
            ("ECHO_DATA", 115, 32768, '8192f'),  # 8192 little-endian floats
            ("QUALITY_STATE", 32883, 1, 'B')
        ]
    else:
        print("Invalid data type")
        return None


    #* Open the file and read the specified fields
    with open(file_path, 'rb') as file:
        results = {}
        for field_name, field_loc, field_len, fmt in record_format:
            file.seek(field_loc - 1)  # Adjust for 0-based index
            data = file.read(field_len)
            if len(data) != field_len:
                print(f"Warning: Expected {field_len} bytes for {field_name}, but got {len(data)} bytes.")
                continue
            try:
                if fmt.endswith('s'):  # Handle string type specially
                    results[field_name] = data.decode().strip()
                else:
                    results[field_name] = struct.unpack(fmt, data)
            except struct.error as e:
                print(f"Error unpacking field {field_name}: {e}")
                continue
        return results


#* Define a function to convert bytes to integer
def bytes_to_int(bytes):
    return int.from_bytes(bytes, byteorder='big')
#* Define a functions to read each modes
def read_radar_mode(mode_value):
    mode_descriptions = {
        0x00: "standby",
        0x0f: "only Channel 1 works",
        0xf0: "only Channel 2 works",
        0xff: "Channel 1 and Channel 2 work"
    }
    return mode_descriptions.get(mode_value, "Unknown mode")
def read_radar_channerl_1_gain_mode(mode_value):
    mode_descriptions = {
        0x00: "variational gain",
        0x01: "fixed gain",
    }
    return mode_descriptions.get(mode_value, "Unknown mode")
def read_radar_channerl_2_gain_mode(mode_value):
    mode_descriptions = {
        0x00: "Antenna A variational gain, Antenna B variational gain",
        0x0f: "Antenna A variational gain, Antenna B fixed gain",
        0xf0: "Antenna A fixed gain, Antenna B variational gain",
        0xff: "Antenna A fixed gain, Antenna B fixed gain"
    }
    return mode_descriptions.get(mode_value, "Unknown mode")
def channel_and_antenna_mark(mode_value):
    mode_descriptions = {
        0x11: "Channel 1",
        0x2A: "Channel 2, Antenna B",
        0x2B: "Channel 2, Antenna B"
    }
    return mode_descriptions.get(mode_value, "Unknown mode")



#* Make 'loaded_data' directry and 'ECHO' directry
loaded_data_dir = os.path.join(os.path.dirname(data_folder_path), 'loaded_data')
if not os.path.exists(loaded_data_dir):
    os.makedirs(loaded_data_dir)
ECHO_dir = os.path.join(os.path.dirname(data_folder_path), 'ECHO')
if not os.path.exists(ECHO_dir):
    os.mkdir(ECHO_dir)



#Ascans = []
position = []
#* Output only the echo data as txt file
for filename in os.listdir(data_folder_path):
    full_path = os.path.join(data_folder_path, filename)
    file_size = os.path.getsize(full_path)


    #* load only '.2B' files
    if full_path.endswith('.2B') == False:
        continue
    #* ファイル名が'._'で始まるファイルは無視する
    if filename.startswith('._'):
        continue

    #* Get the sequence ID from the filename
    sequence_id = filename.split('_')[-2]


    #* make output directry for loaded_data of each sequence
    loaded_data_output_dir = os.path.join(loaded_data_dir, sequence_id)
    if not os.path.exists(loaded_data_output_dir):
        os.makedirs(loaded_data_output_dir)
    

    #* make list for ECHO data obtained each sequence
    ECHO = []


    #* process check print
    print(f'Processing {filename} ({file_size} bytes)')

    if channel == 'LPR_2B' or channel == 'LPR_2A':
        records = int(file_size / 8307)
    elif channel == 'LPR_1':
        records = int(file_size / 32883)


    #* Load the of each records
    for record_index in tqdm(range(records), desc = sequence_id):
        loaded_data = read_binary_data(full_path, record_index * 8307)

        #* write the loaded data to a text file
        with open(loaded_data_output_dir + '/' + filename + '_' + str(record_index) + '.txt', 'w') as file:
            for key, value in loaded_data.items():
                if key == 'FRAME_IDENTIFICATION':
                    file.write(f'{key}: {"Channel 2 data"}\n')
                elif key == 'TIME':
                    seconds = bytes_to_int(value[:4])
                    milliseconds = bytes_to_int(value[4:])
                    reference_time = datetime(2009, 12, 31, 16, 0, 0)
                    UTC_time = reference_time + timedelta(seconds=seconds, milliseconds=milliseconds)
                    file.write(f'{key}: {UTC_time}\n')
                elif key == 'RADAR_WORKING_MODE':
                    mode_value = bytes_to_int(value)
                    mode_description = read_radar_mode(mode_value)
                    file.write(f'{key}: {mode_description}\n')
                elif key == 'RADAR_CHANNEL_1_GAIN_MODE':
                    mode_value = bytes_to_int(value)
                    mode_description = read_radar_channerl_1_gain_mode(mode_value)
                    file.write(f'{key}: {mode_description}\n')
                elif key == 'RADAR_CHANNEL_2_GAIN_MODE':
                    mode_value = bytes_to_int(value)
                    mode_description = read_radar_channerl_2_gain_mode(mode_value)
                    file.write(f'{key}: {mode_description}\n')
                elif key == 'CHANNEL_AND_ANTENNA_MARK':
                    mode_value = bytes_to_int(value)
                    mode_description = channel_and_antenna_mark(mode_value)
                    file.write(f'{key}: {mode_description}\n')
                else:
                    file.write(f'{key}: {value}\n')


        #* 前のファイルと同じpositionのデータは無視する
        if record_index>0 and position[-1] == [loaded_data['XPOSITION'], loaded_data['YPOSITION'], loaded_data['ZPOSITION']]:
            continue
        position.append([loaded_data['XPOSITION'], loaded_data['YPOSITION'], loaded_data['ZPOSITION']])

        #* Save the echo data to a list to make Bscan data
        #* record_indexは最大4桁の整数表記にする
        echo_data = np.insert(loaded_data['ECHO_DATA'], 0, record_index)
        ECHO.append(echo_data)

    #* Save the position data
    #position = np.array(position)
    #print('Position shape:', position)
    #np.savetxt(loaded_data_output_dir + '/position.txt', position)

    #* Save the ECHO data as a txt file
    ECHO = np.array(ECHO).T
    print(ECHO.shape)
    ECHO = ECHO[:, ECHO[0].argsort()]
    np.savetxt(ECHO_dir + '/ECHO_' + sequence_id + '.txt', ECHO)
    print('Finished saving ECHO data')
    print('  ')


"""
#* Save all Ascans as a single Bscan data file
Ascans = np.array(Ascans).T
print("Ascans shape:", Ascans.shape)


#* sort the Ascans by sequence ID
Ascans = Ascans[:, Ascans[0].argsort()]


output_name = os.path.basename(data_path) + '_echo_data.txt'
output_path = os.path.join('Ascans', output_name)
np.savetxt(output_path, Ascans, fmt='%s', delimiter=' ')
print(f'All Ascans are saved as {output_path}')
"""