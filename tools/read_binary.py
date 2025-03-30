import struct
import os
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse
from natsort import natsorted


"""
#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='read_binary.py',
    description='Read binary data files and save the data as text files',
    epilog='End of help message',
    usage='python tools/read_binary.py [path_type]'
)
parser.add_argument('path_type', choices = ['local', 'SSD'], help='Choose the path type')
args = parser.parse_args()
"""




#* Set the data folder path
# if args.path_type == 'local':
#    data_folder_path = 'LPR_2B/original_data'
# elif args.path_type == 'SSD':
data_folder_path = '/Volumes/SSD_Kanda_BUFFALO/LPR/LPR_2B/original_binary'
# else:
#     print('Invalid path type, please choose either local or SSD')
#     exit()



#* check the data folder path
if not os.path.exists(data_folder_path):
    print('Data folder does not exist')
    exit()
else:
    print('Data folder is successfully loaded')

#* check the data type whether it is LPR_2B, LPR_2A, or LPR_1
channel = 'LPR_2B'
print('Channel:', channel)


#* Define a function to read the binary data
def read_binary_data(file_path, start_byte, record_format):
    results = {}
    with open(file_path, 'rb') as file:
        for field_name, field_loc, field_len, fmt in record_format:
            file.seek(start_byte + field_loc - 1)  # Adjust for 0-based index
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

#* Define functions to read each mode
def read_frame_identification(mode_value):
    frame_identification = {
        0x146F2222: 'Channel 2 data'
    }
    return frame_identification.get(mode_value, "Unknown mode")
def read_radar_mode(mode_value):
    mode_descriptions = {
        0x00: "standby",
        0x0f: "only Channel 1 works",
        0xf0: "only Channel 2 works",
        0xff: "Channel 1 and Channel 2 work"
    }
    return mode_descriptions.get(mode_value, "Unknown mode")

def read_radar_channel_1_gain_mode(mode_value):
    mode_descriptions = {
        0x00: "variational gain",
        0x01: "fixed gain",
    }
    return mode_descriptions.get(mode_value, "Unknown mode")

def read_radar_channel_2_gain_mode(mode_value):
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

record_format_2B = [
    ("FRAME_IDENTIFICATION", 1, 4, '4B'),  # 4 UnsignedBytes
    ("TIME", 5, 6, '6B'),  # 6 UnsignedBytes
    ("VELOCITY", 11, 4, '>f'),  # 1 big-endian float
    ("XPOSITION", 15, 4, '>f'), # position of the rover based on the reference point
    ("YPOSITION", 19, 4, '>f'),
    ("ZPOSITION", 23, 4, '>f'),
    ("ATT_PITCHING", 27, 4, '>f'),
    ("ATT_ROLLING", 31, 4, '>f'),
    ("ATT_YAWING", 35, 4, '>f'),
    ("REFERENCE_POINT_XPOSITION", 39, 4, '>f'), # position of the reference point based on the landing site
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
    ("ECHO_DATA", 115, 8192, '2048f'),  # 2048 little-endian floats
    ("QUALITY_STATE", 8307, 1, 'B')
]

record_format_1 = [
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

record_format = record_format_2B if channel in ['LPR_2B', 'LPR_2A'] else record_format_1

#* Make 'loaded_data' directory and 'ECHO' directory
loaded_data_dir = os.path.join(os.path.dirname(data_folder_path), 'loaded_data')
os.makedirs(loaded_data_dir, exist_ok=True)
ECHO_dir = os.path.join(os.path.dirname(data_folder_path), 'loaded_data_echo_position')
os.makedirs(ECHO_dir, exist_ok=True)



position = []
#* Output only the echo data as txt file
for filename in tqdm(natsorted(os.listdir(data_folder_path)), desc='Total Progress'):
    full_path = os.path.join(data_folder_path, filename)
    file_size = os.path.getsize(full_path)

    #* Load only '.2B' files
    if not full_path.endswith('.2B') or filename.startswith('._'):
        continue

    #* Get the sequence ID from the filename
    sequence_id = filename.split('_')[-2]

    #* Make output directory for loaded_data of each sequence
    loaded_data_output_dir = os.path.join(loaded_data_dir, sequence_id) # Save the data in 'loaded_data/sequence_id'
    os.makedirs(loaded_data_output_dir, exist_ok=True)

    #* Process check print
    print(f'Processing {filename} ({file_size} bytes)')

    records = int(file_size / 8307) if channel in ['LPR_2B', 'LPR_2A'] else int(file_size / 32883)

    #* Make list for ECHO data obtained each sequence
    save_data = np.zeros((2055, records))

    #* Load the of each records
    for record_index in tqdm(range(records), desc=sequence_id):
        loaded_data = read_binary_data(full_path, record_index * 8307, record_format)

        #* Write the loaded data to a text file
        with open(f"{loaded_data_output_dir}/{filename}_{record_index}.txt", 'w') as file:
            for key, value in loaded_data.items():
                if key == 'FRAME_IDENTIFICATION':
                    mode_value = bytes_to_int(value)
                    mode_description = read_frame_identification(mode_value)
                    file.write(f'{key}: {mode_description}\n')
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
                    mode_description = read_radar_channel_1_gain_mode(mode_value)
                    file.write(f'{key}: {mode_description}\n')
                elif key == 'RADAR_CHANNEL_2_GAIN_MODE':
                    mode_value = bytes_to_int(value)
                    mode_description = read_radar_channel_2_gain_mode(mode_value)
                    file.write(f'{key}: {mode_description}\n')
                elif key == 'CHANNEL_AND_ANTENNA_MARK':
                    mode_value = bytes_to_int(value)
                    mode_description = channel_and_antenna_mark(mode_value)
                    file.write(f'{key}: {mode_description}\n')
                else:
                    file.write(f'{key}: {value}\n')



        """
        #* Extract the ECHO data
        echo_data = np.insert(loaded_data['ECHO_DATA'], 0, loaded_data['CHANNEL_2_RECORD_COUNT'])


        #* Insert the position data to the echo data
        echo_data = np.insert(echo_data, 1, loaded_data['VELOCITY'][0])
        echo_data = np.insert(echo_data, 2, loaded_data['XPOSITION'][0])
        echo_data = np.insert(echo_data, 3, loaded_data['YPOSITION'][0])
        echo_data = np.insert(echo_data, 4, loaded_data['ZPOSITION'][0])
        echo_data = np.insert(echo_data, 5, loaded_data['REFERENCE_POINT_XPOSITION'][0])
        echo_data = np.insert(echo_data, 6, loaded_data['REFERENCE_POINT_YPOSITION'][0])
        echo_data = np.insert(echo_data, 7, loaded_data['REFERENCE_POINT_ZPOSITION'][0])
        echo_data = np.insert(echo_data, 8, loaded_data['ATT_PITCHING'][0])
        echo_data = np.insert(echo_data, 9, loaded_data['ATT_ROLLING'][0])
        echo_data = np.insert(echo_data, 10, loaded_data['ATT_YAWING'][0])
        """


        #* Save the echo data to a list to make Bscan data
        save_data[0, record_index] = loaded_data['VELOCITY'][0]
        save_data[1, record_index] = loaded_data['XPOSITION'][0]
        save_data[2, record_index] = loaded_data['YPOSITION'][0]
        save_data[3, record_index] = loaded_data['ZPOSITION'][0]
        save_data[4, record_index] = loaded_data['REFERENCE_POINT_XPOSITION'][0]
        save_data[5, record_index] = loaded_data['REFERENCE_POINT_YPOSITION'][0]
        save_data[6, record_index] = loaded_data['REFERENCE_POINT_ZPOSITION'][0]
        save_data[7:, record_index] = loaded_data['ECHO_DATA']
        #print('ECHO data shape:', np.array(loaded_data['ECHO_DATA']).shape)


    #* Save the ECHO data as a txt file
    header = ['VELOCITY', 'XPOSITION', 'YPOSITION', 'ZPOSITION',
                'REFERENCE_POINT_XPOSITION', 'REFERENCE_POINT_YPOSITION', 'REFERENCE_POINT_ZPOSITION',
                'Observed amplitude (7:)']

    save_data[0] = save_data[0].astype(str)

    np.savetxt(f"{ECHO_dir}/data_{sequence_id}.txt", save_data, header=' '.join(header), comments='') # ヘッダーを追加
    print('Finished saving ECHO data')
    print('ECHO shape:', save_data.shape)
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