import struct
import os
import numpy as np

data_path = 'LPR_2B/CE4_GRAS_LPR-2B_SCI_N_20231216075001_20231217065500_0316_A.2B'
data_folder_path = 'LPR_2B'


#* Define a function to read the binary data
def read_binary_data(file_path):
    # Define the structure of a single record based on the provided metadata
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
        ("ECHO_DATA", 115, 8192, '2048f'),  # 2048 little-endian floats
        ("QUALITY_STATE", 8307, 1, 'B')
    ]

    # Open the file and read the specified fields
    with open(file_path, 'rb') as file:
        results = {}
        for field_name, start_byte, length, fmt in record_format:
            file.seek(start_byte - 1)  # Adjust for 0-based index
            data = file.read(length)
            if fmt.endswith('s'):  # Handle string type specially
                results[field_name] = data.decode().strip()
            else:
                results[field_name] = struct.unpack(fmt, data)
        return results



#* Load the binary data
output_dir_path = os.path.dirname(data_path)
Ascans = []
#* Output only the echo data as txt file
for filename in os.listdir(data_folder_path):
    full_path = os.path.join(data_folder_path, filename)
    # load only '.2B' files
    if full_path.endswith('.2B') == False:
        continue

    Ascan_output_dir = os.path.join(data_folder_path, 'Ascan')
    if not os.path.exists(Ascan_output_dir):
        os.makedirs(Ascan_output_dir)
    loaded_data = read_binary_data(full_path)
    np.savetxt(Ascan_output_dir + '/' + filename + '_echo_data.txt', loaded_data['ECHO_DATA'])

    Ascans.append(loaded_data['ECHO_DATA'])


Ascans = np.array(Ascans).T
print(Ascans.shape)

output_name = os.path.basename(data_folder_path) + '_echo_data.txt'
output_path = os.path.join('Ascans', output_name)
np.savetxt(output_path, Ascans)