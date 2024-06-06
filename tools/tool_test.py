import numpy as np
import struct



# read binarty, binary is '30b2 c438'
data = b'\x30\xb2\xc4\x38'
read = struct.unpack('!f', data)
print(read)
# メートルを緯度に変換，月の半径は1737.4kmとする
lat = np.arcsin(read[0]/1737.4)
print(lat*180/np.pi)

# 16進数'30b2 c438'をfloatに変換
data = 0xaf2286c3
read = struct.unpack('>f', struct.pack('>I', data))[0]
print(read)