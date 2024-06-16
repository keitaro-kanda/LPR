import numpy as np
import matplotlib.pyplot as plt
import struct
import datetime
import tkinter as tk
from tkinter import filedialog

def read_lpr02e(pname, fname):
    with open(f"{pname}/{fname}", 'rb') as f:
        dchar = f.read()

    if 'LPR-1' in fname:
        ch = 1
    elif 'LPR-2' in fname:
        ch = 2
    else:
        print(f"{fname} is not a correct LPR data file name")
        return 0, []

    if fname.endswith('2A') or fname.endswith('2a'):
        dlevel = 210
    elif fname.endswith('2B') or fname.endswith('2b'):
        dlevel = 211
    elif fname.endswith('2C') or fname.endswith('2c'):
        dlevel = 212
    else:
        print(f"{fname} is not a correct 2 level LPR data file name!")
        return 0, []

    dchar = np.frombuffer(dchar, dtype=np.uint8)

    if ch == 1:
        a1 = np.where(dchar == 0x14)[0]
        a1 = a1[a1 < (len(dchar) - 4)]
        a2 = a1[dchar[a1 + 1] == 0x6F]
        a3 = a2[dchar[a2 + 2] == 0x11]
        trace_bpos = a3[dchar[a3 + 3] == 0x11]
    else:
        a1 = np.where(dchar == 0x14)[0]
        a1 = a1[a1 < (len(dchar) - 4)]
        a2 = a1[dchar[a1 + 1] == 0x6F]
        a3 = a2[dchar[a2 + 2] == 0x22]
        trace_bpos = a3[dchar[a3 + 3] == 0x22]

    print(f"trace_bpos: {trace_bpos}")  # デバッグ用に追加

    # ファイルサイズを取得
    file_size = len(dchar)
    print(f"File size: {file_size} bytes")  # デバッグ用に追加

    traces = []
    for k in trace_bpos:
        if k - 1 >= file_size or k - 1 < 0:
            print(f"Skipping invalid position {k - 1}")
            continue

        trace = {}
        with open(f"{pname}/{fname}", 'rb') as f:
            f.seek(k - 1)
            trace['tag'] = f.read(4).hex()
            trace['tstamp_sec'] = struct.unpack('I', f.read(4))[0]
            trace['tstamp_msec'] = struct.unpack('H', f.read(2))[0]
            timestamp = trace['tstamp_sec'] + trace['tstamp_msec'] / 1000
            trace['tstr'] = datetime.datetime(2010, 1, 1) + datetime.timedelta(seconds=timestamp)
            trace['velocity'] = struct.unpack('f', f.read(4))[0]
            trace['position_x'] = struct.unpack('f', f.read(4))[0]
            trace['position_y'] = struct.unpack('f', f.read(4))[0]
            trace['position_z'] = struct.unpack('f', f.read(4))[0]
            trace['pose_u'] = struct.unpack('f', f.read(4))[0]
            trace['pose_v'] = struct.unpack('f', f.read(4))[0]
            trace['pose_w'] = struct.unpack('f', f.read(4))[0]
            
            if dlevel == 211 or dlevel == 212:
                trace['position_xref'] = struct.unpack('f', f.read(4))[0]
                trace['position_yref'] = struct.unpack('f', f.read(4))[0]
                trace['position_zref'] = struct.unpack('f', f.read(4))[0]
                trace['pose_uref'] = struct.unpack('f', f.read(4))[0]
                trace['pose_vref'] = struct.unpack('f', f.read(4))[0]
                trace['pose_wref'] = struct.unpack('f', f.read(4))[0]
            
            trace['data_block_num'] = struct.unpack('H', f.read(2))[0]
            # Add other fields similarly based on MATLAB code
            trace['nbyte_trace'] = struct.unpack('H', f.read(2))[0]
            trace['tracenum1'] = struct.unpack('H', f.read(2))[0]
            trace['tracenum2'] = struct.unpack('H', f.read(2))[0]
            channel = struct.unpack('B', f.read(1))[0]
            trace['channel'] = channel
            
            try:
                data_length = trace['nbyte_trace']
                if channel == 0x11:
                    trace['nsamp'] = data_length // 4
                    trace['data'] = np.fromfile(f, dtype=np.float32, count=8192)
                elif channel == 0x2A:
                    trace['nsamp'] = data_length // 8
                    trace['data'] = np.fromfile(f, dtype=np.float32, count=2048)
                elif channel == 0x2B:
                    trace['nsamp'] = data_length // 8
                    trace['data'] = np.fromfile(f, dtype=np.float32, count=2048)
                    #trace['data'] = struct.unpack('f' * trace['nsamp'], f.read(trace['nsamp'] * 4))
                trace['qs'] = struct.unpack('B', f.read(1))[0]
                traces.append(trace)
                print(f"Read trace at position {k - 1} with {len(trace['data']) if 'data' in trace else 'no'} samples.")  # デバッグ用に追加

            except Exception as e:
                print(f"Failed to read data at position {k - 1}: {e}")

    ntr = len(traces)
    return ntr, traces


def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("LPR Files", "*.2B *.2A *.2C")])
    if not file_path:
        print("No file selected.")
        return
    
    pname, fname = file_path.rsplit('/', 1)
    ntrace, DAT = read_lpr02e(pname, fname)
    
    if ntrace == 0:
        print("Failed to read data.")
        return

    # デバッグ用に、読み込んだデータの形状を確認
    data_shapes = [d['data'].shape for d in DAT if 'data' in d]
    print(f"Data shapes: {data_shapes}")

    sum2B = np.array([d['data'] for d in DAT if 'data' in d])
    if sum2B.ndim == 1:
        sum2B = sum2B[np.newaxis, :]  # 1次元の場合に2次元に変換

    nsample, ntrace = sum2B.shape

    dt = 0.3125
    t = np.arange(nsample) * dt
    x = np.arange(ntrace)

    sum2B_gain = sum2B * np.arange(1, nsample + 1)[:, None]

    plt.figure(figsize=(10, 6))
    plt.imshow(sum2B, aspect='auto', extent=[x.min(), x.max(), t.max(), t.min()], cmap='gray', vmin=-300, vmax=500)
    plt.colorbar()
    plt.ylabel('Time (ns)')
    plt.xlabel('Trace')
    plt.title('CE4 CH2B Data')
    plt.show()

if __name__ == "__main__":
    main()


