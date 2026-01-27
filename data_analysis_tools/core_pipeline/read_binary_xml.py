#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import struct
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from tqdm import tqdm
from natsort import natsorted


channel_name = input('Select channel name (1, 2A, 2B): ').strip()
if channel_name not in ['1', '2A', '2B']:
    print('Invalid channel name. Please enter 1, 2A, or 2B.')
    sys.exit(1)

# --- パスの設定 ---
if channel_name == '1':
    data_folder = '/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_1/original_binary'
    label_folder = data_folder
elif channel_name == '2A':  # 2A チャンネル
    data_folder = '/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2A/original_binary'
    label_folder = data_folder  # ラベルフォルダは同じ
elif channel_name == '2B':  # 2B チャンネル
    data_folder = '/Volumes/SSD_Kanda_SAMSUNG/LPR/LPR_2B/original_binary'
    label_folder = data_folder

for path in (data_folder, label_folder):
    if not os.path.isdir(path):
        print(f"フォルダが見つかりません: {path}")
        sys.exit(1)

print(f"データフォルダ: {data_folder}")
print(f"ラベルフォルダ: {label_folder}")

# --- XML ラベル解析 ---
def parse_label(label_path):
    ns = {'p':'http://pds.nasa.gov/pds4/pds/v1'}
    tree = ET.parse(label_path)
    root = tree.getroot()
    tb = root.find('.//p:Table_Binary', ns)
    rec = tb.find('p:Record_Binary', ns)
    rec_len = int(rec.find('p:record_length', ns).text)
    fields = []
    echo = None
    for fb in rec.findall('p:Field_Binary', ns):
        name = fb.find('p:name', ns).text
        off  = int(fb.find('p:field_location', ns).text) - 1
        ln   = int(fb.find('p:field_length', ns).text)
        dt   = fb.find('p:data_type', ns).text
        # データ型判定
        if dt == 'UnsignedByte':
            fmt = f'{ln}B'
        elif dt == 'UnsignedLSB2':
            fmt = f'<{ln//2}H'
        elif dt == 'IEEE754MSBSingle':
            fmt = '>f'; ln = 4
        elif dt == 'IEEE754LSBSingle':
            fmt = '<f'; ln = 4
        else:
            fmt = f'{ln}B'
        # REFERENCE_POINT 系はファイル上リトルエンディアンで格納されている
        if name.startswith('REFERENCE_POINT'):
            fmt = '<f'
            ln  = 4
        fields.append((name, off, ln, fmt))
    gf = rec.find('p:Group_Field_Binary', ns)
    if gf is not None:
        name  = gf.find('p:name', ns).text
        off   = int(gf.find('p:group_location', ns).text) - 1
        reps  = int(gf.find('p:repetitions', ns).text)
        fmt   = f'<{reps}f'
        ln    = reps * 4
        echo  = (name, off, ln, fmt, reps)
    return rec_len, fields, echo

# --- 出力ディレクトリ ---
base        = os.path.dirname(data_folder)
loaded_dir  = os.path.join(base, 'loaded_data')
echo_dir    = os.path.join(base, 'loaded_data_echo_position')
os.makedirs(loaded_dir, exist_ok=True)
os.makedirs(echo_dir, exist_ok=True)

# --- ヘルパー ---
def bytes_to_int(b):
    raw = bytes(b) if isinstance(b, (list, tuple)) else b
    return int.from_bytes(raw, byteorder='big')

def read_record(f, start, defs):
    rec = {}
    for name, off, ln, fmt in defs:
        f.seek(start + off)
        buf = f.read(ln)
        if len(buf) != ln:
            rec[name] = ()
        else:
            try:
                rec[name] = struct.unpack(fmt, buf)
            except:
                rec[name] = ()
    return rec

# --- メイン処理 ---
for fname in tqdm(natsorted(os.listdir(data_folder)), desc='ファイル処理'):
    if not fname.endswith('.2B') or fname.startswith('._'):
        continue
    bin_path   = os.path.join(data_folder, fname)
    label_path = os.path.splitext(bin_path)[0] + '.2BL'
    if not os.path.isfile(label_path):
        print(f"ラベルが見つかりません: {label_path}")
        continue
    rec_len, fields, echo = parse_label(label_path)
    echo_name, echo_off, echo_len, echo_fmt, echo_count = echo
    size = os.path.getsize(bin_path)
    nrec = size // rec_len
    seq_id = fname.split('_')[-2]

    seq_dir = os.path.join(loaded_dir, seq_id)
    os.makedirs(seq_dir, exist_ok=True)

    # 必要な8種類の項目のみ
    header = [
        'VELOCITY',
        'XPOSITION',
        'YPOSITION',
        'ZPOSITION',
        'REFERENCE_POINT_XPOSITION',
        'REFERENCE_POINT_YPOSITION',
        'REFERENCE_POINT_ZPOSITION'
    ]
    # 行列の形状: 7 行の各パラメータ + echo_count 行の観測データ
    mat = np.zeros((len(header) + echo_count, nrec), float)

    with open(bin_path, 'rb') as f:
        for i in range(nrec):
            rec = read_record(f, i*rec_len, fields)
            out_txt = os.path.join(seq_dir, f"{fname}_{i}.txt")
            with open(out_txt, 'w') as tf:
                for name in header:
                    v = rec.get(name, ())
                    if name == 'FRAME_IDENTIFICATION':
                        tf.write(f"{name}: {hex(bytes_to_int(v))}\n")
                    elif name == 'TIME':
                        sec = bytes_to_int(v[:4]); ms = bytes_to_int(v[4:])
                        dt  = datetime(2009,12,31,16,0,0) + timedelta(seconds=sec, milliseconds=ms)
                        tf.write(f"{name}: {dt.isoformat()}\n")
                    else:
                        tf.write(f"{name}: {v}\n")
            # matrix fill
            for idx_h, name in enumerate(header):
                mat[idx_h, i] = rec.get(name, (0,))[0]
            # echo read directly
            f.seek(i * rec_len + echo_off)
            buf = f.read(echo_len)
            echo_vals = struct.unpack(echo_fmt, buf)
            mat[len(header):, i] = echo_vals

    out_echo = os.path.join(echo_dir, f"data_{seq_id}.txt")
    np.savetxt(out_echo, mat, header=' '.join(header + [echo_name]), comments='')
    print(f"保存: {out_echo} shape={mat.shape}")


