import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
from tqdm import tqdm
import gc
from scipy.signal import hilbert



#* Get input parameters
print('データファイルのパスを入力してください:')
data_path = input().strip()
if not os.path.exists(data_path):
    print('エラー: 指定されたファイルが存在しません')
    exit(1)

print('peakデータファイルのパスを入力してください:')
peak_path = input().strip()
if not os.path.exists(peak_path):
    print('エラー: 指定されたファイルが存在しません')
    exit(1)

#* Define the output directory
output_dir = os.path.join(os.path.dirname(peak_path), 'detect_hyperbola')
os.makedirs(output_dir, exist_ok=True)

output_dir_trim_png = os.path.join(output_dir, 'trim/png')
os.makedirs(output_dir_trim_png, exist_ok=True)
output_dir_trim_pdf = os.path.join(output_dir, 'trim/pdf')
os.makedirs(output_dir_trim_pdf, exist_ok=True)


#* Parameters
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]



#* Load data
print('Loading data...')
data = np.loadtxt(data_path, delimiter=' ')
time = np.arange(data.shape[0]) * sample_interval / 1e-9 # [ns]
print('Data shape:', data.shape)
print('Finished loading data.')
print(' ')

print('Loading peak data...')
peak_data = np.loadtxt(peak_path, delimiter=' ')
print('Peak data shape:', peak_data.shape)
print('Finished loading peak data.')
print(' ')


#* 