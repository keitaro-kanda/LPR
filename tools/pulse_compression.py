import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
# import argparse
from tqdm import tqdm


#* Get input parameters
print('データファイルのパスを入力してください:')
data_path = input().strip()
if not os.path.exists(data_path):
    print('エラー: 指定されたファイルが存在しません')
    exit(1)

print('フィルタの種類を選択してください (matched, wiener):')
func_type = input().strip()
if func_type not in ['matched', 'wiener']:
    print('エラー: 無効なフィルタ種類です')
    exit(1)

#* Data path
if func_type == 'matched':
    output_dir = os.path.join(os.path.dirname(data_path), 'Matched_filter')
elif func_type == 'wiener':
    output_dir = os.path.join(os.path.dirname(data_path), 'Wiener_filter')
else:
    raise ValueError('Invalid function type')
os.makedirs(output_dir, exist_ok=True)


#* Load data
print('Loading data...')
data = np.loadtxt(data_path, delimiter=' ')


#* Parameters
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]



#* Define the function to calculate the gaussian wave
def gaussian(t_array, mu, sigma):
        a = 1 / (sigma * np.sqrt(2 * np.pi))
        b = np.exp(-0.5 * (((t_array - mu) / sigma) ** 2))
        sig = a * b

        fft = np.fft.fft(sig)
        freq = np.fft.fftfreq(len(t_array), d=sample_interval)
        return sig, fft, freq



#* Define the function to calculate the matched filter
def mached_filter(Ascan, reference_sig):
    reference_sig = np.concatenate([np.flip(reference_sig), np.zeros(len(Ascan) - len(reference_sig))])
    fft_refer = np.fft.fft(np.conj(reference_sig))
    fft_data = np.fft.fft(Ascan)
    conv = np.fft.ifft(fft_refer * fft_data)
    return np.real(conv)



#* Define the function to winer filter
def wiener_filter(Ascan, beta, reference_sig):
    reference_sig = np.concatenate([np.flip(reference_sig), np.zeros(len(Ascan) - len(reference_sig))])
    fft_refer = np.fft.fft(np.conj(reference_sig))
    fft_data = np.fft.fft(Ascan)
    wiener = np.fft.ifft(fft_data * np.conj(fft_refer) / (np.abs(fft_refer) ** 2 + beta**2))
    return np.real(wiener)



#* Make the reference signal
t = np.arange(0, 20e-9, sample_interval)
mu = 3e-9
sigma = 0.3e-9
refer, fft, freq = gaussian(t, mu, sigma)
refer = np.abs(refer) / np.max(np.abs(refer))
fft = 10 * np.log10(np.abs(fft) / np.max(np.abs(fft)))




#* Apply the matched filter to the data
data_compressed = np.zeros(data.shape)
for i in tqdm(range(data.shape[1]), desc='Calculating pulse compression'):
    if func_type == 'matched':
        data_compressed[:, i] = mached_filter(data[:, i], refer)
    elif func_type == 'wiener':
        wiener_beta = 1e-2
        data_compressed[:, i] = wiener_filter(data[:, i], wiener_beta, refer)


if func_type == 'matched':
    np.savetxt(os.path.join(output_dir, 'matched_filter.txt'), data_compressed, delimiter=' ')
elif func_type == 'wiener':
    output_dir = os.path.join(output_dir, f'beta_{wiener_beta}')
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, 'wiener_filter.txt'), data_compressed, delimiter=' ')


#data_matched = 10 * np.log10(data_compressed / np.max(data_compressed))


#* Plot
print('Plotting...')
plt.figure(figsize=(18, 6), facecolor='w', edgecolor='w', tight_layout=True)
im = plt.imshow(data_compressed, cmap='seismic', aspect='auto',
                extent=[0, data_compressed.shape[1] * trace_interval,
                data_compressed.shape[0] * sample_interval / 1e-9, 0],
                vmin=-np.max(np.abs(data_compressed))/5, vmax=np.max(np.abs(data_compressed))/5
                )

plt.xlabel('x [m]', fontsize=20)
plt.ylabel('Time [ns]', fontsize=20)
plt.tick_params(labelsize=18)


delvider = axgrid1.make_axes_locatable(plt.gca())
cax = delvider.append_axes('right', size='5%', pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Amplitude', fontsize=20)
cbar.ax.tick_params(labelsize=18)


if func_type == 'matched':
    plt.savefig(os.path.join(output_dir, 'matched_filter.png'), format='png', dpi=120)
    plt.savefig(os.path.join(output_dir, 'matched_filter.pdf'), format='pdf', dpi=600)
elif func_type == 'wiener':
    plt.savefig(os.path.join(output_dir, 'wiener_filter.png'), format='png', dpi=120)
    plt.savefig(os.path.join(output_dir, 'wiener_filter.pdf'), format='pdf', dpi=600)
plt.show()


#* Plot and save the reference signal
fig, ax = plt.subplots(1, 2, figsize=(18, 10), facecolor='w', edgecolor='w', tight_layout=True)

ax[0].plot(t/1e-9, refer[:len(t)])
ax[0].set_xlabel('Time [ns]', fontsize=20)
ax[0].set_ylabel('Amplitude', fontsize=20)
ax[0].tick_params(labelsize=18)
ax[0].grid()
ax[0].text(10.0, 0.8, f'$\mu$ = {mu/1e-9} ns\n$\sigma$ = {sigma/1e-9} ns', fontsize=20)

ax[1].plot(freq[1: len(freq)//2]/1e6, fft[1: len(fft)//2])
ax[1].set_xlabel('Frequency [MHz]', fontsize=20)
ax[1].set_ylabel('Amplitude [dB]', fontsize=20)
ax[1].tick_params(labelsize=18)
ax[1].grid()

plt.savefig(os.path.join(output_dir, 'reference_signal.png'), format='png', dpi=120)
plt.savefig(os.path.join(output_dir, 'reference_signal.pdf'), format='pdf', dpi=600)
plt.show()