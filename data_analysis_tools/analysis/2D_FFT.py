#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lpr_bscan_2dfft.py
==================

CE-4/LPR の B-scan（2 次元の txt。各列が 1 本の A-scan）から、いくつかの範囲
（解析窓）を切り出し、2 次元 FFT（空間スペクトル）を求める。解析窓はグループに
分ける（既定：最上部レゴリス層の 'regolith' と、比較用に反射の強い 'strong'）。
シミュレーションの大きさ（測線の長さ）を決めるため、水平方向の区間長 Ls を
何通りか変えて、区間ごとのスペクトルとその平均を出力する。

手順
    1. データを読み込む（行 = 時間サンプル、列 = トレース）
    2. 解析窓（WINDOWS）ごとに、時間 TIME_LEN_NS 分を切り出す
    3. 解析窓を長さ Ls の区間に分け、区間ごとに 2 次元 FFT → パワースペクトル密度
    4. 同じ Ls の区間を、グループごとに平均する（解析窓ごとの平均も記録）

軸
    k_x : 水平波数 [rad/m]（トレース間隔 DX_M から）
    k_y : 鉛直波数 [rad/m]。時間を深さ y = v t / 2（v = c / sqrt(EPS_R)）に直して
          FFT した波数。媒質の鉛直スペクトルの波数（ブラッグ波数 2k）と同じ物差し。
          k_y = 4 pi f sqrt(EPS_R) / c なので、周波数 f と 1 対 1 に対応する。

スペクトルの正規化（区間長が違っても重ねて比べられるように密度で表す）
    S(k_x, k_y) = |FFT(w * d)|^2 dx dy / ((2 pi)^2 N_x N_y <w^2>)
    （∫∫ S dk_x dk_y = 区間内の振幅の分散。振幅の単位はデータのまま）
    パネル 2 の S_x(k_x) は、S を k_y について LPR の帯域（BAND_MHZ）の中だけ積分した
    片側スペクトル。パネル 3 の S_y(k_y) は、S を k_x の全範囲で積分した片側スペクトル。

分解能とナイキスト波数（パワーの図に縦線と数値で示す）
    分解能（波数の刻み） Δk_x = 2 pi / (N_x dx)、Δk_y = 2 pi / (N_y dy)
    ナイキスト波数       k_N,x = pi / dx、      k_N,y = pi / dy
    dy = v DT / 2 は 1 サンプルあたりの深さ（往復なので 1/2）。
    既定値では Δk_y = 0.726 rad/m（100 ns = 8.65 m）、k_N,y = 116.2 rad/m（1.6 GHz）、
    k_N,x = 87.3 rad/m。窓関数をかけると実効的な分解能は Δk より 1.5-2 倍ほど粗い。

出力（データファイルの親ディレクトリに 2D_FFT/ を作る）
    2D_FFT/
        overview_windows.png       B-scan 全体と解析窓
        mean_<グループ>_L<Ls>m.png        グループ・区間長ごとの平均（3 パネル）
        mean_all_L<Ls>m.png               全解析窓（全グループ）をまとめた平均（3 パネル）
        compare_lengths_<グループ>.png    区間長による違い（S_x, S_y の平均を重ねる。all もあり）
        compare_groups_L<Ls>m.png         グループの違い（同じ区間長で重ねる）
        spectra_mean.npz           平均スペクトルの数値
        run_info.json              設定と区間の一覧
        L<Ls>m/<窓>_x<始点>-<終点>m.png   区間ごとの結果（B-scan + 3 パネル）
        _cache_<データ名>.npy      読み込みの高速化用（txt より新しければ再利用）
"""

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# =============================================================================
# 設定  [EDIT HERE]
# =============================================================================
# --- データの形式（要確認：読み込み時に表示する行数・列数と、図の時間・距離の範囲が合うか）
DX_M = 0.036            # トレース間隔 [m]
DT_NS = 0.3125          # 時間サンプル間隔 [ns]（LPR CH2：640 ns / 2048 点）
T0_NS = 0.0             # 1 行目の時刻 [ns]（time-zero 補正後に地表面 = 0 ns なら 0）
EPS_R = 3.0             # 時間 → 深さの換算に使う比誘電率（B-scan 図の深さ軸と同じ値）
BAND_MHZ = (250.0, 750.0)  # LPR CH2 の動作帯域 [MHz]（Fang et al. 2014）
DELIMITER = None        # None で自動（カンマ / タブ / 空白）

# --- 解析窓
#     x は移動距離 [m]、t0 は窓の上端の時刻 [ns]。時間の長さは全窓で共通（TIME_LEN_NS）
#     group が同じ窓の区間をまとめて平均する（グループごとに図を作る）
#     regolith: 最上部のレゴリス層のうち、強い反射・層境界・縦縞を避けた範囲
#               （地表直下 0-30 ns は直接波・地表反射の残りが強いので外している）
#     strong  : 比較用。反射が比較的強い範囲
#     random  : 比較用。深部のランダムノイズの範囲
WINDOWS = [
    dict(name='A', x_m=(20.0, 250.0), t0_ns=30.0, group='regolith'),    # 下の層境界 ~140-160 ns
    dict(name='B', x_m=(740.0, 990.0), t0_ns=30.0, group='regolith'),   # 下の層境界 ~190 ns。1030 m の縦縞の手前まで
    dict(name='C', x_m=(470.0, 700.0), t0_ns=200.0, group='strong'),    # 比較的反射が強い領域
    dict(name='D', x_m=(1000.0, 1250.0), t0_ns=150.0, group='strong'),  # 比較的反射が強い領域
    dict(name='E', x_m=(200.0, 400.0), t0_ns=500.0, group='random'),    # 深部のランダムノイズ
    dict(name='F', x_m=(1200.0, 1400.0), t0_ns=500.0, group='random'),   # 深部のランダムノイズ
]
TIME_LEN_NS = 100.0     # 解析窓の時間の長さ [ns]（ε_r = 3 で深さ 8.65 m 分）

# --- 区間長（水平方向のデータ長）
SEG_LENGTHS_M = (4.0, 10.0, 20.0, 50.0)
SEG_OVERLAP = 0.0       # 区間の重なりの割合（0 = 重ならない、0.5 = 半分重ねる）

# --- 前処理
WINDOW_FUNC = 'hann'    # 'hann' / 'blackmanharris' / 'none'（x と y の両方にかける）
BACKGROUND = 'none'     # 'none' / 'window'（解析窓内の平均トレースを各トレースから引く）
INVALID_FRAC_MAX = 0.01 # 区間内の無効なサンプルの割合がこれを超えたら除外
ZERO_RUN_MIN = 16       # 0 がこの点数以上（16 点 = 5 ns）続く所だけを詰め物（無効）とみなす
                        # 振幅が整数で弱い所では、ちょうど 0 の値がたまたま多く出るため、
                        # 単独の 0 は無効にしない

# --- 出力
SAVE_SEGMENT_FIGS = True  # 区間ごとの図を保存する
KY_MAX_PLOT = None      # 図の k_y の上限 [rad/m]。None でナイキスト波数まで（80 で約 1.1 GHz）
DB_RANGE = 50.0         # 2 次元スペクトル図の色の幅 [dB]
OVERVIEW_STRIDE = 10    # 全体図で間引くトレース数
DPI = 130
USE_CACHE = True

C0 = 299792458.0


# =============================================================================
# 読み込み
# =============================================================================
def _sniff_delimiter(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for ln in fh:
            s = ln.strip()
            if s and not s.startswith('#'):
                if ',' in s:
                    return ','
                if '\t' in s:
                    return '\t'
                return None
    return None


def load_data(path, outdir):
    stem = os.path.splitext(os.path.basename(path))[0]
    cache = os.path.join(outdir, '_cache_{}.npy'.format(stem))
    if USE_CACHE and os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        print('キャッシュを読み込みます: {}'.format(cache))
        return np.load(cache)
    t = time.time()
    delim = DELIMITER if DELIMITER is not None else _sniff_delimiter(path)
    print('txt を読み込みます（区切り: {}）...'.format(repr(delim) if delim else '空白'))
    try:
        import pandas as pd
        df = pd.read_csv(path, header=None, sep=delim if delim else r'\s+', engine='c',
                         comment='#', dtype=np.float32)
        data = df.to_numpy(dtype=np.float32)
        del df
    except ImportError:
        data = np.loadtxt(path, delimiter=delim, dtype=np.float32, comments='#')
    print('  {:.0f} s'.format(time.time() - t))
    if USE_CACHE:
        np.save(cache, data)
    return data


# =============================================================================
# スペクトル
# =============================================================================
def window_1d(n, kind=WINDOW_FUNC):
    if kind == 'none':
        return np.ones(n)
    if kind == 'hann':
        return np.hanning(n)
    if kind == 'blackmanharris':
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        t = 2 * np.pi * np.arange(n) / (n - 1)
        return a0 - a1 * np.cos(t) + a2 * np.cos(2 * t) - a3 * np.cos(3 * t)
    raise ValueError('WINDOW_FUNC は hann / blackmanharris / none')


def velocity():
    return C0 / np.sqrt(EPS_R)


def dy_m():
    return velocity() * DT_NS * 1e-9 / 2.0      # 1 サンプルあたりの深さ [m]


def ky_to_mhz(ky):
    return np.asarray(ky) * velocity() / (4 * np.pi) / 1e6


def mhz_to_ky(f):
    return 4 * np.pi * np.asarray(f) * 1e6 / velocity()


class Grid:
    """区間（N_y x N_x）の波数軸。S は [k_y >= 0 の行, k_x の全列] だけを持つ"""

    def __init__(self, ny, nx):
        self.ny, self.nx = ny, nx
        self.dy, self.dx = dy_m(), DX_M
        self.kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(nx, d=self.dx))
        ky_all = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(ny, d=self.dy))
        self.ky_sel = ky_all >= 0
        self.ky = ky_all[self.ky_sel]
        self.dkx = 2 * np.pi / (nx * self.dx)
        self.dky = 2 * np.pi / (ny * self.dy)
        self.wy, self.wx = window_1d(ny), window_1d(nx)
        self.w2 = np.outer(self.wy, self.wx)
        self.w2mean = float((self.w2 ** 2).mean())
        lo, hi = mhz_to_ky(BAND_MHZ[0]), mhz_to_ky(BAND_MHZ[1])
        self.band = (self.ky >= lo) & (self.ky <= hi)
        self.kx_pos = self.kx > 0
        self.ky_pos = self.ky > 0

    def psd(self, block):
        """block[y, x] -> S[k_y >= 0, k_x]（fftshift 済み）"""
        b = block - block.mean()
        X = np.fft.fftshift(np.fft.fft2(b * self.w2))
        S = np.abs(X) ** 2 * self.dx * self.dy / ((2 * np.pi) ** 2 * self.ny * self.nx * self.w2mean)
        return S[self.ky_sel]

    def marginals(self, S):
        """片側の 1 次元スペクトル。S_x は帯域内の k_y だけで積分、S_y は全 k_x で積分"""
        # S(-kx, -ky) = S(kx, ky) なので、k_y >= 0 側の和を 2 倍すると k_y 全体の和になる
        Sx_full = 2.0 * S[self.band].sum(axis=0) * self.dky        # k_x 全体（両側）
        Sx = _fold(Sx_full, self.kx)
        Sy = 2.0 * S.sum(axis=1) * self.dkx                          # k_y > 0 の片側
        return Sx, Sy[self.ky_pos]


def _fold(s_full, k):
    """両側スペクトル s_full(k)（fftshift 済み）を片側 s(k) + s(-k)（k > 0）にまとめる"""
    i0 = int(np.argmin(np.abs(k)))                 # k = 0 の位置
    m = np.arange(1, len(k) - i0)                  # k > 0 の番号
    return s_full[i0 + m] + s_full[i0 - m]         # fftshift 後は -k が必ずある


# =============================================================================
# 区間の切り出し
# =============================================================================
def window_block(data, w):
    nt, ntr = data.shape
    it0 = int(round((w['t0_ns'] - T0_NS) / DT_NS))
    ny = int(round(TIME_LEN_NS / DT_NS))
    ix0 = int(round(w['x_m'][0] / DX_M))
    ix1 = int(round(w['x_m'][1] / DX_M))
    if it0 < 0 or it0 + ny > nt or ix0 < 0 or ix1 > ntr or ix1 <= ix0:
        raise SystemExit('解析窓 {} がデータの範囲外です（時刻 {}-{} ns、距離 {}-{} m、データは '
                         '{:.1f}-{:.1f} ns, 0-{:.1f} m）'.format(
                             w['name'], w['t0_ns'], w['t0_ns'] + TIME_LEN_NS, *w['x_m'],
                             T0_NS, T0_NS + nt * DT_NS, ntr * DX_M))
    blk = data[it0:it0 + ny, ix0:ix1].astype(np.float64)
    if BACKGROUND == 'window':
        good = ~invalid_mask(blk)
        mean_tr = np.where(good, blk, 0).sum(axis=1) / np.maximum(good.sum(axis=1), 1)
        blk = np.where(good, blk - mean_tr[:, None], blk)
    return blk, it0, ix0


def invalid_mask(blk, n=None):
    """無効なサンプル：NaN / inf と、時間方向に 0 が n 点以上続く所（地形補正の詰め物や
    欠けたトレース）。単独の 0 は有効な値として扱う。"""
    n = ZERO_RUN_MIN if n is None else n
    bad = ~np.isfinite(blk)
    z = (blk == 0).astype(np.int32)
    if n <= 1:
        return bad | z.astype(bool)
    ny = z.shape[0]
    if ny < n:
        return bad
    c = np.concatenate([np.zeros((1, z.shape[1]), np.int32), np.cumsum(z, axis=0)], axis=0)
    full = (c[n:] - c[:-n]) == n                  # [ny - n + 1, nx]：i から n 点がすべて 0
    cov = np.zeros((ny + 1, z.shape[1]), np.int32)  # 長さ n の窓が覆うサンプルに印を付ける
    cov[:ny - n + 1] += full
    cov[n:ny + 1] -= full
    run = np.cumsum(cov, axis=0)[:ny] > 0
    return bad | run


def segments(w, blk, ix0, Ls, bad_all):
    nx = int(round(Ls / DX_M))
    hop = max(1, int(round(nx * (1.0 - SEG_OVERLAP))))
    out = []
    for s in range(0, blk.shape[1] - nx + 1, hop):
        frac = float(bad_all[:, s:s + nx].mean())
        x0 = (ix0 + s) * DX_M
        out.append(dict(win=w['name'], x0=x0, x1=x0 + nx * DX_M, col=s, nx=nx,
                        invalid=frac, ok=frac <= INVALID_FRAC_MAX))
    return out


# =============================================================================
# 図
# =============================================================================
def _db(a):
    return 10 * np.log10(np.maximum(a, 1e-300))


WIN_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#e377c2']
GROUP_STYLES = ['-', '--', ':', '-.']


ALL_NAME = 'all'          # 全解析窓をまとめた平均の名前


def _group(w):
    return w.get('group', 'default')


def groups():
    out = []
    for w in WINDOWS:
        if _group(w) not in out:
            out.append(_group(w))
    return out


def _win_color(name):
    i = [w['name'] for w in WINDOWS].index(name)
    return WIN_COLORS[i % len(WIN_COLORS)]


def ky_max(g=None):
    kn = np.pi / dy_m()
    return kn if KY_MAX_PLOT is None else min(KY_MAX_PLOT, kn)


def draw_map(ax, g, S, vmax):
    m = g.ky <= ky_max() + 1e-9
    im = ax.imshow(_db(S[m]), origin='lower', aspect='auto', cmap='magma',
                   extent=[g.kx[0] - g.dkx / 2, g.kx[-1] + g.dkx / 2,
                           g.ky[m][0] - g.dky / 2, g.ky[m][-1] + g.dky / 2],
                   vmin=vmax - DB_RANGE, vmax=vmax)
    for f in BAND_MHZ:
        ax.axhline(mhz_to_ky(f), color='cyan', ls='--', lw=0.8)
    ax.set_xlabel('k_x [rad/m]')
    ax.set_ylabel('k_y [rad/m]')
    ax.set_title('S(k_x, k_y)')
    ax.legend(handles=[Line2D([], [], color='cyan', ls='--', lw=1,
                              label=f'{BAND_MHZ[0]:g}–{BAND_MHZ[1]:g} MHz')],
              loc='upper right', fontsize=7.5, framealpha=0.7)
    return im


def _res_lines(ax, dk, kn, axis, show_dk=True):
    """分解能 Δk（一点鎖線）とナイキスト波数 k_N（点線）の縦線と、数値の注記"""
    hs = []
    if show_dk:
        ax.axvline(dk, color='tab:olive', ls='-.', lw=1.0)
        hs.append(Line2D([], [], color='tab:olive', ls='-.', lw=1.0,
                         label=f'Δk_{axis} = {dk:.3f} rad/m'))
    lo, hi = ax.get_xlim()
    ax.axvline(kn, color='tab:gray', ls=':', lw=1.4)
    lab = f'k_N,{axis} = {kn:.1f} rad/m'
    if kn > hi:
        lab += ' (off scale)'
    hs.append(Line2D([], [], color='tab:gray', ls=':', lw=1.4, label=lab))
    ax.set_xlim(lo, hi)
    return hs


def draw_sx(ax, g, curves, ylim=None):
    k = g.kx[g.kx_pos]
    for c in curves:
        ax.semilogx(k, _db(c['s']), color=c['color'], lw=c.get('lw', 1.3), ls=c.get('ls', '-'),
                    label=c.get('label'))
    ax.set_xlim(0.8 * g.dkx, 1.25 * np.pi / DX_M)
    hs = _res_lines(ax, g.dkx, np.pi / DX_M, 'x')
    ax.set_xlabel('k_x [rad/m]')
    ax.set_ylabel('S_x [dB]')
    ax.set_title(f'S_x(k_x), k_y in {BAND_MHZ[0]:g}–{BAND_MHZ[1]:g} MHz')
    ax.grid(alpha=0.3, which='both')
    if ylim:
        ax.set_ylim(*ylim)
    h0 = [Line2D([], [], color=c['color'], lw=c.get('lw', 1.3), ls=c.get('ls', '-'),
                 label=c['label']) for c in curves if c.get('label')]
    ax.legend(handles=h0 + hs, fontsize=7.5, loc='lower left')


def draw_sy(ax, g, curves, ylim=None):
    k = g.ky[g.ky_pos]
    km = ky_max()
    m = k <= km + 1e-9
    for c in curves:
        ax.plot(k[m], _db(c['s'][m]), color=c['color'], lw=c.get('lw', 1.3), ls=c.get('ls', '-'),
                label=c.get('label'))
    ax.axvspan(mhz_to_ky(BAND_MHZ[0]), mhz_to_ky(BAND_MHZ[1]), color='cyan', alpha=0.12)
    ax.set_xlim(0, km * 1.02)
    hs = _res_lines(ax, g.dky, np.pi / dy_m(), 'y')
    ax.set_xlabel('k_y [rad/m]')
    ax.set_ylabel('S_y [dB]')
    ax.set_title('S_y(k_y)')
    ax.grid(alpha=0.3)
    sec = ax.secondary_xaxis('top', functions=(ky_to_mhz, mhz_to_ky))
    sec.set_xlabel('f [MHz]')
    if ylim:
        ax.set_ylim(*ylim)
    h0 = [Line2D([], [], color=c['color'], lw=c.get('lw', 1.3), ls=c.get('ls', '-'),
                 label=c['label']) for c in curves if c.get('label')]
    h0.append(Patch(fc='cyan', alpha=0.25, label=f'{BAND_MHZ[0]:g}–{BAND_MHZ[1]:g} MHz'))
    ax.legend(handles=h0 + hs, fontsize=7.5, loc='lower center')


def _ylim(arrs, rng=60.0):
    top = max(float(_db(a).max()) for a in arrs)
    return top - rng, top + 5


def plot_overview(data, outdir, amp):
    nt, ntr = data.shape
    st = max(1, OVERVIEW_STRIDE)
    img = np.nan_to_num(data[:, ::st])
    fig, ax = plt.subplots(figsize=(16, 5.5), layout='constrained')
    ax.imshow(img, aspect='auto', cmap='seismic', vmin=-amp, vmax=amp,
              extent=[0, ntr * DX_M, T0_NS + nt * DT_NS, T0_NS])
    gl = groups()
    for w in WINDOWS:
        col = _win_color(w['name'])
        ls = GROUP_STYLES[gl.index(_group(w)) % len(GROUP_STYLES)]
        ax.add_patch(Rectangle((w['x_m'][0], w['t0_ns']), w['x_m'][1] - w['x_m'][0], TIME_LEN_NS,
                               fill=False, ec=col, lw=2, ls=ls))
        ax.text(w['x_m'][0], w['t0_ns'] - 4, w['name'], color=col, fontsize=12, va='bottom',
                fontweight='bold')
    ax.legend(handles=[Line2D([], [], color='k', lw=2, ls=GROUP_STYLES[i % len(GROUP_STYLES)],
                              label=gname) for i, gname in enumerate(gl)],
              loc='lower right', fontsize=9, framealpha=0.85)
    ax.set_xlabel('Moving distance [m]')
    ax.set_ylabel('Time [ns]')
    sec = ax.secondary_yaxis('right', functions=(lambda t: t * 1e-9 * velocity() / 2,
                                                 lambda d: d * 2 / velocity() * 1e9))
    sec.set_ylabel(f'Depth [m] (ε_r = {EPS_R:g})')
    ax.set_title(f'Analysis windows ({TIME_LEN_NS:g} ns each)')
    fig.savefig(os.path.join(outdir, 'overview_windows.png'), dpi=DPI)
    plt.close(fig)


def _lname(Ls):
    return f'{Ls:03.0f}' if float(Ls).is_integer() else f'{Ls:05.1f}'.replace('.', 'p')


def plot_mean(outdir, gname, Ls, g, res, vmax):
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6), layout='constrained',
                            gridspec_kw={'width_ratios': [1.2, 1, 1]})
    im = draw_map(axs[0], g, res['S'], vmax)
    fig.colorbar(im, ax=axs[0], label='dB')
    cx = [dict(s=res['Sx'], color='k', lw=2.2, label=f'{gname} mean ({res["n"]} segments)')]
    cy = [dict(s=res['Sy'], color='k', lw=2.2, label=f'{gname} mean ({res["n"]} segments)')]
    gl = groups()
    for name, r in res['by_win'].items():
        col = _win_color(name)
        gw = _group(next(w for w in WINDOWS if w['name'] == name))
        ls = GROUP_STYLES[gl.index(gw) % len(GROUP_STYLES)]
        lab = f'{name} ({r["n"]})' if gname != ALL_NAME else f'{name}, {gw} ({r["n"]})'
        cx.append(dict(s=r['Sx'], color=col, lw=1.0, ls=ls, label=lab))
        cy.append(dict(s=r['Sy'], color=col, lw=1.0, ls=ls, label=lab))
    arrs_x = [res['Sx']] + [r['Sx'] for r in res['by_win'].values()]
    arrs_y = [res['Sy']] + [r['Sy'] for r in res['by_win'].values()]
    draw_sx(axs[1], g, cx, _ylim(arrs_x))
    draw_sy(axs[2], g, cy, _ylim(arrs_y))
    names = ', '.join(res['by_win'])
    fig.suptitle(f'Mean spectrum, {gname}, L = {Ls:g} m (windows {names}, {TIME_LEN_NS:g} ns, '
                 f'window: {WINDOW_FUNC})', fontsize=13)
    fig.savefig(os.path.join(outdir, f'mean_{gname}_L{_lname(Ls)}m.png'), dpi=DPI)
    plt.close(fig)


def plot_segment(path, seg, blk, g, S, Sx, Sy, vmax, ylim_x, ylim_y, amp, t0_ns):
    fig = plt.figure(figsize=(16, 10.5), layout='constrained')
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(blk, aspect='auto', cmap='seismic', vmin=-amp, vmax=amp,
              extent=[seg['x0'], seg['x1'], t0_ns + TIME_LEN_NS, t0_ns])
    ax.set_xlabel('Moving distance [m]')
    ax.set_ylabel('Time [ns]')
    ax.set_title('B-scan')
    sec = ax.secondary_yaxis('right', functions=(lambda t: t * 1e-9 * velocity() / 2,
                                                 lambda d: d * 2 / velocity() * 1e9))
    sec.set_ylabel(f'Depth [m] (ε_r = {EPS_R:g})')
    ax = fig.add_subplot(gs[0, 1])
    im = draw_map(ax, g, S, vmax)
    fig.colorbar(im, ax=ax, label='dB')
    draw_sx(fig.add_subplot(gs[1, 0]), g, [dict(s=Sx, color='k')], ylim_x)
    draw_sy(fig.add_subplot(gs[1, 1]), g, [dict(s=Sy, color='k')], ylim_y)
    fig.suptitle(f'Window {seg["win"]} ({seg["group"]}), x = {seg["x0"]:.2f}–{seg["x1"]:.2f} m '
                 f'(L = {seg["x1"] - seg["x0"]:.2f} m), t = {t0_ns:g}–{t0_ns + TIME_LEN_NS:g} ns',
                 fontsize=13)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _compare_axes(axs, kxmin):
    axs[0].set_xlabel('k_x [rad/m]')
    axs[0].set_ylabel('S_x [dB]')
    axs[0].set_title(f'S_x(k_x), k_y in {BAND_MHZ[0]:g}–{BAND_MHZ[1]:g} MHz')
    axs[0].set_xlim(0.8 * kxmin, 1.25 * np.pi / DX_M)
    axs[1].axvspan(mhz_to_ky(BAND_MHZ[0]), mhz_to_ky(BAND_MHZ[1]), color='cyan', alpha=0.12)
    axs[1].set_xlim(0, ky_max() * 1.02)
    axs[1].set_xlabel('k_y [rad/m]')
    axs[1].set_ylabel('S_y [dB]')
    axs[1].set_title('S_y(k_y)')
    sec = axs[1].secondary_xaxis('top', functions=(ky_to_mhz, mhz_to_ky))
    sec.set_xlabel('f [MHz]')
    for ax in axs:
        ax.grid(alpha=0.3, which='both')


def plot_compare_lengths(outdir, gname, res_by_L):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.8), layout='constrained')
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(res_by_L)))
    hx, hy = [], []
    for (Ls, (g, res)), col in zip(res_by_L.items(), cols):
        axs[0].semilogx(g.kx[g.kx_pos], _db(res['Sx']), color=col, lw=1.5)
        axs[0].axvline(g.dkx, color=col, ls='-.', lw=0.9)
        hx.append(Line2D([], [], color=col, lw=1.5,
                         label=f'L = {Ls:g} m ({res["n"]} seg.), Δk_x = {g.dkx:.3f}'))
        ky = g.ky[g.ky_pos]
        m = ky <= ky_max() + 1e-9
        axs[1].plot(ky[m], _db(res['Sy'][m]), color=col, lw=1.5)
        hy.append(Line2D([], [], color=col, lw=1.5, label=f'L = {Ls:g} m'))
    g0 = next(iter(res_by_L.values()))[0]
    _compare_axes(axs, min(g.dkx for g, _ in res_by_L.values()))
    hx += _res_lines(axs[0], g0.dkx, np.pi / DX_M, 'x', show_dk=False)
    hx.append(Line2D([], [], color='k', ls='-.', lw=0.9, label='Δk_x (colour = L)'))
    hy += _res_lines(axs[1], g0.dky, np.pi / dy_m(), 'y')
    axs[0].legend(handles=hx, fontsize=7.5, loc='lower left')
    axs[1].legend(handles=hy, fontsize=7.5, loc='lower center')
    top = max(float(_db(r['Sx']).max()) for _, r in res_by_L.values())
    axs[0].set_ylim(top - 60, top + 5)
    top = max(float(_db(r['Sy']).max()) for _, r in res_by_L.values())
    axs[1].set_ylim(top - 60, top + 5)
    fig.suptitle(f'Mean spectra for different segment lengths ({gname})', fontsize=13)
    fig.savefig(os.path.join(outdir, f'compare_lengths_{gname}.png'), dpi=DPI)
    plt.close(fig)


def plot_compare_groups(outdir, Ls, res_by_group):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.8), layout='constrained')
    hx, hy = [], []
    for i, (gname, (g, res)) in enumerate(res_by_group.items()):
        ls = GROUP_STYLES[i % len(GROUP_STYLES)]
        axs[0].semilogx(g.kx[g.kx_pos], _db(res['Sx']), color='k', lw=1.6, ls=ls)
        ky = g.ky[g.ky_pos]
        m = ky <= ky_max() + 1e-9
        axs[1].plot(ky[m], _db(res['Sy'][m]), color='k', lw=1.6, ls=ls)
        for name, r in res['by_win'].items():
            col = _win_color(name)
            axs[0].semilogx(g.kx[g.kx_pos], _db(r['Sx']), color=col, lw=0.9, ls=ls)
            axs[1].plot(ky[m], _db(r['Sy'][m]), color=col, lw=0.9, ls=ls)
            hx.append(Line2D([], [], color=col, lw=0.9, ls=ls, label=f'{name} ({r["n"]})'))
        hx.insert(len(hx) - len(res['by_win']),
                  Line2D([], [], color='k', lw=1.6, ls=ls, label=f'{gname} mean ({res["n"]})'))
    g0 = next(iter(res_by_group.values()))[0]
    _compare_axes(axs, g0.dkx)
    hy = list(hx)
    hx += _res_lines(axs[0], g0.dkx, np.pi / DX_M, 'x')
    hy += _res_lines(axs[1], g0.dky, np.pi / dy_m(), 'y')
    axs[0].legend(handles=hx, fontsize=7.5, loc='lower left')
    axs[1].legend(handles=hy, fontsize=7.5, loc='lower center')
    top = max(float(_db(r['Sx']).max()) for _, r in res_by_group.values())
    axs[0].set_ylim(top - 60, top + 5)
    top = max(float(_db(r['Sy']).max()) for _, r in res_by_group.values())
    axs[1].set_ylim(top - 60, top + 5)
    fig.suptitle(f'Mean spectra by group, L = {Ls:g} m', fontsize=13)
    fig.savefig(os.path.join(outdir, f'compare_groups_L{_lname(Ls)}m.png'), dpi=DPI)
    plt.close(fig)


# =============================================================================
def main():
    #* Get input parameters
    print('データファイルのパスを入力してください:')
    data_path = input().strip()
    if not os.path.exists(data_path):
        print('エラー: 指定されたファイルが存在しません')
        exit(1)

    t_all = time.time()
    outdir = os.path.join(os.path.dirname(os.path.abspath(data_path)), '2D_FFT')
    os.makedirs(outdir, exist_ok=True)

    data = load_data(data_path, outdir)
    nt, ntr = data.shape
    n_lead0 = int(np.argmax(np.any(np.nan_to_num(data) != 0, axis=1)))
    print('データ: {} 行（時間）x {} 列（トレース）'.format(nt, ntr))
    print('  時間 {:.2f}-{:.2f} ns（DT_NS = {} ns、T0_NS = {} ns）'.format(
        T0_NS, T0_NS + nt * DT_NS, DT_NS, T0_NS))
    print('  距離 0-{:.1f} m（DX_M = {} m）'.format(ntr * DX_M, DX_M))
    print('  NaN {} 個 / 先頭から全トレースで 0 の行 {} 行'.format(
        int(np.isnan(data).sum()), n_lead0))
    print('  深さ換算 ε_r = {}、1 サンプル {:.4f} m、v = {:.4f} m/ns'.format(
        EPS_R, dy_m(), velocity() * 1e-9))
    print('  ナイキスト波数 k_N,x = {:.2f} rad/m、k_N,y = {:.2f} rad/m（{:.0f} MHz）'.format(
        np.pi / DX_M, np.pi / dy_m(), float(ky_to_mhz(np.pi / dy_m()))))
    if nt > ntr:
        print('  注意: 行数が列数より多い。各列が A-scan になっているか確認すること')

    finite = data[np.isfinite(data) & (data != 0)]
    amp = float(np.percentile(np.abs(finite[::max(1, finite.size // 2_000_000)]), 99))
    del finite
    plot_overview(data, outdir, amp)
    print('-> {}'.format(os.path.join(outdir, 'overview_windows.png')))

    blocks = {}
    for w in WINDOWS:
        blk, it0, ix0 = window_block(data, w)
        bad = invalid_mask(blk)
        blocks[w['name']] = (w, blk, ix0, bad)
        print('解析窓 {}（{}）: x = {:g}-{:g} m（{} トレース）、t = {:g}-{:g} ns（{} 点）、'
              '無効なサンプル {:.2%}（単独の 0 は {:.2%}）'.format(
                  w['name'], _group(w), *w['x_m'], blk.shape[1], w['t0_ns'],
                  w['t0_ns'] + TIME_LEN_NS, blk.shape[0], float(bad.mean()),
                  float(((blk == 0) & ~bad).mean())))
    del data

    gl = groups()
    # 平均をとる単位：各グループと、グループが 2 つ以上あれば全解析窓をまとめたもの（all）
    avg_units = gl + ([ALL_NAME] if len(gl) > 1 else [])
    results = {gname: {} for gname in avg_units}     # results[group][Ls] = (grid, res)
    info = dict(settings={k: v for k, v in globals().items()
                          if k.isupper() and isinstance(v, (int, float, str, tuple, list,
                                                            dict, type(None)))},
                data_path=os.path.abspath(data_path), shape=[nt, ntr],
                nyquist=dict(kx=np.pi / DX_M, ky=np.pi / dy_m()), lengths={})
    for Ls in SEG_LENGTHS_M:
        t = time.time()
        ny = int(round(TIME_LEN_NS / DT_NS))
        nx = int(round(Ls / DX_M))
        g = Grid(ny, nx)
        segs, spectra = [], []
        for name, (w, blk, ix0, bad) in blocks.items():
            for s in segments(w, blk, ix0, Ls, bad):
                s['group'] = _group(w)
                segs.append(s)
                if s['ok']:
                    b = np.nan_to_num(blk[:, s['col']:s['col'] + nx])
                    spectra.append(g.psd(b))
                else:
                    spectra.append(None)
        good = [i for i, sp in enumerate(spectra) if sp is not None]
        print('L = {:g} m: 区間 {}（除外 {}）、Δk_x = {:.3f} rad/m、Δk_y = {:.3f} rad/m'.format(
            Ls, len(good), len(segs) - len(good), g.dkx, g.dky))
        for w in WINDOWS:
            n_all = sum(1 for s in segs if s['win'] == w['name'])
            n_ok = sum(1 for i in good if segs[i]['win'] == w['name'])
            msg = '    {}: 使用 {} / {}'.format(w['name'], n_ok, n_all)
            if n_ok == 0:
                msg += '  ** 全区間が除外されたので、この窓はどの図にも入りません **'
            print(msg)
        vmax_g, ylim_g = {}, {}
        for gname in avg_units:
            gi = [i for i in good if gname == ALL_NAME or segs[i]['group'] == gname]
            if not gi:
                print('  {}: 有効な区間がありません'.format(gname))
                continue
            Smean = np.mean([spectra[i] for i in gi], axis=0)
            Sx, Sy = g.marginals(Smean)
            res = dict(S=Smean, Sx=Sx, Sy=Sy, n=len(gi), by_win={})
            for w in WINDOWS:
                idx = [i for i in gi if segs[i]['win'] == w['name']]
                if idx:
                    sx, sy = g.marginals(np.mean([spectra[i] for i in idx], axis=0))
                    res['by_win'][w['name']] = dict(Sx=sx, Sy=sy, n=len(idx))
            results[gname][Ls] = (g, res)
            vmax_g[gname] = float(_db(Smean[g.ky <= ky_max() + 1e-9]).max())
            ylx, yly = _ylim([Sx], 50.0), _ylim([Sy], 60.0)
            ylim_g[gname] = ((ylx[0], ylx[1] + 10), (yly[0], yly[1] + 10))
            plot_mean(outdir, gname, Ls, g, res, vmax_g[gname])
            print('  {}: {} 区間 -> {}'.format(
                gname, len(gi), os.path.join(outdir, f'mean_{gname}_L{_lname(Ls)}m.png')))
        here = {gname: results[gname][Ls] for gname in gl if Ls in results[gname]}   # all は除く
        if len(here) > 1:
            plot_compare_groups(outdir, Ls, here)

        if SAVE_SEGMENT_FIGS:
            sub = os.path.join(outdir, f'L{_lname(Ls)}m')
            os.makedirs(sub, exist_ok=True)
            for i in good:
                s = segs[i]
                w, blk, ix0, _ = blocks[s['win']]
                sx, sy = g.marginals(spectra[i])
                fn = '{}_x{:07.2f}-{:07.2f}m.png'.format(s['win'], s['x0'], s['x1'])
                plot_segment(os.path.join(sub, fn), s, np.nan_to_num(blk[:, s['col']:s['col'] + nx]),
                             g, spectra[i], sx, sy, vmax_g[s['group']] + 10, *ylim_g[s['group']],
                             amp, w['t0_ns'])
            print('  -> {}/（{} 枚）'.format(sub, len(good)))
        info['lengths'][str(Ls)] = dict(
            n_used=len(good), n_excluded=len(segs) - len(good), dkx=g.dkx, dky=g.dky, nx=nx, ny=ny,
            segments=[{k: v for k, v in s.items() if k != 'col'} for s in segs])
        print('  ({:.0f} s)'.format(time.time() - t))

    np.savez(os.path.join(outdir, 'spectra_mean.npz'),
             **{f'{gname}_L{_lname(Ls)}_{key}': val
                for gname, byL in results.items() for Ls, (g, r) in byL.items()
                for key, val in (('kx', g.kx), ('ky', g.ky), ('S', r['S']),
                                 ('kx_pos', g.kx[g.kx_pos]), ('Sx', r['Sx']),
                                 ('ky_pos', g.ky[g.ky_pos]), ('Sy', r['Sy']),
                                 ('n', np.array(r['n'])))})
    for gname, byL in results.items():
        if byL:
            plot_compare_lengths(outdir, gname, byL)
    with open(os.path.join(outdir, 'run_info.json'), 'w', encoding='utf-8') as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1, default=str)
    print('完了（{:.0f} s）。出力先: {}'.format(time.time() - t_all, outdir))


if __name__ == '__main__':
    main()