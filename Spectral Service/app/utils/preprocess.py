from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter

def resample_to_fixed(wave: np.ndarray, flux: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    w_new = np.linspace(wave.min(), wave.max(), n)
    f_new = np.interp(w_new, wave, flux)
    return w_new.astype(np.float32), f_new.astype(np.float32)

def compute_baseline(flux: np.ndarray, win: int = 151, poly: int = 3) -> np.ndarray:
    n = len(flux)
    win = int(win)
    if win >= n:
        win = n - 1
    if win < 11:
        win = 11
    if win % 2 == 0:
        win += 1
    b = savgol_filter(flux.astype(float), window_length=win, polyorder=poly)
    eps = 1e-12
    b = np.clip(b, np.percentile(b, 1), np.percentile(b, 99)) + eps
    return b.astype(np.float32)

def make_channels(wave: np.ndarray, flux: np.ndarray, baseline_win: int) -> np.ndarray:
    base = compute_baseline(flux, win=baseline_win, poly=3)
    r = (flux / base) - 1.0
    r = (r - np.median(r)) / (np.std(r) + 1e-8)
    d1 = np.gradient(r, wave)
    d2 = np.gradient(d1, wave)
    X = np.stack([r, d1, d2], axis=0).astype(np.float32)  # (3,N)
    return X
