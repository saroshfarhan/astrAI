from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter

# IR wavelength constraint (must match training configuration in train_ir.py)
IR_WAVELENGTH_RANGE = (5.0, 36.0)  # micrometers

def crop_to_wavelength_range(
    wave: np.ndarray,
    flux: np.ndarray,
    wave_min: float,
    wave_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Crop spectrum to specified wavelength range.

    Used for IR data to focus on the overlapping region (5-36 um) where
    training data has consistent coverage.
    """
    mask = (wave >= wave_min) & (wave <= wave_max)
    return wave[mask], flux[mask]

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
    """Create 3-channel representation: [normalized, 1st derivative, 2nd derivative].

    Uses robust normalization to handle diverse flux scales across different planets.
    IMPORTANT: Must match training code in train_ir.py for consistent predictions.
    """
    # Robust baseline normalization
    base = compute_baseline(flux, win=baseline_win, poly=3)
    r = (flux / base) - 1.0

    # Robust z-score normalization using median absolute deviation (MAD)
    median = np.median(r)
    mad = np.median(np.abs(r - median))
    r = (r - median) / (mad * 1.4826 + 1e-8)  # 1.4826 makes MAD consistent with std for normal dist

    # Clip outliers for stability
    r = np.clip(r, -5, 5)

    # Derivatives (already scale-invariant due to normalization)
    d1 = np.gradient(r, wave)
    d2 = np.gradient(d1, wave)

    # Normalize derivatives independently for stability
    d1 = (d1 - np.median(d1)) / (np.std(d1) + 1e-8)
    d2 = (d2 - np.median(d2)) / (np.std(d2) + 1e-8)

    X = np.stack([r, d1, d2], axis=0).astype(np.float32)  # (3,N)
    return X
