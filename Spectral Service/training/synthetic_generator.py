# spectral_service/training/synthetic_generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.signal import savgol_filter

BandMap = Dict[str, List[Tuple[float, float]]]  # species -> [(center, fwhm), ...]

# -------------------------
# Default UV (your expanded 12)
# -------------------------
UV_SPECIES = [
    "CH4", "NH3", "C2H2", "C2H6",
    "PH3", "C6H6", "C4H2",
    "C2H4", "C3H4", "GeH4", "AsH3", "CO"
]

UV_BANDS: BandMap = {
    "CH4":  [(2350, 260), (2750, 320)],
    "C2H2": [(2700, 140), (2810, 130), (2200, 150)],
    "C2H6": [(2400, 170), (2550, 150)],

    "NH3":  [(2050, 140), (2150, 140), (2250, 100)],
    "PH3":  [(2100, 200), (2300, 150)],

    "C6H6": [(2550, 80), (2600, 60)],
    "C4H2": [(2200, 50), (2320, 50), (2440, 50)],

    "C2H4": [(1700, 150), (1850, 100)],
    "C3H4": [(1900, 150), (2050, 100)],
    "GeH4": [(1950, 200), (2100, 100)],
    "AsH3": [(2000, 200), (2150, 100)],
    "CO":   [(1990, 40), (2060, 40)],
}

# -------------------------
# Starter IR set (EDIT ME as you refine)
# Units assumed consistent with your IR PKLs (often microns)
# If your IR is in nm, this still works, but band centers should match that unit.
# -------------------------
IR_SPECIES = ["CH4", "NH3", "PH3", "CO", "C2H6", "C2H2"]

IR_BANDS: BandMap = {
    # rough starter bands, replace with proper IR lines once you confirm IR units
    "CH4": [(3.3, 0.25), (2.3, 0.20)],
    "NH3": [(10.5, 0.6), (2.0, 0.20)],
    "PH3": [(4.3, 0.30)],
    "CO":  [(4.7, 0.15)],
    "C2H6": [(3.4, 0.25)],
    "C2H2": [(3.0, 0.20)],
}

# -------------------------
# Planet priors (tweakable)
# -------------------------
DEFAULT_PRIORS_UV = {
    "JUPITER": {"CH4": 0.90, "NH3": 0.35, "C2H2": 0.20, "C2H6": 0.20, "PH3": 0.15, "CO": 0.10},
    "SATURN":  {"CH4": 0.85, "NH3": 0.15, "C2H2": 0.35, "C2H6": 0.35, "PH3": 0.30, "C6H6": 0.20, "C4H2": 0.25},
    "URANUS":  {"CH4": 0.95, "NH3": 0.05, "C2H2": 0.20, "C2H6": 0.25, "C6H6": 0.10, "C4H2": 0.15},
    "MARS":    {"CO": 0.40, "CH4": 0.10, "C2H4": 0.10, "C2H2": 0.10},  # Mars UV is different; keep priors conservative
}

DEFAULT_PRIORS_IR = {
    "SATURN": {"CH4": 0.80, "NH3": 0.20, "PH3": 0.35, "CO": 0.10, "C2H6": 0.30, "C2H2": 0.20},
    "URANUS": {"CH4": 0.95, "NH3": 0.05, "PH3": 0.10, "CO": 0.10, "C2H6": 0.25, "C2H2": 0.20},
}

# -------------------------
# Core math
# -------------------------
def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def compute_baseline(flux: np.ndarray, win: int, poly: int = 3) -> np.ndarray:
    n = len(flux)
    win = int(win)
    if win >= n:
        win = n - 1
    win = max(11, win)
    if win % 2 == 0:
        win += 1
    b = savgol_filter(flux.astype(float), window_length=win, polyorder=poly)
    b = np.clip(b, np.percentile(b, 1), np.percentile(b, 99))
    return (b.astype(np.float32) + 1e-12)

def make_channels(wave: np.ndarray, flux: np.ndarray, win: int) -> np.ndarray:
    base = compute_baseline(flux, win=win, poly=3)
    r = (flux / base) - 1.0
    r = (r - np.median(r)) / (np.std(r) + 1e-8)
    d1 = np.gradient(r, wave)
    d2 = np.gradient(d1, wave)
    return np.stack([r, d1, d2], axis=0).astype(np.float32)  # (3, N)

def resample_to_fixed(wave: np.ndarray, flux: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    w_new = np.linspace(float(wave.min()), float(wave.max()), n)
    f_new = np.interp(w_new, wave, flux)
    return w_new.astype(np.float32), f_new.astype(np.float32)

def sample_labels(
    species: List[str],
    priors: Dict[str, float],
    rng: np.random.Generator
) -> Dict[str, int]:
    y = {sp: 0 for sp in species}
    for sp in species:
        p = float(priors.get(sp, 0.03))
        y[sp] = 1 if rng.random() < p else 0
    if sum(y.values()) == 0:
        # ensure at least one label
        pick = rng.choice(species)
        y[pick] = 1
    return y

def synth_spectrum(
    wave: np.ndarray,
    real_flux: np.ndarray,
    labels: Dict[str, int],
    bands: BandMap,
    rng: np.random.Generator,
    win: int,
) -> np.ndarray:
    baseline = compute_baseline(real_flux, win=win, poly=3)
    x = (wave - wave.min()) / (wave.max() - wave.min() + 1e-12)
    drift = 1.0 + rng.normal(0, 0.01) + rng.normal(0, 0.01) * (x - 0.5)
    spec = baseline * drift

    for sp, present in labels.items():
        if not present:
            continue
        for (c, w0) in bands.get(sp, []):
            depth = rng.uniform(0.02, 0.22)
            width = float(w0) * rng.uniform(0.85, 1.25)
            c_jit = float(c) + rng.normal(0, 0.02 * float(w0) if float(w0) > 1 else 0.02)
            dip = 1.0 - depth * gaussian(wave, c_jit, width)
            spec *= dip

    sigma = 0.01 * (np.ptp(spec) + 1e-8)  # NumPy 2.0 safe
    noise = rng.normal(0, sigma, size=len(spec))
    noise = np.convolve(noise, np.ones(7) / 7, mode="same")
    return (spec + noise).astype(np.float32)

def build_synthetic_from_planet(
    wave_raw: np.ndarray,
    flux_raw: np.ndarray,
    species: List[str],
    bands: BandMap,
    priors: Dict[str, float],
    n_resample: int,
    n: int,
    win: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    w_fix, f_fix = resample_to_fixed(wave_raw, flux_raw, n_resample)

    X_list, Y_list = [], []
    for _ in range(n):
        lab = sample_labels(species, priors, rng)
        spec = synth_spectrum(w_fix, f_fix, lab, bands, rng, win=win)
        X = make_channels(w_fix, spec, win=win)
        y = np.array([lab[sp] for sp in species], dtype=np.float32)
        X_list.append(X)
        Y_list.append(y)

    X = np.stack(X_list, axis=0)  # (B,3,N)
    Y = np.stack(Y_list, axis=0)  # (B,K)
    return X, Y
