"""Physics-based data augmentation for spectral data.

This module provides realistic augmentation techniques that mimic
real-world instrumental and atmospheric effects.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


def augment_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    seed: int | None = None,
    config: dict | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply physics-based augmentation to a spectrum.

    Args:
        wave: Wavelength array
        flux: Flux array
        seed: Random seed for reproducibility
        config: Augmentation configuration (optional)

    Returns:
        Augmented (wavelength, flux) tuple
    """
    rng = np.random.default_rng(seed)

    # Default configuration
    if config is None:
        config = {
            'wavelength_shift': True,
            'flux_scaling': True,
            'baseline_tilt': True,
            'baseline_curve': True,
            'noise': True,
            'resolution_smooth': True,
            'feature_scaling': True,
        }

    wave_aug = wave.copy()
    flux_aug = flux.copy()

    # 1. Wavelength shift (instrument calibration: ±0.2%)
    if config.get('wavelength_shift', True):
        shift = rng.uniform(-0.002, 0.002)
        wave_aug = wave * (1 + shift)

    # 2. Flux scaling (exposure time/distance variation: ±10%)
    if config.get('flux_scaling', True):
        scale = rng.uniform(0.9, 1.1)
        flux_aug *= scale

    # 3. Baseline tilt (atmospheric extinction)
    if config.get('baseline_tilt', True):
        tilt = rng.uniform(-0.05, 0.05)
        x_norm = (wave_aug - wave_aug.min()) / (wave_aug.max() - wave_aug.min() + 1e-12)
        flux_aug *= (1 + tilt * (x_norm - 0.5))

    # 4. Baseline curvature (optical effects)
    if config.get('baseline_curve', True):
        if rng.random() < 0.5:
            curve = rng.uniform(-0.03, 0.03)
            x_norm = (wave_aug - wave_aug.min()) / (wave_aug.max() - wave_aug.min() + 1e-12)
            flux_aug *= (1 + curve * (x_norm - 0.5)**2)

    # 5. Detector noise (photon shot noise, readout noise)
    if config.get('noise', True):
        # SNR between 50 and 200 (typical for astronomical spectra)
        snr = rng.uniform(50, 200)
        noise_std = np.ptp(flux_aug) / snr
        noise = rng.normal(0, noise_std, len(flux_aug))
        # Smooth noise slightly (correlated noise from detector)
        if len(noise) > 7:
            noise = np.convolve(noise, np.ones(7)/7, mode='same')
        flux_aug += noise

    # 6. Resolution degradation (different spectrograph resolution)
    if config.get('resolution_smooth', True):
        if rng.random() < 0.3:  # Only 30% of the time
            kernel_width = rng.choice([3, 5, 7])
            kernel = np.ones(kernel_width) / kernel_width
            flux_aug = np.convolve(flux_aug, kernel, mode='same')

    # 7. Feature depth scaling (composition variation: ±20%)
    # Simulates temporal/spatial variations in atmospheric composition
    if config.get('feature_scaling', True):
        feature_scale = rng.uniform(0.8, 1.2)
        # Apply to deviations from median (preserves baseline, scales features)
        baseline = np.median(flux_aug)
        flux_aug = baseline + (flux_aug - baseline) * feature_scale

    return wave_aug, flux_aug


def create_augmented_dataset(
    wave: np.ndarray,
    flux: np.ndarray,
    n_augmentations: int,
    seed_offset: int = 0
) -> Tuple[list[np.ndarray], list[np.ndarray]]:
    """Create multiple augmented versions of a spectrum.

    Args:
        wave: Original wavelength array
        flux: Original flux array
        n_augmentations: Number of augmented versions to create
        seed_offset: Seed offset for different planets

    Returns:
        Lists of (wavelengths, fluxes) for each augmented version
    """
    waves_aug = []
    fluxes_aug = []

    for i in range(n_augmentations):
        seed = seed_offset * 10000 + i
        wave_aug, flux_aug = augment_spectrum(wave, flux, seed=seed)
        waves_aug.append(wave_aug)
        fluxes_aug.append(flux_aug)

    return waves_aug, fluxes_aug


# Preset configurations for different augmentation strengths
AUGMENTATION_PRESETS = {
    'minimal': {
        'wavelength_shift': True,
        'flux_scaling': True,
        'baseline_tilt': False,
        'baseline_curve': False,
        'noise': True,
        'resolution_smooth': False,
        'feature_scaling': False,
    },
    'standard': {
        'wavelength_shift': True,
        'flux_scaling': True,
        'baseline_tilt': True,
        'baseline_curve': True,
        'noise': True,
        'resolution_smooth': True,
        'feature_scaling': True,
    },
    'aggressive': {
        'wavelength_shift': True,
        'flux_scaling': True,
        'baseline_tilt': True,
        'baseline_curve': True,
        'noise': True,
        'resolution_smooth': True,
        'feature_scaling': True,
    }
}
