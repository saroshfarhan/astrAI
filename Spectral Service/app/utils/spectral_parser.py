"""Universal Spectral Parser - Extract meaningful features from spectral data for LLM visibility.

This module bridges the gap between raw spectral arrays and LLM understanding by:
1. Identifying key absorption/emission features
2. Finding peaks and valleys in scientifically relevant wavelength ranges
3. Generating human-readable summaries of spectral characteristics

This gives the LLM "eyes" to see actual spectral patterns instead of just filenames.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.signal import find_peaks, savgol_filter


# Known molecular absorption windows (in micrometers)
MOLECULAR_WINDOWS = {
    "CO2": [(4.2, 4.4), (14.0, 16.0)],  # Strong absorption
    "H2O": [(2.5, 3.0), (5.5, 7.5)],     # Water vapor bands
    "CH4": [(3.2, 3.5), (7.5, 8.0)],     # Methane
    "O3": [(9.4, 9.8)],                   # Ozone
    "N2O": [(4.4, 4.6), (7.5, 8.5)],     # Nitrous oxide
    "CO": [(4.6, 4.8)],                   # Carbon monoxide
    "NH3": [(9.0, 11.0)],                 # Ammonia
    "H2S": [(3.8, 4.0)],                  # Hydrogen sulfide
}


def generate_spectral_summary(
    wavelength: np.ndarray,
    flux: np.ndarray,
    top_features: int = 5
) -> Dict[str, Any]:
    """Generate comprehensive spectral summary for LLM consumption.

    Args:
        wavelength: Wavelength array (in micrometers or nanometers)
        flux: Flux/radiance array
        top_features: Number of top features to report

    Returns:
        Dictionary with spectral characteristics in natural language
    """
    # Ensure sorted
    idx = np.argsort(wavelength)
    wl = wavelength[idx]
    fl = flux[idx]

    # Detect wavelength unit
    wl_unit = "nm" if np.median(wl) > 100 else "μm"
    if wl_unit == "nm":
        wl = wl / 1000.0  # Convert to micrometers for consistency

    # Calculate basic statistics
    wl_min, wl_max = wl.min(), wl.max()
    fl_min, fl_max = fl.min(), fl.max()
    fl_median = np.median(fl)
    fl_std = np.std(fl)

    # Detect domain
    if wl_max < 1.0:
        domain = "UV/Visible"
    elif wl_min > 2.0 and wl_max > 5.0:
        domain = "Infrared"
    else:
        domain = "Mixed UV-IR"

    # Find peaks and valleys
    peaks, valleys = find_spectral_features(wl, fl)

    # Check molecular absorption windows
    molecular_signatures = detect_molecular_signatures(wl, fl)

    # Generate natural language summary
    summary = {
        "domain": domain,
        "wavelength_range": f"{wl_min:.2f}-{wl_max:.2f} μm",
        "flux_statistics": {
            "min": float(fl_min),
            "max": float(fl_max),
            "median": float(fl_median),
            "std": float(fl_std),
            "dynamic_range": float(fl_max / (fl_min + 1e-12))
        },
        "peaks": peaks[:top_features],
        "valleys": valleys[:top_features],
        "molecular_signatures": molecular_signatures,
        "natural_language": generate_natural_language_description(
            domain, wl_min, wl_max, peaks, valleys, molecular_signatures
        )
    }

    return summary


def find_spectral_features(
    wavelength: np.ndarray,
    flux: np.ndarray,
    prominence_threshold: float = 0.05
) -> Tuple[List[Dict], List[Dict]]:
    """Find significant peaks and valleys in the spectrum.

    Args:
        wavelength: Wavelength array
        flux: Flux array
        prominence_threshold: Minimum prominence as fraction of flux range

    Returns:
        (peaks, valleys) - Lists of feature dictionaries
    """
    # Smooth the spectrum for robust feature detection
    if len(flux) > 10:
        smoothed = savgol_filter(flux, window_length=min(11, len(flux) // 2 * 2 + 1), polyorder=3)
    else:
        smoothed = flux

    flux_range = flux.max() - flux.min()
    min_prominence = prominence_threshold * flux_range

    # Find peaks (emission features)
    peak_idx, peak_props = find_peaks(smoothed, prominence=min_prominence)
    peaks = []
    for i, idx in enumerate(peak_idx):
        peaks.append({
            "wavelength": float(wavelength[idx]),
            "flux": float(flux[idx]),
            "prominence": float(peak_props["prominences"][i]),
            "type": "emission"
        })

    # Find valleys (absorption features) - invert and find peaks
    valley_idx, valley_props = find_peaks(-smoothed, prominence=min_prominence)
    valleys = []
    for i, idx in enumerate(valley_idx):
        valleys.append({
            "wavelength": float(wavelength[idx]),
            "flux": float(flux[idx]),
            "depth": float(valley_props["prominences"][i]),
            "type": "absorption"
        })

    # Sort by prominence/depth
    peaks = sorted(peaks, key=lambda x: x["prominence"], reverse=True)
    valleys = sorted(valleys, key=lambda x: x["depth"], reverse=True)

    return peaks, valleys


def detect_molecular_signatures(
    wavelength: np.ndarray,
    flux: np.ndarray
) -> List[Dict[str, Any]]:
    """Check for known molecular absorption signatures.

    Args:
        wavelength: Wavelength array (in micrometers)
        flux: Flux array

    Returns:
        List of detected molecular signatures with confidence scores
    """
    signatures = []

    for molecule, windows in MOLECULAR_WINDOWS.items():
        for wl_min, wl_max in windows:
            # Check if we have data in this window
            in_window = (wavelength >= wl_min) & (wavelength <= wl_max)
            if not np.any(in_window):
                continue

            # Extract window data
            wl_window = wavelength[in_window]
            fl_window = flux[in_window]

            if len(fl_window) < 3:
                continue

            # Calculate absorption depth in this window
            baseline = np.percentile(fl_window, 90)  # Upper envelope
            min_flux = fl_window.min()
            depth = (baseline - min_flux) / baseline if baseline > 0 else 0

            # Confidence based on depth and window coverage
            if depth > 0.05:  # At least 5% absorption
                confidence = min(depth * 2, 1.0)  # Cap at 1.0

                signatures.append({
                    "molecule": molecule,
                    "wavelength_window": f"{wl_min:.1f}-{wl_max:.1f} μm",
                    "depth": float(depth),
                    "confidence": float(confidence),
                    "min_wavelength": float(wl_window[fl_window.argmin()])
                })

    # Sort by confidence
    signatures = sorted(signatures, key=lambda x: x["confidence"], reverse=True)

    return signatures


def generate_natural_language_description(
    domain: str,
    wl_min: float,
    wl_max: float,
    peaks: List[Dict],
    valleys: List[Dict],
    molecular_signatures: List[Dict]
) -> str:
    """Generate human-readable description of spectral features.

    This is the key output that gets fed to the LLM.

    Args:
        domain: Spectral domain (UV/IR/Mixed)
        wl_min, wl_max: Wavelength range
        peaks: List of emission peaks
        valleys: List of absorption valleys
        molecular_signatures: Detected molecular features

    Returns:
        Natural language description string
    """
    parts = []

    # Domain and range
    parts.append(f"DATA_OBSERVATION: Spectrum covers {domain} range ({wl_min:.2f}-{wl_max:.2f} μm).")

    # Molecular signatures (most important)
    if molecular_signatures:
        top_molecules = molecular_signatures[:3]
        mol_str = ", ".join([
            f"{sig['molecule']} (depth: {sig['depth']:.1%}, at {sig['min_wavelength']:.2f} μm)"
            for sig in top_molecules
        ])
        parts.append(f"STRONG ABSORPTION FEATURES: {mol_str}.")

    # Valleys (absorption lines)
    if valleys:
        top_valleys = valleys[:3]
        valley_str = ", ".join([
            f"{v['wavelength']:.2f} μm (depth: {v['depth']:.2f})"
            for v in top_valleys
        ])
        parts.append(f"ABSORPTION VALLEYS detected at: {valley_str}.")

    # Peaks (emission lines)
    if peaks:
        top_peaks = peaks[:3]
        peak_str = ", ".join([
            f"{p['wavelength']:.2f} μm (prominence: {p['prominence']:.2f})"
            for p in top_peaks
        ])
        parts.append(f"EMISSION PEAKS detected at: {peak_str}.")

    # Summary statement
    feature_count = len(molecular_signatures) + len(peaks) + len(valleys)
    parts.append(f"TOTAL FEATURES: {feature_count} significant spectral features identified.")

    return " ".join(parts)


def summarize_for_llm(wavelength: np.ndarray, flux: np.ndarray) -> str:
    """Convenience function to get just the natural language summary.

    This is the main function to call from agents.

    Args:
        wavelength: Wavelength array
        flux: Flux array

    Returns:
        Natural language description string for LLM consumption
    """
    summary = generate_spectral_summary(wavelength, flux)
    return summary["natural_language"]
