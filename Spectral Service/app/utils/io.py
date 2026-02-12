from __future__ import annotations
import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

def load_pkl_spectrum(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # Handle both 'wavelength' and 'wave' keys
    w = np.asarray(obj.get("wavelength", obj.get("wave")), dtype=float)
    flux = np.asarray(obj["flux"], dtype=float)

    m = np.isfinite(w) & np.isfinite(flux)
    w, flux = w[m], flux[m]
    idx = np.argsort(w)
    w, flux = w[idx], flux[idx]
    meta = {"target": obj.get("target", "unknown"), "notes": obj.get("notes", "")}
    return w, flux, meta

def load_fits_spectrum(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    try:
        from astropy.io import fits
    except Exception as e:
        raise RuntimeError("astropy is required to read FITS. pip install astropy") from e

    p = Path(path)
    with fits.open(path) as hdul:
        # Debug: print available HDUs and columns
        debug_info = []
        for i, hdu in enumerate(hdul):
            hdu_type = type(hdu).__name__
            debug_info.append(f"HDU {i}: {hdu_type}")
            if hasattr(hdu, "columns") and hdu.columns:
                cols = [c.name for c in hdu.columns]
                debug_info.append(f"  Columns: {cols}")

        # Try common column names first
        data = None
        for hdu in hdul:
            if hasattr(hdu, "data") and hdu.data is not None:
                data = hdu.data
                break
        if data is None:
            raise ValueError(f"No FITS data found. HDU structure:\n" + "\n".join(debug_info))

        # Extended naming patterns for exoplanet data
        colnames = [c.lower() for c in getattr(data, "columns", []).names] if hasattr(data, "columns") else []
        def pick_col(candidates):
            for c in candidates:
                if c in colnames:
                    return c
            return None

        # More comprehensive column name patterns
        w_col = pick_col(["wavelength", "wave", "lambda", "lam", "wl", "wavel", "freq", "frequency"])
        f_col = pick_col(["flux", "intensity", "spec", "signal", "counts", "net", "gross", "flux_density"])

        if w_col and f_col:
            w = np.asarray(data[w_col], dtype=float)
            flux = np.asarray(data[f_col], dtype=float)
        else:
            # Fallback: try image-like arrays (last resort)
            arr = np.asarray(data, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                w = arr[:, 0]
                flux = arr[:, 1]
            elif arr.ndim == 1:
                # Single spectrum case: generate wavelength indices
                flux = arr
                w = np.arange(len(flux), dtype=float)
            else:
                error_msg = f"Could not infer wavelength/flux columns.\n"
                error_msg += f"Available columns: {colnames}\n"
                error_msg += "\n".join(debug_info)
                raise ValueError(error_msg)

    m = np.isfinite(w) & np.isfinite(flux)
    w, flux = w[m], flux[m]
    idx = np.argsort(w)
    w, flux = w[idx], flux[idx]
    meta = {"target": p.stem, "notes": "loaded from FITS"}
    return w, flux, meta

def load_spectrum(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    ext = Path(path).suffix.lower()
    if ext in [".pkl", ".pickle"]:
        return load_pkl_spectrum(path)
    if ext in [".fits", ".fit", ".fts"]:
        return load_fits_spectrum(path)
    raise ValueError(f"Unsupported file type: {ext}")
