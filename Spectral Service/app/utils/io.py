from __future__ import annotations
import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

def load_pkl_spectrum(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # Handle structured numpy array (numpy.record) format - Earth PKL files
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        # Extract field names
        field_names = [name.lower() for name in obj.dtype.names]

        # Find wavelength field
        wl_candidates = ["wavelength", "wavelength_um", "wavelength_nm", "wave", "wl", "lambda"]
        wl_field = next((name for name in obj.dtype.names if name.lower() in wl_candidates), None)

        # Find flux field
        fx_candidates = ["flux", "radiance", "radiance_final", "intensity", "reflectance"]
        fx_field = next((name for name in obj.dtype.names if name.lower() in fx_candidates), None)

        if wl_field and fx_field:
            w = np.asarray(obj[wl_field], dtype=float)
            flux = np.asarray(obj[fx_field], dtype=float)
        else:
            raise ValueError(f"Cannot detect wavelength/flux fields in structured array: {obj.dtype.names}")

    # Handle dictionary format
    elif isinstance(obj, dict):
        # Try wavelength variations
        wl_candidates = ["wavelength", "wavelength_um", "wave", "wl"]
        w_key = next((k for k in wl_candidates if k in obj), None)

        # Try flux variations
        fx_candidates = ["flux", "radiance", "radiance_final", "intensity"]
        f_key = next((k for k in fx_candidates if k in obj), None)

        if w_key and f_key:
            w = np.asarray(obj[w_key], dtype=float)
            flux = np.asarray(obj[f_key], dtype=float)
        else:
            raise ValueError(f"Cannot detect wavelength/flux keys in dict: {list(obj.keys())}")

    else:
        raise ValueError(f"Unsupported PKL format: {type(obj)}")

    m = np.isfinite(w) & np.isfinite(flux)
    w, flux = w[m], flux[m]
    idx = np.argsort(w)
    w, flux = w[idx], flux[idx]
    meta = {"target": "unknown", "notes": "loaded from PKL"}
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
