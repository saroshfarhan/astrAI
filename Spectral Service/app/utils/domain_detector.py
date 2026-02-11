"""Domain detection for spectral data (UV vs IR).

Detects whether spectral data is UV or IR based on wavelength range.
"""
import numpy as np
from typing import Literal

Domain = Literal["UV", "IR", "UNKNOWN"]


def detect_domain(wavelength: np.ndarray, flux: np.ndarray) -> Domain:
    """Detect if spectrum is UV, IR, or unknown based on wavelength range.

    Args:
        wavelength: Wavelength array (in angstroms or microns, auto-detected)
        flux: Flux array

    Returns:
        "UV", "IR", or "UNKNOWN"

    UV Range: 1000-4000 Å (0.1-0.4 μm)
    IR Range: 0.7-25 μm (7000-250000 Å)
    """
    if len(wavelength) == 0:
        return "UNKNOWN"

    w_min = np.min(wavelength)
    w_max = np.max(wavelength)
    w_median = np.median(wavelength)

    # Auto-detect units (angstroms vs microns)
    if w_median > 100:
        # Likely angstroms (Å)
        # UV: 1000-4000 Å
        # Visible: 4000-7000 Å
        # IR: 7000+ Å

        if w_max < 5000:
            # Mostly UV range
            return "UV"
        elif w_min > 7000:
            # Mostly IR range
            return "IR"
        elif 1000 <= w_median <= 4000:
            # Median in UV
            return "UV"
        elif w_median > 7000:
            # Median in IR
            return "IR"
        else:
            # Mixed or visible - check majority
            uv_coverage = np.sum((wavelength >= 1000) & (wavelength <= 4000))
            ir_coverage = np.sum(wavelength > 7000)

            if uv_coverage > ir_coverage:
                return "UV"
            elif ir_coverage > uv_coverage:
                return "IR"
            else:
                return "UNKNOWN"

    else:
        # Likely microns (μm)
        # UV: 0.1-0.4 μm
        # Visible: 0.4-0.7 μm
        # IR: 0.7-25 μm

        if w_max < 0.5:
            # Mostly UV range
            return "UV"
        elif w_min > 0.7:
            # Mostly IR range
            return "IR"
        elif 0.1 <= w_median <= 0.4:
            # Median in UV
            return "UV"
        elif w_median > 0.7:
            # Median in IR
            return "IR"
        else:
            # Mixed or visible
            uv_coverage = np.sum((wavelength >= 0.1) & (wavelength <= 0.4))
            ir_coverage = np.sum(wavelength > 0.7)

            if uv_coverage > ir_coverage:
                return "UV"
            elif ir_coverage > uv_coverage:
                return "IR"
            else:
                return "UNKNOWN"


def get_wavelength_unit(wavelength: np.ndarray) -> str:
    """Detect if wavelength is in angstroms or microns.

    Args:
        wavelength: Wavelength array

    Returns:
        "angstrom" or "micron"
    """
    w_median = np.median(wavelength)

    if w_median > 100:
        return "angstrom"
    else:
        return "micron"
