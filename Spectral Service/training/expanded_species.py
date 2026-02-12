"""Expanded species lists for UV and IR spectral analysis.

This module contains significantly expanded element/molecule lists
compared to the original synthetic_generator.py
"""

# ============================================================================
# UV SPECIES (Expanded from 12 to 28)
# ============================================================================

UV_SPECIES_EXPANDED = [
    # Major hydrocarbons (11)
    "CH4",      # Methane
    "C2H2",     # Acetylene
    "C2H4",     # Ethylene
    "C2H6",     # Ethane
    "C3H4",     # Propyne
    "C3H8",     # Propane
    "C4H2",     # Diacetylene
    "C4H10",    # Butane
    "C6H6",     # Benzene
    "C7H8",     # Toluene
    "C8H10",    # Xylene

    # Nitrogen compounds (6)
    "NH3",      # Ammonia
    "HCN",      # Hydrogen cyanide
    "N2O",      # Nitrous oxide
    "NO",       # Nitric oxide
    "NO2",      # Nitrogen dioxide
    "N2",       # Nitrogen

    # Oxygen compounds (6)
    "CO",       # Carbon monoxide
    "CO2",      # Carbon dioxide
    "H2O",      # Water
    "O2",       # Oxygen
    "O3",       # Ozone
    "SO2",      # Sulfur dioxide

    # Sulfur compounds (2)
    "H2S",      # Hydrogen sulfide
    "OCS",      # Carbonyl sulfide

    # Others (3)
    "PH3",      # Phosphine
    "GeH4",     # Germane
    "AsH3",     # Arsine
]

# ============================================================================
# IR SPECIES (Expanded from 6 to 22)
# ============================================================================

IR_SPECIES_EXPANDED = [
    # Primary IR active (10)
    "CH4",      # Methane (strong IR)
    "NH3",      # Ammonia (strong IR)
    "H2O",      # Water (strong IR)
    "CO2",      # Carbon dioxide (strong IR)
    "CO",       # Carbon monoxide
    "C2H6",     # Ethane
    "C2H2",     # Acetylene
    "PH3",      # Phosphine
    "SO2",      # Sulfur dioxide
    "H2S",      # Hydrogen sulfide

    # Secondary IR active (12)
    "C3H8",     # Propane
    "C4H10",    # Butane
    "HCN",      # Hydrogen cyanide
    "N2O",      # Nitrous oxide
    "NO2",      # Nitrogen dioxide
    "O3",       # Ozone
    "OCS",      # Carbonyl sulfide
    "HCl",      # Hydrogen chloride
    "HF",       # Hydrogen fluoride
    "C6H6",     # Benzene
    "C2H4",     # Ethylene
    "C4H2",     # Diacetylene
]

# ============================================================================
# PLANET COMPOSITION LABELS
# Based on literature and observations
# These are binary labels (1 = present, 0 = absent/unconfirmed)
# ============================================================================

PLANET_COMPOSITIONS_UV = {
    "JUPITER": {
        # Hydrocarbons
        "CH4": 1, "C2H2": 1, "C2H4": 1, "C2H6": 1, "C3H4": 1, "C6H6": 1,
        # Nitrogen compounds
        "NH3": 1, "HCN": 1,
        # Oxygen compounds
        "CO": 1, "CO2": 1, "H2O": 1,
        # Sulfur compounds
        "H2S": 1, "OCS": 1,
        # Others
        "PH3": 1, "GeH4": 1, "AsH3": 1,
        # Source: Galileo/Cassini/HST/JWST
    },
    "SATURN": {
        # Hydrocarbons
        "CH4": 1, "C2H2": 1, "C2H6": 1, "C3H4": 1, "C3H8": 1, "C4H2": 1, "C6H6": 1,
        # Nitrogen compounds
        "NH3": 1, "HCN": 1,
        # Oxygen compounds
        "CO": 1, "CO2": 1, "H2O": 1,
        # Sulfur compounds
        "H2S": 1,
        # Others
        "PH3": 1, "GeH4": 1, "AsH3": 1,
        # Source: Cassini/CIRS/HST
    },
    "URANUS": {
        # Hydrocarbons
        "CH4": 1, "C2H2": 1, "C2H6": 1,
        # Nitrogen compounds
        "NH3": 1, "HCN": 1,
        # Oxygen compounds
        "CO": 1, "CO2": 1, "H2O": 1,
        # Sulfur compounds
        "H2S": 1,
        # Source: Voyager/Gemini-NIFS/HST
    },
    "NEPTUNE": {
        # Hydrocarbons
        "CH4": 1, "C2H2": 1, "C2H4": 1, "C2H6": 1, "C3H4": 1, "C4H2": 1,
        # Nitrogen compounds
        "NH3": 1, "HCN": 1,
        # Oxygen compounds
        "CO": 1, "CO2": 1, "H2O": 1,
        # Sulfur compounds
        "H2S": 1,
        # Source: Voyager/HST/JWST
    },
    "MARS": {
        # Hydrocarbons (controversial/trace)
        "CH4": 1,  # Curiosity detected; TGO disputed
        # Nitrogen compounds
        "NO": 1, "N2": 1,
        # Oxygen compounds
        "CO": 1, "CO2": 1, "H2O": 1, "O2": 1, "O3": 1,
        # Source: Curiosity/TGO/MRO
    },
    "VENUS": {
        # Nitrogen compounds
        "NO": 1, "N2": 1,
        # Oxygen compounds
        "CO": 1, "CO2": 1, "H2O": 1, "O3": 1, "SO2": 1,
        # Sulfur compounds
        "H2S": 1, "OCS": 1,
        # Source: Venus Express/Akatsuki/ground-based
    },
}

PLANET_COMPOSITIONS_IR = {
    "JUPITER": {
        # Primary IR active
        "CH4": 1, "NH3": 1, "H2O": 1, "CO2": 1, "CO": 1,
        "C2H6": 1, "C2H2": 1, "PH3": 1, "H2S": 1,
        # Secondary IR active
        "HCN": 1, "C6H6": 1, "C2H4": 1,
        # Source: Galileo/Cassini/JWST
    },
    "SATURN": {
        # Primary IR active
        "CH4": 1, "NH3": 1, "H2O": 1, "CO2": 1, "CO": 1,
        "C2H6": 1, "C2H2": 1, "PH3": 1, "H2S": 1,
        # Secondary IR active
        "C3H8": 1, "HCN": 1, "C6H6": 1, "C4H2": 1,
        # Source: Cassini/CIRS
    },
    "URANUS": {
        # Primary IR active
        "CH4": 1, "NH3": 1, "H2O": 1, "CO2": 1, "CO": 1,
        "C2H6": 1, "C2H2": 1, "H2S": 1,
        # Secondary IR active
        "HCN": 1,
        # Source: Voyager/Gemini-NIFS
    },
    "NEPTUNE": {
        # Primary IR active
        "CH4": 1, "NH3": 1, "H2O": 1, "CO2": 1, "CO": 1,
        "C2H6": 1, "C2H2": 1, "H2S": 1,
        # Secondary IR active
        "HCN": 1, "C2H4": 1, "C4H2": 1,
        # Source: Voyager/HST
    },
    "MARS": {
        # Primary IR active
        "CH4": 1, "H2O": 1, "CO2": 1, "CO": 1,
        # Secondary IR active
        "O3": 1,
        # Source: Curiosity/TGO
    },
    "VENUS": {
        # Primary IR active
        "H2O": 1, "CO2": 1, "CO": 1, "SO2": 1, "H2S": 1,
        # Secondary IR active
        "O3": 1, "OCS": 1, "HCl": 1, "HF": 1,
        # Source: Venus Express/Akatsuki
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_labels_for_planet(planet_name: str, species_list: list, domain: str = "UV") -> list:
    """Get binary labels for a planet.

    Args:
        planet_name: Name of planet (e.g., "JUPITER")
        species_list: List of species to get labels for
        domain: "UV" or "IR"

    Returns:
        Binary label array (1 = present, 0 = absent)
    """
    compositions = PLANET_COMPOSITIONS_UV if domain == "UV" else PLANET_COMPOSITIONS_IR

    planet_comp = compositions.get(planet_name.upper(), {})

    labels = []
    for species in species_list:
        labels.append(planet_comp.get(species, 0))

    return labels


def expand_labels_with_unknowns(planet_name: str, species_list: list, domain: str = "UV") -> list:
    """Get labels with conservative assumption for unknowns.

    For unconfirmed species, assumes 30% probability of presence.

    Args:
        planet_name: Name of planet
        species_list: List of species
        domain: "UV" or "IR"

    Returns:
        Binary label array with probabilistic unknowns
    """
    import numpy as np

    compositions = PLANET_COMPOSITIONS_UV if domain == "UV" else PLANET_COMPOSITIONS_IR
    planet_comp = compositions.get(planet_name.upper(), {})

    rng = np.random.default_rng(hash(planet_name) % 2**32)

    labels = []
    for species in species_list:
        if species in planet_comp:
            labels.append(planet_comp[species])
        else:
            # Unknown - assume 30% probability of trace presence
            labels.append(1 if rng.random() < 0.3 else 0)

    return labels
