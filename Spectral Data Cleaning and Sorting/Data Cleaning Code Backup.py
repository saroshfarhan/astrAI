pip install git+https://github.com/astropy/astroquery.git

from astroquery.mast import Observations

# Search for Hubble observations of Jupiter with spectroscopic data
obs_table = Observations.query_criteria(
    target_name="JUPITER", 
    dataproduct_type="spectrum", 
    instrument_name="WFC3/UVIS"
)
# Download the calibrated science products
Observations.download_products(obs_table[0])

pip install astropy matplotlib requests

import os
from astroquery.mast import Observations

TARGET = "NEPTUNE"
OUTPUT_DIR = r"E:\DataFecked Hackathon\Neptune_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_rich_neptune():
    print(f"🚀 HUNTING FOR NEPTUNE COMPOSITE DATA...")
    
    # Query for Neptune spectra on HST
    obs_table = Observations.query_criteria(
        target_name=TARGET + "*",
        dataproduct_type="spectrum",
        obs_collection="HST",
        instrument_name="STIS*" 
    )
    
    # We want G140L (covers ~1150-1700 A) AND G230L (covers ~1700-3100 A)
    # This combination gives us the widest possible "Rich" range.
    wide_gratings = ["G140L", "G230L"]
    
    for grating in wide_gratings:
        print(f"🔍 Searching for {grating}...")
        matches = [row for row in obs_table if grating in str(row['filters'])]
        
        if matches:
            # Sort by exposure time to get the highest signal-to-noise
            matches.sort(key=lambda x: x['t_exptime'])
            best_obs = matches[-1] 
            
            print(f"✅ Found {grating}! ObsID: {best_obs['obsid']} | Time: {best_obs['t_exptime']}s")
            
            products = Observations.get_product_list(best_obs)
            # Filter for the calibrated 1D spectrum (x1d)
            filtered = Observations.filter_products(products, productSubGroupDescription="X1D")
            
            if len(filtered) > 0:
                Observations.download_products(filtered, download_dir=OUTPUT_DIR)
                print(f"📂 {grating} data downloaded successfully.")
        else:
            print(f"⚠️ No {grating} found. Neptune might be hiding that day.")

fetch_rich_neptune()

def fetch_neptune_fuv_surrogate():
    print(f"📡 SEARCHING FOR NEPTUNE FUV (G140L) SURROGATE...")
    
    obs_table = Observations.query_criteria(
        target_name="NEPTUNE*",
        dataproduct_type="spectrum",
        obs_collection="HST",
        filters="G140L" # Specifically looking for the Far-UV
    )
    
    if len(obs_table) > 0:
        # Sort by exposure time to get the cleanest signal
        obs_table.sort("t_exptime")
        best_fuv = obs_table[-1]
        print(f"✅ Found FUV Surrogate! ObsID: {best_fuv['obsid']} | Date: {best_fuv['t_min']}")
        
        products = Observations.get_product_list(best_fuv)
        filtered = Observations.filter_products(products, productSubGroupDescription="X1D")
        
        Observations.download_products(filtered[:1], download_dir=OUTPUT_DIR)
        print(f"📂 FUV Surrogate secured.")
    else:
        print("❌ Still no FUV found. Neptune is being very shy.")

fetch_neptune_fuv_surrogate()

import os
import numpy as np
from astroquery.mast import Observations

# --- CONFIGURATION ---
TARGETS = ["MARS"] 
# Ensure this points to your specific Mars directory
OUTPUT_DIR = r"E:\DataFecked Hackathon\Mars_Full_Profile" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_mars_full_profile(target_name):
    print(f"\n🚀 MISSION: MARS FULL PROFILE SEARCH")
    
    # 1. Broad Query for STIS on Mars
    obs_table = Observations.query_criteria(
        target_name=target_name + "*",
        dataproduct_type="spectrum",
        obs_collection="HST",
        instrument_name="STIS*" 
    )
    
    if len(obs_table) == 0:
        print("❌ No STIS data found for Mars.")
        return False
        
    # 2. SELECTION LOGIC: Hunting for the NUV (G230L)
    # We want G230L specifically because it covers the widest range (1700-3150 A)
    nuv_candidates = [row for row in obs_table if "G230L" in str(row['filters'])]
    
    print(f"📊 Market Report: Found {len(nuv_candidates)} NUV (G230L) candidates.")

    if not nuv_candidates:
        print("⚠️ No G230L found, looking for any 230-series (NUV) filter...")
        nuv_candidates = [row for row in obs_table if "230" in str(row['filters'])]

    # Sort by exposure time to get the clearest signal (Mars is bright, but SNR helps)
    nuv_candidates.sort(key=lambda x: x['t_exptime'])
    
    # Take the top candidate
    best_nuv = nuv_candidates[-1]
    
    print(f"🔍 Selected ObsID: {best_nuv['obsid']} | Filter: {best_nuv['filters']} | ExpTime: {best_nuv['t_exptime']}s")
    
    # ... (inside your fetch_mars_full_profile function, replace step 3) ...

    # 3. Flexible Product Logic
    data_products = Observations.get_product_list(best_nuv)
    
    # List of product types that contain the 1D spectrum we need
    potential_types = ["X1D", "S1D", "SX1", "X1DINTS"]
    
    download_candidates = None
    for p_type in potential_types:
        filtered = Observations.filter_products(data_products, productSubGroupDescription=p_type)
        if len(filtered) > 0:
            download_candidates = filtered
            print(f"🎯 Found {p_type} for Mars NUV!")
            break
            
    if download_candidates is not None:
        Observations.download_products(download_candidates[:1], download_dir=OUTPUT_DIR)
        print(f"🎉 MARS NUV Profile secured.")
    else:
        # If still nothing, let's print what IS there so we can debug
        print("❌ Still no 1D spectrum found. Available products are:")
        print(np.unique(data_products['productSubGroupDescription']))
        
    return True

for target in TARGETS:
    fetch_mars_full_profile(target)

import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter
import pickle
import os

# --- UNIVERSAL SPECTRAL FEATURE DICTIONARY ---
PLANETARY_SIGNATURES = {
    "EMISSION": {
        "H I (Lyman-alpha)": 1215.67,
        "O I (Oxygen)": 1304.86,
        "O I (Forbidden)": 1356.0,
        "N I (Nitrogen)": 1493.0,
        "C I (Carbon)": 1657.0,
        "Mg I (Magnesium)": 2852.0,
    },
    "ABSORPTION": {
        "CH4 (Methane)": (1400, 1600),
        "C2H2 (Acetylene)": (1700, 1900),
        "O3 (Ozone)": (2400, 2600),
        "SO2 (Sulfur Dioxide)": (2000, 2300)
    }
}

PLANETARY_SIGNATURES["EMISSION"].update({
    "CO Cameron Bands": 2100.0,  # Center of the broad emission forest
    "CO Fourth Positive": 1548.0, # Secondary CO marker
})

PLANETARY_SIGNATURES["ABSORPTION"].update({
    "O3 (Ozone)": (2400, 2600),   # Martian UV shield
    "Ice Haze": (2000, 2400)      # Dust and CO2 ice clouds
})


def identify_elements(wave, flux, planet_type="GAS_GIANT", threshold=1.2):
    print(f"\n--- 🧪 CHEMICAL AUDIT ({planet_type}) ---")
    w_min, w_max = np.min(wave), np.max(wave)
    print(f"DEBUG: Analyzing slice {w_min:.1f} - {w_max:.1f} Å")

    # --- 1. SCAN FOR EMISSION SPIKES (Works for ALL planets) ---
    for name, line_wave in PLANETARY_SIGNATURES["EMISSION"].items():
        if w_min < line_wave < w_max:
            # Find the closest pixel to the theoretical line
            idx = (np.abs(wave - line_wave)).argmin()
            
            # Look at a 20-pixel local neighborhood to see if it's a "spike"
            local_window = flux[max(0, idx-10):min(len(flux), idx+10)]
            local_avg = np.mean(local_window)
            
            if flux[idx] > local_avg * threshold:
                print(f"✨ DETECTED EMISSION: {name} at {wave[idx]:.1f} Å")
            else:
                print(f"ℹ️ {name} emission line in range, but no spike detected.")

    # --- 2. SCAN FOR ABSORPTION BANDS (Giants only) ---
    if planet_type in ["GAS_GIANT", "ICE_GIANT"]:
        for name, (low, high) in PLANETARY_SIGNATURES["ABSORPTION"].items():
            intersect_min = max(w_min, low)
            intersect_max = min(w_max, high)
            
            if (intersect_max - intersect_min) > 50:
                mask = (wave >= intersect_min) & (wave <= intersect_max)
                avg_zone = np.mean(flux[mask])
                
                if avg_zone < np.mean(flux) * (1/threshold):
                    print(f"☁️ DETECTED ABSORPTION: {name} (Visible: {intersect_min:.0f}-{intersect_max:.0f} Å)")
                else:
                    print(f"ℹ️ {name} absorption range visible, but signal strength too high.")
                
def process_universal_raw(file_paths, planet_name, planet_type, threshold=1.2):
    """Stitches all file data together so the Master PKL isn't just the last file."""
    print(f"\n--- 🛰️ PROCESSING: {planet_name.upper()} ---")
    
    all_waves = []
    all_fluxes = []
    
    for f_path in file_paths:
        with fits.open(f_path) as hdul:
            data = hdul[1].data
            w_raw = data['WAVELENGTH'].flatten()
            f_raw = data['FLUX'].flatten()
            
            # Sort individual file
            s_idx = np.argsort(w_raw)
            w, f = w_raw[s_idx], f_raw[s_idx]
            
            # Clean NaNs
            mask = ~np.isnan(f)
            all_waves.append(w[mask])
            all_fluxes.append(f[mask])

            # Run Audit on this specific slice
            identify_elements(w[mask], f[mask], planet_type=planet_type, threshold=threshold)

    # --- STITCHING STEP ---
    # Combine all pieces into one master array
    master_wave = np.concatenate(all_waves)
    master_flux = np.concatenate(all_fluxes)
    
    # Final Sort (Crucial because File 2 might come before File 1 in wavelength)
    final_idx = np.argsort(master_wave)
    master_wave = master_wave[final_idx]
    master_flux = master_flux[final_idx]

    # Save the FULL range to the PKL
    out_file = f"{planet_name.upper()}_RAW_MASTER.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump({
            "wave": master_wave, 
            "flux": master_flux, 
            "target": planet_name, 
            "type": planet_type
        }, f)
    
    print(f"✅ STITCHED SUCCESS: {out_file} created with range {master_wave[0]:.1f} - {master_wave[-1]:.1f} Å")
    return out_file
    
# --- EXECUTION ---
#uranus_files = [r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\o65ha30a0\o65ha30a0_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\o65ha30b0\o65ha30b0_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\o65ha30c0\o65ha30c0_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\o65ha30d0\o65ha30d0_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\o65ha30e0\o65ha30e0_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\o65ha3010\o65ha3010_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\o65ha3020\o65ha3020_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\o65ha3030\o65ha3030_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\o65ha3040\o65ha3040_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\o65ha3050\o65ha3050_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\o65ha3060\o65ha3060_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\o65ha3070\o65ha3070_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\o65ha3080\o65ha3080_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\o65ha3090\o65ha3090_x1d.fits"]
#saturn_files = [r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Saturn - mastDownload\UV data\o65ha2010\o65ha2010_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Saturn - mastDownload\UV data\o65ha2020\o65ha2020_x1d.fits"]
#venus_files = [r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Venus - mastDownload\UV data\o6bbb1020\o6bbb1020_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Venus - mastDownload\UV data\obo0c1010\obo0c1010_x1d.fits"]
mars_files = [r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Mars - mastDownload\UV data\ocjhd3010\ocjhd3010_x1d.fits",r"E:\DataFecked Hackathon\Mars_Full_Profile\mastDownload\HST\o43va4010\o43va4010_sx1.fits"]
#neptune_files = [r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune_Data\mastDownload\HST\o65ha50a0\o65ha50a0_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune_Data\mastDownload\HST\o65ha50b0\o65ha50b0_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune_Data\mastDownload\HST\o65ha50c0\o65ha50c0_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune_Data\mastDownload\HST\o65ha50d0\o65ha50d0_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune_Data\mastDownload\HST\o65ha50e0\o65ha50e0_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune_Data\mastDownload\HST\o65ha5010\o65ha5010_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune_Data\mastDownload\HST\o65ha5020\o65ha5020_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune_Data\mastDownload\HST\o65ha5030\o65ha5030_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune_Data\mastDownload\HST\o65ha5040\o65ha5040_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune_Data\mastDownload\HST\o65ha5050\o65ha5050_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune_Data\mastDownload\HST\o65ha5060\o65ha5060_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune_Data\mastDownload\HST\o65ha5070\o65ha5070_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune_Data\mastDownload\HST\o65ha5080\o65ha5080_x1d.fits",r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune_Data\mastDownload\HST\o65ha5090\o65ha5090_x1d.fits"]
process_universal_raw(mars_files, "MARS", "TERRESTIAL",threshold=1.2)



import pickle
import numpy as np

# 1. Load the actual data
neptune_path = r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\NEPTUNE_RAW_MASTER.pkl"
with open(neptune_path, "rb") as f:
    neptune_data = pickle.load(f)

# 2. Extract the arrays from the dictionary
wave = neptune_data["wave"]
flux = neptune_data["flux"]

print(f"🔵 NEPTUNE AUDIT RANGE: {np.min(wave):.1f} - {np.max(wave):.1f} Å")

# 3. Run the audit using the ACTUAL extracted variables
# We use threshold=1.15 to be sensitive to Acetylene (C2H2)
identify_elements(wave, flux, planet_type="GAS_GIANT", threshold=1.15)

import pickle
import pandas as pd
import numpy as np

# 1. Map your Final Masters
FINAL_PLANETS = {
    "VENUS": r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Venus - mastDownload\UV data\VENUS_RAW_MASTER.pkl",
    "MARS": r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Mars - mastDownload\UV data\MARS_RAW_MASTER.pkl",
    "JUPITER": r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Jupiter - mastDownload\UV data\JUPITER_RAW_MASTER.pkl",
    "SATURN": r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Saturn - mastDownload\UV data\SATURN_RAW_MASTER.pkl",
    "URANUS": r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\URANUS_RAW_MASTER.pkl",
    "NEPTUNE": r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune - mastDownload\UV data\NEPTUNE_RAW_MASTER.pkl"
}

summary_results = []

for name, path in FINAL_PLANETS.items():
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        summary_results.append({
            "Planet": name,
            "Wavelength_Range": f"{np.min(data['wave']):.0f}-{np.max(data['wave']):.0f} Å",
            "Mean_Flux": f"{np.mean(data['flux']):.2e}",
            "Status": "✅ AUDITED"
        })
    except FileNotFoundError:
        summary_results.append({"Planet": name, "Status": "❌ MISSING"})

# 2. Create the Master CSV for your presentation
df_final = pd.DataFrame(summary_results)
df_final.to_csv(r"E:\DataFecked Hackathon\SOLAR_SYSTEM_UV_CATALOG.csv", index=False)

print("🏆 THE UV CATALOG IS COMPLETE 🏆")
print(df_final)

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. SETUP: Define your Master Files ---
# Make sure these paths match exactly where your .pkl files are

PLANET_FILES = {
    "VENUS": r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Venus - mastDownload\UV data\VENUS_RAW_MASTER.pkl",
    "MARS": r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Mars - mastDownload\UV data\MARS_RAW_MASTER.pkl",
    "JUPITER": r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Jupiter - mastDownload\UV data\JUPITER_RAW_MASTER.pkl",
    "SATURN": r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Saturn - mastDownload\UV data\SATURN_RAW_MASTER.pkl",
    "URANUS": r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Uranus - mastDownload\UV data\URANUS_RAW_MASTER.pkl",
    "NEPTUNE": r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\Neptune - mastDownload\UV data\NEPTUNE_RAW_MASTER.pkl"
}


OUTPUT_DIR = r"E:\DataFecked Hackathon"
all_data_frames = []

# --- 2. PLOTTING SETUP ---
plt.figure(figsize=(14, 12))
plt.style.use('dark_background') # Looks professional for space data

offset = 0 # This pushes each planet up the y-axis
spacing = 2 # How much space between planets

print("🚀 STARTING FINAL MERGE & PLOT GENERATION...\n")

# --- 3. THE LOOP: Process Each Planet ---
# We reverse the list so Venus is at the bottom (closest to Sun) and Neptune at top
for planet_name, file_path in reversed(PLANET_FILES.items()):
    try:
        # A. Load the Data
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        wave = data["wave"]
        flux = data["flux"]
        
        # B. Normalize Data (Crucial for Visualization)
        # We divide by the median so huge Jupiter doesn't drown out tiny Neptune
        # We add a small constant (1e-18) to avoid divide-by-zero errors
        norm_flux = flux / (np.nanmedian(flux) + 1e-18)
        
        # C. Add to Plot
        # We shift the wavelength slightly for visual alignment if needed, but here we plot raw
        plt.plot(wave, norm_flux + offset, label=planet_name, linewidth=1.2, alpha=0.9)
        
        # Add a text label next to the line
        plt.text(np.min(wave) - 100, 1 + offset, planet_name, 
                 fontsize=12, fontweight='bold', color='white', ha='right')
        
        # D. Prepare Data for CSV Export (The LLM Part)
        # We create a mini-dataframe for this planet
        df_planet = pd.DataFrame({
            "Planet": planet_name,
            "Wavelength_A": wave,
            "Flux_Raw": flux,
            "Flux_Normalized": norm_flux
        })
        all_data_frames.append(df_planet)
        
        print(f"✅ Processed {planet_name}: {len(wave)} points.")
        
        # Increase offset for the next planet
        offset += spacing

    except FileNotFoundError:
        print(f"❌ MISSING: Could not find file for {planet_name}")
    except Exception as e:
        print(f"⚠️ ERROR on {planet_name}: {e}")

# --- 4. FINALIZE THE PLOT ---
plt.title("The Solar System's Ultraviolet Fingerprint", fontsize=18, color='white', pad=20)
plt.xlabel("Wavelength (Å)", fontsize=14, color='white')
plt.ylabel("Relative Flux (Offset for Clarity)", fontsize=14, color='white')
plt.xlim(1000, 3200) # Trim the x-axis to the interesting region
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper right', frameon=False)

# Save the plot
plot_path = os.path.join(OUTPUT_DIR, "FINAL_WATERFALL_PLOT.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n🖼️  PLOT SAVED: {plot_path}")
plt.show()

# --- 5. SAVE THE MERGED DATA FOR AI ---
# Combine all mini-dataframes into one massive Master Log
master_df = pd.concat(all_data_frames, ignore_index=True)

# Save as CSV
csv_path = os.path.join(OUTPUT_DIR, "SOLAR_SYSTEM_SPECTRA_MERGED.csv")
master_df.to_csv(csv_path, index=False)

print(f"📦 DATA MERGED: {csv_path}")
print(f"📊 Total Data Points: {len(master_df)}")
print("🎉 READY FOR LLM INGESTION!")

import numpy as np
import matplotlib.pyplot as plt
import pickle

# 1. Load your Stitched Master
master_path = r"E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\NEPTUNE_RAW_MASTER.pkl"
with open(master_path, 'rb') as f:
    master = pickle.load(f)

wave = master['wave']
flux = master['flux']

# 2. Setup the Figure
plt.figure(figsize=(15, 7))

# Use a palette for the 14-file composite if you are still plotting individual lines
# Otherwise, plot the master in one high-contrast color
plt.plot(wave, flux, color='black', linewidth=0.5, alpha=0.9, label="Stitched Venus Master")

# 3. FORCE THE RANGE
# This ensures the plot doesn't show empty space or cut off data
plt.xlim(wave.min(), wave.max())

# Use a log scale for the Y-axis? 
# Uranus is very dim; log scale helps see the absorption dips clearly.
plt.yscale('log') 

# 4. Mark the Audit Successes
# features = {
#     "Ethane": (1584, 1650),
#     "Acetylene": (1700, 1900),
#     "Ozone": (2400, 2600)
# }

# for name, (low, high) in features.items():
#     plt.axvspan(low, high, color='yellow', alpha=0.1, label=f"{name} Band")

plt.title(f"Venus Master Spectrum: {wave.min():.1f} - {wave.max():.1f} Å", fontsize=14)
plt.xlabel("Wavelength (Å)")
plt.ylabel("Flux")
plt.legend(loc='upper right', fontsize='small')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.show()

import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter
import pickle
import os

def process_bulk_planet_data(file_paths, planet_name):
    all_fluxes = []
    common_wave = None

    for f_path in file_paths:
        with fits.open(f_path) as hdul:
            data = hdul[1].data
            # Extract - handling both SX1 and X1D formats
            wave = data['WAVELENGTH'][0] if 'X1D' in f_path else data['WAVELENGTH'].flatten()
            flux = data['FLUX'][0] if 'X1D' in f_path else data['FLUX'].flatten()
            dq = data['DQ'][0] if 'X1D' in f_path else data['DQ'].flatten()

            # Clean
            mask = (dq == 0) & (flux > 0)
            if common_wave is None:
                common_wave = wave[mask]
            
            # Interpolate to ensure they all line up perfectly
            interp_flux = np.interp(common_wave, wave[mask], flux[mask])
            all_fluxes.append(interp_flux)

    # Average and Smooth
    final_flux = np.mean(all_fluxes, axis=0)
    final_smooth = savgol_filter(final_flux, window_length=11, polyorder=3)

    # Save for the Brain Team
    out_file = f"{planet_name.upper()}_MASTER.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump({"wave": common_wave, "flux": final_smooth, "target": planet_name}, f)
    
    print(f"✅ Processed {len(file_paths)} files for {planet_name}. Saved as {out_file}")

# Example Usage
mars_files = ["E:\Data Fecked Hackathon\Spectral Data Cleaning and Sorting\mastDownload\HST\ocjhd3010\ocjhd3010_x1d.fits"]
process_bulk_planet_data(mars_files, "Mars")

import pickle
import matplotlib.pyplot as plt
import os

def visualize_pkl(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 1. Smart Key Detection
    # Wavelength might be stored as 'wave', 'wavelength', or 'wavelengths'
    wave_keys = ['wave', 'wavelength', 'wavelengths']
    flux_keys = ['flux', 'clean_flux', 'flux_cleaned']
    
    wave = next((data[k] for k in wave_keys if k in data), None)
    flux = next((data[k] for k in flux_keys if k in data), None)
    planet = data.get('target', data.get('planet', 'Unknown Planet'))

    if wave is None or flux is None:
        print(f"❌ Could not find wave/flux keys in {file_path}. Keys present: {list(data.keys())}")
        return

    # 2. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(wave, flux, color='darkorange', linewidth=1.5, label='Master Signal')
    
    # Add scientific context
    plt.title(f"Master Spectral Signature: {planet.upper()}", fontsize=14, fontweight='bold')
    plt.xlabel("Wavelength (Angstroms)", fontsize=12)
    plt.ylabel("Normalized Flux", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Add "Chemical Checkpoints" for the judges
    # If the wavelength covers the UV range (1500-3000A)
    #if wave.min() < 3000:
     #   plt.axvspan(1700, 1900, color='blue', alpha=0.1, label='Hydrocarbon Absorption')
      #  plt.axvspan(2100, 2300, color='green', alpha=0.1, label='Ammonia Band')

    plt.legend()
    
    # Save a copy for your presentation deck
    image_name = f"{planet}_plot.png"
    plt.savefig(image_name)
    plt.show()
    
    print(f"✅ Plot generated for {planet}!")
    print(f"📸 Image saved as: {image_name}")

# --- RUN IT FOR SATURN ---
visualize_pkl('SATURN_UV_MASTER.pkl')

# --- RUN IT FOR URANUS (once you finish merging) ---
#visualize_pkl('URANUS_MASTER.pkl')

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pickle

def process_jupiter_spectrum(file_path):
    print(f"🛠️  Cleaning File: {file_path}")
    
    with fits.open(file_path) as hdul:
        # SX1 data is in Extension 1
        data = hdul[1].data
        
        # 1. Extraction (SX1 usually uses WAVELENGTH and FLUX)
        wave = data['WAVELENGTH'].flatten()
        flux = data['FLUX'].flatten()
        dq = data['DQ'].flatten() # Data Quality flags
        
        # 2. Basic Cleaning: Remove 'bad' pixels
        # DQ == 0 is the gold standard for 'good data'
        mask = (dq == 0) & (flux > 0) # Also remove negative flux (physical noise)
        w_clean = wave[mask]
        f_clean = flux[mask]
        
        # 3. Smoothing: Savitzky-Golay Filter
        # This removes the 'fuzz' but keeps the chemical 'dips'
        # window_length must be odd. 11 or 15 is usually good for Jupiter.
        f_smooth = savgol_filter(f_clean, window_length=15, polyorder=3)
        
        return w_clean, f_clean, f_smooth

# --- EXECUTION ---
file_to_use = r"E:\DataFecked Hackathon\mastDownload\HST\o4ym01010\o4ym01010_sx1.fits"
wavelength, raw_flux, clean_flux = process_jupiter_spectrum(file_to_use)

# 4. Save as .pkl for the "Brain" Team (Squad B)
processed_data = {
    "target": "JUPITER",
    "wavelength": wavelength,
    "clean_flux": clean_flux,
    "unit_wave": "Angstroms",
    "unit_flux": "erg/s/cm2/A"
}

with open('cleaned_jupiter_spectrum.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

print("✅ Data cleaned and saved to 'cleaned_jupiter_spectrum.pkl'")

# 5. Visual Check

plt.figure(figsize=(12, 6))
plt.plot(wavelength, raw_flux, color='lightgray', label='Raw Data', alpha=0.5)
plt.plot(wavelength, clean_flux, color='crimson', label='Cleaned Signal', linewidth=1.5)
plt.title("Jupiter Spectral Signature (Person 2 Cleaned Output)")
plt.xlabel("Wavelength (Angstroms)")
plt.ylabel("Flux")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pickle

def process_jupiter_spectrum(file_path):
    print(f"🛠️  Cleaning File: {file_path}")
    
    with fits.open(file_path) as hdul:
        # SX1 data is in Extension 1
        data = hdul[1].data
        
        # 1. Extraction (SX1 usually uses WAVELENGTH and FLUX)
        wave = data['WAVELENGTH'].flatten()
        flux = data['FLUX'].flatten()
        dq = data['DQ'].flatten() # Data Quality flags
        
        # 2. Basic Cleaning: Remove 'bad' pixels
        # DQ == 0 is the gold standard for 'good data'
        mask = (dq == 0) & (flux > 0) # Also remove negative flux (physical noise)
        w_clean = wave[mask]
        f_clean = flux[mask]
        
        # 3. Smoothing: Savitzky-Golay Filter
        # This removes the 'fuzz' but keeps the chemical 'dips'
        # window_length must be odd. 11 or 15 is usually good for Jupiter.
        f_smooth = savgol_filter(f_clean, window_length=15, polyorder=3)
        
        return w_clean, f_clean, f_smooth

# --- EXECUTION ---
file_to_use = r"E:\DataFecked Hackathon\mastDownload\HST\o4ym01020\o4ym01020_sx1.fits"
wavelength, raw_flux, clean_flux = process_jupiter_spectrum(file_to_use)

# 4. Save as .pkl for the "Brain" Team (Squad B)
processed_data = {
    "target": "JUPITER",
    "wavelength": wavelength,
    "clean_flux": clean_flux,
    "unit_wave": "Angstroms",
    "unit_flux": "erg/s/cm2/A"
}

with open('cleaned_jupiter_spectrum.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

print("✅ Data cleaned and saved to 'cleaned_jupiter_spectrum.pkl'")

# 5. Visual Check

plt.figure(figsize=(12, 6))
plt.plot(wavelength, raw_flux, color='lightgray', label='Raw Data', alpha=0.5)
plt.plot(wavelength, clean_flux, color='crimson', label='Cleaned Signal', linewidth=1.5)
plt.title("Jupiter Spectral Signature (Person 2 Cleaned Output)")
plt.xlabel("Wavelength (Angstroms)")
plt.ylabel("Flux")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pickle

def process_jupiter_spectrum(file_path):
    print(f"🛠️  Cleaning File: {file_path}")
    
    with fits.open(file_path) as hdul:
        # SX1 data is in Extension 1
        data = hdul[1].data
        
        # 1. Extraction (SX1 usually uses WAVELENGTH and FLUX)
        wave = data['WAVELENGTH'].flatten()
        flux = data['FLUX'].flatten()
        dq = data['DQ'].flatten() # Data Quality flags
        
        # 2. Basic Cleaning: Remove 'bad' pixels
        # DQ == 0 is the gold standard for 'good data'
        mask = (dq == 0) & (flux > 0) # Also remove negative flux (physical noise)
        w_clean = wave[mask]
        f_clean = flux[mask]
        
        # 3. Smoothing: Savitzky-Golay Filter
        # This removes the 'fuzz' but keeps the chemical 'dips'
        # window_length must be odd. 11 or 15 is usually good for Jupiter.
        f_smooth = savgol_filter(f_clean, window_length=15, polyorder=3)
        
        return w_clean, f_clean, f_smooth

# --- EXECUTION ---
file_to_use = r"E:\DataFecked Hackathon\mastDownload\HST\o4ym01030\o4ym01030_sx1.fits"
wavelength, raw_flux, clean_flux = process_jupiter_spectrum(file_to_use)

# 4. Save as .pkl for the "Brain" Team (Squad B)
processed_data = {
    "target": "JUPITER",
    "wavelength": wavelength,
    "clean_flux": clean_flux,
    "unit_wave": "Angstroms",
    "unit_flux": "erg/s/cm2/A"
}

with open('cleaned_jupiter_spectrum.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

print("✅ Data cleaned and saved to 'cleaned_jupiter_spectrum.pkl'")

# 5. Visual Check

plt.figure(figsize=(12, 6))
plt.plot(wavelength, raw_flux, color='lightgray', label='Raw Data', alpha=0.5)
plt.plot(wavelength, clean_flux, color='crimson', label='Cleaned Signal', linewidth=1.5)
plt.title("Jupiter Spectral Signature (Person 2 Cleaned Output)")
plt.xlabel("Wavelength (Angstroms)")
plt.ylabel("Flux")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

import os
os.chdir("E:\\DataFecked Hackathon\\Jupiter - mastDownload\\Merged File")
os.getcwd()

import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter
import pickle

def clean_and_grid(file_path, target_wave):
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        wave, flux, dq = data['WAVELENGTH'].flatten(), data['FLUX'].flatten(), data['DQ'].flatten()
        # Clean bad pixels
        mask = (dq == 0) & (flux > 0)
        # Interpolate onto the master grid so we can average them
        flux_interp = np.interp(target_wave, wave[mask], flux[mask], left=0, right=0)
        return flux_interp

# 1. Define your files
files = [
    r"E:\DataFecked Hackathon\Jupiter - mastDownload\HST\o4ym01010\o4ym01010_sx1.fits",
    r"E:\DataFecked Hackathon\Jupiter - mastDownload\HST\o4ym01020\o4ym01020_sx1.fits",
    r"E:\DataFecked Hackathon\Jupiter - mastDownload\HST\o4ym01030\o4ym01030_sx1.fits"
]

# 2. Create a master wavelength grid (based on the first file)
with fits.open(files[0]) as h:
    master_wave = h[1].data['WAVELENGTH'].flatten()

# 3. Process and Average
all_fluxes = [clean_and_grid(f, master_wave) for f in files]
average_flux = np.mean(all_fluxes, axis=0)
final_smooth = savgol_filter(average_flux, window_length=15, polyorder=3)

# 4. SAVE THE ONE FILE FOR THE BRAIN TEAM
master_data = {
    "target": "JUPITER",
    "wavelength": master_wave,
    "flux": final_smooth,
    "notes": "Averaged 010, 020, 030 SX1 files. UV range."
}

with open('JUPITER_MASTER_SPECTRA.pkl', 'wb') as f:
    pickle.dump(master_data, f)

print("🚀 DONE! Send 'JUPITER_MASTER_SPECTRA.pkl' to the Brain Team.")

import pickle
with open('JUPITER_MASTER_SPECTRA.pkl', 'rb') as f:
    df = pd.read_pickle(f)
df
# data['flux'] is now the clean line for their LLM prompt!

### InfraRed Data

import numpy as np
import pickle
import matplotlib.pyplot as plt

def process_ir_fits_to_pkl(fits_filename, output_pkl):
    """
    Universally reads IRTF/SpeX style FITS, cleans, and saves to .pkl
    """
    # Open binary FITS to avoid dependency issues
    with open(fits_filename, 'rb') as f:
        # 1. Skip Header (Search for 'END')
        header_content = b""
        while b'END     ' not in header_content:
            header_content += f.read(2880)
        
        # 2. Find dimensions (Assume 32-bit float, NAXIS1 x NAXIS2)
        # For IRTF: Row 0 = Wave, Row 1 = Flux, Row 2 = Error
        # Adjust naxis1/naxis2 based on your specific FITS header if needed
        data = np.frombuffer(f.read(), dtype='>f4')
        naxis1 = 4165 # Standard IRTF length; update if different
        data = data[:naxis1*3].reshape((3, naxis1))
        
        wave = data[0].byteswap().newbyteorder().astype(float)
        flux = data[1].byteswap().newbyteorder().astype(float)
        
        # 3. Cleaning: Remove non-finite (NaN) and non-physical (Negative) flux
        mask = np.isfinite(flux) & (flux > 0)
        clean_wave = wave[mask]
        clean_flux = flux[mask]
        
        # 4. Save to PKL
        master_data = {'wave': clean_wave, 'flux': clean_flux}
        with open(output_pkl, 'wb') as pkl_file:
            pickle.dump(master_data, pkl_file)
        
        print(f"Successfully created {output_pkl}")
        return clean_wave, clean_flux

# Example Usage:
ir_wave, ir_flux = process_ir_fits_to_pkl('uranus_spectrum_irtf.fits', 'URANUS_IR_MASTER.pkl')

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def create_planet_master(fits_filename, planet_name, observation_type, output_pkl):
    """
    fits_filename: Path to your FITS file
    planet_name: 'URANUS', 'JUPITER', etc.
    observation_type: 'IR' or 'UV'
    """
    
    # 1. MANUALLY READ FITS (Safe for all environments)
    with open(fits_filename, 'rb') as f:
        header_content = b""
        while b'END     ' not in header_content:
            header_content += f.read(2880)
        
        # Read the raw data buffer
        raw_data = np.frombuffer(f.read(), dtype='>f4')
        
        # IRTF Data standard: 3 rows (Wave, Flux, Error)
        # Note: If your next planet FITS has a different length, 
        # you can adjust naxis1 or use raw_data.reshape(3, -1)
        n_points = len(raw_data) // 3
        data = raw_data[:n_points*3].reshape((3, n_points))
        
        # Convert to little-endian for Python processing
        wave = data[0].byteswap().newbyteorder().astype(float)
        flux = data[1].byteswap().newbyteorder().astype(float)

    # 2. DATA CLEANING
    # Remove NaNs, infinities, and non-physical negative flux
    mask = np.isfinite(flux) & (flux > 0)
    clean_wave = wave[mask]
    clean_flux = flux[mask]

    # 3. BUILD THE UNIVERSAL DICTIONARY
    master_data = {
        'target_name': planet_name.upper(),
        'obs_type': observation_type.upper(),
        'wave': clean_wave,
        'flux': clean_flux,
        'wave_unit': 'um' if observation_type.upper() == 'IR' else 'Angstrom',
        'timestamp': '2000-05-18' # You can extract this from header if needed
    }

    # 4. SAVE TO PKL
    with open(output_pkl, 'wb') as pkl_file:
        pickle.dump(master_data, pkl_file)
    
    print(f"✅ Success: {planet_name} {observation_type} Master File saved to {output_pkl}")
    
    # 5. AUTO-PLOT FOR VALIDATION
    plt.figure(figsize=(8, 4))
    plt.plot(clean_wave, clean_flux, label=f"{planet_name} {observation_type}")
    plt.title(f"Master Spectrum: {planet_name} ({observation_type})")
    plt.xlabel(f"Wavelength ({master_data['wave_unit']})")
    plt.ylabel("Flux")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    return master_data

# --- HOW TO USE FOR URANUS ---
#uranus_ir = create_planet_master('uranus_spectrum_irtf.fits', 'URANUS', 'IR', 'URANUS_IR_MASTER.pkl')

# --- HOW TO USE FOR JUPITER (Next) ---
saturn_ir = create_planet_master('saturn_spectrum_irtf.fits', 'SATURN', 'IR', 'SATURN_IR_MASTER.pkl')

import pickle
with open('URANUS_IR_MASTER.pkl', 'rb') as f:
    df = pd.read_pickle(f)
df

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# 1. Open the IRTF Uranus File
with fits.open('saturn_spectrum_irtf.fits') as hdul:
    data = hdul[0].data
    header = hdul[0].header
    # In IRTF files, Row 0 is Wavelength (um), Row 1 is Flux, Row 2 is Uncertainty
    wave_ir = data[0]
    flux_ir = data[1]

# 2. Plotting the IR Signature
plt.figure(figsize=(12, 5))
plt.plot(wave_ir, flux_ir, color='teal', label='Saturn IR (Composition)')
plt.title("Saturn IR Spectral Fingerprint (IRTF)")
plt.xlabel("Wavelength (microns)")
plt.ylabel("Flux (W/m^2/um)")
plt.grid(alpha=0.3)
plt.show()

import pickle
import numpy as np
import pandas as pd

def generate_image_team_guide(planet_name, uv_pkl_path, ir_pkl_path):
    """
    Combines IR Dips and UV Peaks into a search strategy for the Image Team.
    """
    results = []

    # 1. Process IR Data (Looking for Dips)
    with open(ir_pkl_path, 'rb') as f:
        ir_data = pickle.load(f)
    
    # Define standard Methane search zones
    ir_zones = [
        (1.1, 1.2, "Methane Band 1 (Deep Cloud)"),
        (1.3, 1.5, "Methane Band 2 (Storm Layer)"),
        (1.6, 2.5, "Primary Methane Absorption (Rings/Storms)")
    ]
    
    for low, high, science in ir_zones:
        mask = (ir_data['wave'] >= low) & (ir_data['wave'] <= high)
        if np.any(mask):
            # Find the absolute lowest flux in this zone
            idx_min = np.argmin(ir_data['flux'][mask])
            wave_val = ir_data['wave'][mask][idx_min]
            results.append({
                'Planet': planet_name,
                'Spectrum': 'IR',
                'Feature': 'DIP (Minimum)',
                'Wavelength': f"{wave_val:.3f} um",
                'Science_Goal': science,
                'Recommended_PDS_Filter': 'ch4_u / ch4_js'
            })

    # 2. Process UV Data (Looking for Peaks)
    with open(uv_pkl_path, 'rb') as f:
        uv_data = pickle.load(f)
        
    uv_zones = [
        (1200, 1600, "Auroral Emission Lines"),
        (2000, 2600, "High-Altitude Haze")
    ]

    for low, high, science in uv_zones:
        mask = (uv_data['wave'] >= low) & (uv_data['wave'] <= high)
        if np.any(mask):
            # Find the absolute highest flux in this zone
            idx_max = np.argmax(uv_data['flux'][mask])
            wave_val = uv_data['wave'][mask][idx_max]
            results.append({
                'Planet': planet_name,
                'Spectrum': 'UV',
                'Feature': 'PEAK (Maximum)',
                'Wavelength': f"{wave_val:.1f} Å",
                'Science_Goal': science,
                'Recommended_PDS_Filter': 'uv / violet'
            })

    # 3. Create the Guide
    df_guide = pd.DataFrame(results)
    output_name = f"{planet_name}_search_guide.csv"
    df_guide.to_csv(output_name, index=False)
    
    print(f"--- {planet_name} IMAGE SEARCH STRATEGY ---")
    print(df_guide.to_string(index=False))
    return df_guide

# --- EXECUTION ---
guide = generate_image_team_guide('SATURN', 'SATURN_UV_MASTER.pkl', 'SATURN_IR_MASTER.pkl')