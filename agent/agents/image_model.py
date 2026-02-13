"""Image Model Agent - Generates spectral barcode visualizations from UV/IR data.

This agent handles spectral fingerprint visualization by:
1. Loading UV and IR .pkl files for multiple planets
2. Creating combined spectral barcodes (fingerprints)
3. Generating similarity heatmaps and clustering dendrograms
4. Returning 4 visualization figures for display
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import mlflow

from ..base import BaseAgent
from ..state import PipelineState


class ImageModelAgent(BaseAgent):
    """Generates spectral barcode visualizations from UV/IR planetary data.

    Responsibilities:
    - Load UV and IR .pkl files for all planets
    - Generate combined spectral fingerprints (barcodes)
    - Create similarity heatmaps showing planet relationships
    - Generate hierarchical clustering dendrograms
    - Return all 4 visualizations for display in Streamlit
    """

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize image model agent.

        Args:
            data_dir: Directory containing UV/IR .pkl files (defaults to Spectral Service/data/real)
        """
        super().__init__("ImageModelAgent")

        # Set default data directory
        if data_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            data_dir = project_root / "Spectral Service" / "data" / "real"

        self.data_dir = Path(data_dir)
        self.planets = ["jupiter", "mars", "neptune", "venus", "saturn", "uranus"]

    def process(self, state: PipelineState) -> PipelineState:
        """Process spectral data and generate visualizations.

        Args:
            state: Pipeline state with optional input_path for uploaded PKL file

        Returns:
            Updated state with image_visualizations (list of 4 figure dictionaries)
        """
        input_path = state.get("input_path")

        try:
            mlflow.log_param("image_agent_mode", "spectral_barcode")
            mlflow.log_param("data_directory", str(self.data_dir))
            if input_path:
                mlflow.log_param("input_file", str(input_path))
        except Exception:
            pass

        # Load planet data
        try:
            with mlflow.start_span(name="load_spectral_data") as span:
                # ALWAYS load all training planets for comparison
                # This ensures we have enough data for similarity analysis
                try:
                    planet_data = self._load_all_planets()
                    if not planet_data:
                        raise ValueError(f"No training planets found in {self.data_dir}. Check if PKL files exist.")
                except Exception as e:
                    error_msg = f"Failed to load training planets: {e}"
                    state.setdefault("errors", []).append(error_msg)
                    state["image_visualizations"] = []
                    return state

                # If user uploaded a specific planet, try to add it to the comparison
                uploaded_planet = None
                if input_path and Path(input_path).suffix == '.pkl':
                    try:
                        uploaded_data = self._load_uploaded_planet(input_path)
                        # Merge uploaded planet with training data
                        planet_data.update(uploaded_data)
                        uploaded_planet = list(uploaded_data.keys())[0]
                        mode = "uploaded_with_training"
                    except Exception as e:
                        # If upload fails, just use training data
                        state.setdefault("errors", []).append(f"Could not load uploaded file: {e}. Using training planets only.")
                        mode = "training_only"
                else:
                    mode = "all_training"

                span.set_attribute("mode", mode)
                span.set_attribute("uploaded_planet", uploaded_planet or "none")
                span.set_attribute("planets_loaded", len(planet_data))
                span.set_attribute("planets_list", ", ".join(planet_data.keys()))

                mlflow.log_param("visualization_mode", mode)
                mlflow.log_param("uploaded_planet", uploaded_planet or "none")
                mlflow.log_param("planets_loaded", ", ".join(planet_data.keys()))
                mlflow.log_metric("planet_count", len(planet_data))

                # Store metadata for UI display
                state.setdefault("metadata", {})["planets_analyzed"] = list(planet_data.keys())
                state["metadata"]["uploaded_planet"] = uploaded_planet or "none"
                state["metadata"]["total_planets"] = len(planet_data)

        except Exception as exc:
            error_msg = f"Failed to load planet data: {exc}"
            state.setdefault("errors", []).append(error_msg)
            state["image_visualizations"] = []
            return state

        if not planet_data:
            state.setdefault("errors", []).append("No planet data found")
            state["image_visualizations"] = []
            return state

        # Generate all 4 visualizations
        try:
            with mlflow.start_span(name="generate_visualizations") as span:
                start_time = time.time()

                visualizations = []
                num_planets = len(planet_data)
                planet_names = ", ".join([p.title() for p in planet_data.keys()])

                # 1. Individual barcodes
                barcode_fig = self._create_combined_barcodes(planet_data)
                visualizations.append({
                    "title": f"Spectral Fingerprints ({num_planets} Planets)",
                    "figure": barcode_fig,
                    "description": f"Combined UV+IR spectral fingerprints: {planet_names}"
                })

                # 2. Similarity heatmap
                heatmap_fig = self._create_similarity_heatmap(planet_data)
                visualizations.append({
                    "title": f"Planet Similarity Matrix ({num_planets}×{num_planets})",
                    "figure": heatmap_fig,
                    "description": f"Cosine distance matrix comparing {num_planets} planetary spectra"
                })

                # 3. Clustering dendrogram
                dendro_fig = self._create_dendrogram(planet_data)
                visualizations.append({
                    "title": f"Hierarchical Clustering ({num_planets} Planets)",
                    "figure": dendro_fig,
                    "description": f"Dendrogram showing spectral similarity relationships"
                })

                # 4. Spectral overlay plot (COMMENTED OUT - not needed)
                # overlay_fig = self._create_spectral_overlay(planet_data)
                # visualizations.append({
                #     "title": f"Normalized Spectra Overlay ({num_planets} Planets)",
                #     "figure": overlay_fig,
                #     "description": "All planet spectra overlaid for comparison"
                # })

                generation_time = time.time() - start_time
                span.set_attribute("visualization_count", len(visualizations))
                span.set_attribute("generation_time_ms", int(generation_time * 1000))

                mlflow.log_metric("visualization_generation_time_ms", generation_time * 1000)
                mlflow.log_metric("visualization_count", len(visualizations))

        except Exception as exc:
            state.setdefault("errors", []).append(f"Visualization generation failed: {exc}")
            state["image_visualizations"] = []
            return state

        # Update state
        state["image_visualizations"] = visualizations

        try:
            mlflow.log_metric("image_agent_success", 1)
        except Exception:
            pass

        return state

    def _load_uploaded_planet(self, pkl_path: str) -> Dict[str, Dict[str, Any]]:
        """Load a single uploaded planet PKL file.

        Expects either:
        1. A combined PKL with both UV and IR data
        2. Or tries to find matching UV/IR pair based on filename

        Args:
            pkl_path: Path to uploaded PKL file

        Returns:
            Dict with single planet data
        """
        pkl_file = Path(pkl_path)
        planet_data = {}

        # Extract planet name from filename
        # Handle formats like: jupiter_combined.pkl, jupiter.pkl, jupiter_uv.pkl, etc.
        planet_name = pkl_file.stem.replace('_combined', '').replace('_uv', '').replace('_ir', '').lower()

        try:
            # Try to load as combined file first
            data = self._load_pkl(pkl_file)

            # Check if it's a dictionary with 'uv' and 'ir' keys
            if isinstance(data, dict) and 'uv' in data and 'ir' in data:
                uv_df = self._to_spectrum_df(data['uv'])
                ir_df = self._to_spectrum_df(data['ir'])
            else:
                # If it's a single spectrum, try to find its pair
                # Check if filename contains '_uv' or '_ir'
                if '_uv' in pkl_file.stem:
                    uv_df = self._to_spectrum_df(data)
                    # Try to find matching IR file in same directory first
                    ir_path = pkl_file.parent / pkl_file.name.replace('_uv', '_ir')
                    if ir_path.exists():
                        ir_df = self._to_spectrum_df(self._load_pkl(ir_path))
                    else:
                        # Fallback: try data directory
                        ir_path = self.data_dir / f"{planet_name}_ir.pkl"
                        if ir_path.exists():
                            ir_df = self._to_spectrum_df(self._load_pkl(ir_path))
                        else:
                            raise ValueError(
                                f"Image Analysis requires both UV and IR data for spectral fingerprints.\n"
                                f"Missing: {planet_name}_ir.pkl\n\n"
                                f"Options:\n"
                                f"  1. Upload both {planet_name}_uv.pkl and {planet_name}_ir.pkl\n"
                                f"  2. Use {planet_name}_combined.pkl\n"
                                f"  3. For unknown exoplanets: Use 'Spectral Analysis' mode instead (accepts single FITS/CSV)"
                            )

                elif '_ir' in pkl_file.stem:
                    ir_df = self._to_spectrum_df(data)
                    # Try to find matching UV file in same directory first
                    uv_path = pkl_file.parent / pkl_file.name.replace('_ir', '_uv')
                    if uv_path.exists():
                        uv_df = self._to_spectrum_df(self._load_pkl(uv_path))
                    else:
                        # Fallback: try data directory
                        uv_path = self.data_dir / f"{planet_name}_uv.pkl"
                        if uv_path.exists():
                            uv_df = self._to_spectrum_df(self._load_pkl(uv_path))
                        else:
                            raise ValueError(
                                f"Image Analysis requires both UV and IR data for spectral fingerprints.\n"
                                f"Missing: {planet_name}_uv.pkl\n\n"
                                f"Options:\n"
                                f"  1. Upload both {planet_name}_uv.pkl and {planet_name}_ir.pkl\n"
                                f"  2. Use {planet_name}_combined.pkl\n"
                                f"  3. For unknown exoplanets: Use 'Spectral Analysis' mode instead (accepts single FITS/CSV)"
                            )
                else:
                    # Assume it's UV data, try to find IR in same directory or data directory
                    uv_df = self._to_spectrum_df(data)

                    # Try same directory first
                    ir_path = pkl_file.parent / f"{planet_name}_ir.pkl"
                    if not ir_path.exists():
                        # Try data directory
                        ir_path = self.data_dir / f"{planet_name}_ir.pkl"

                    if ir_path.exists():
                        ir_df = self._to_spectrum_df(self._load_pkl(ir_path))
                    else:
                        raise ValueError(
                            f"Image Analysis requires both UV and IR data for spectral fingerprints.\n"
                            f"Missing: {planet_name}_ir.pkl\n\n"
                            f"Options:\n"
                            f"  1. Upload both {planet_name}_uv.pkl and {planet_name}_ir.pkl\n"
                            f"  2. Use {planet_name}_combined.pkl\n"
                            f"  3. For unknown exoplanets: Use 'Spectral Analysis' mode instead (accepts single FITS/CSV)"
                        )

            # Get normalized fingerprint
            w, n = self._get_normalized_fingerprint(uv_df, ir_df)

            planet_data[planet_name] = {
                'uv_df': uv_df,
                'ir_df': ir_df,
                'wavelength': w,
                'normalized_flux': n
            }

        except Exception as e:
            raise ValueError(f"Failed to load planet data from {pkl_file.name}: {e}")

        return planet_data

    def _load_all_planets(self) -> Dict[str, Dict[str, Any]]:
        """Load UV and IR data for all planets from data directory.

        Returns:
            Dict mapping planet name to {'uv_df', 'ir_df', 'wavelength', 'normalized_flux'}
        """
        planet_data = {}

        for planet in self.planets:
            uv_path = self.data_dir / f"{planet}_uv.pkl"
            ir_path = self.data_dir / f"{planet}_ir.pkl"

            # Skip if either file missing
            if not uv_path.exists() or not ir_path.exists():
                continue

            try:
                # Load UV and IR
                uv_df = self._to_spectrum_df(self._load_pkl(uv_path))
                ir_df = self._to_spectrum_df(self._load_pkl(ir_path))

                # Get normalized fingerprint
                w, n = self._get_normalized_fingerprint(uv_df, ir_df)

                planet_data[planet] = {
                    'uv_df': uv_df,
                    'ir_df': ir_df,
                    'wavelength': w,
                    'normalized_flux': n
                }
            except Exception as e:
                print(f"Warning: Failed to load {planet}: {e}")
                continue

        return planet_data

    def _load_pkl(self, path: Path):
        """Load pickle file."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def _to_spectrum_df(self, obj) -> pd.DataFrame:
        """Convert any pkl structure to standard DataFrame with [wavelength, flux] columns."""
        # DataFrame format
        if isinstance(obj, pd.DataFrame):
            cols = {c.lower(): c for c in obj.columns}

            wl_candidates = ["wavelength", "wave", "wl", "lambda", "lam",
                           "wavelength_um", "wavelength_nm", "wavelength_m"]
            fx_candidates = ["flux", "intensity", "radiance", "radiance_final", "reflectance",
                           "flux_mjy_sr", "flux_jy", "f"]

            wl = next((cols[k] for k in wl_candidates if k in cols), None)
            fx = next((cols[k] for k in fx_candidates if k in cols), None)

            if wl is None or fx is None:
                raise ValueError(f"Cannot detect wavelength/flux columns: {list(obj.columns)}")

            return obj[[wl, fx]].rename(columns={wl: "wavelength", fx: "flux"}).copy()

        # Dictionary format
        if isinstance(obj, dict):
            keys = {str(k).lower(): k for k in obj.keys()}

            wl_candidates = ["wavelength", "wave", "wl", "lambda", "lam", "wavelength_um"]
            fx_candidates = ["flux", "intensity", "radiance", "radiance_final", "reflectance", "f"]

            wl_k = next((keys[k] for k in wl_candidates if k in keys), None)
            fx_k = next((keys[k] for k in fx_candidates if k in keys), None)

            if wl_k is None or fx_k is None:
                raise ValueError(f"Cannot detect wavelength/flux keys: {list(obj.keys())[:30]}")

            return pd.DataFrame({
                "wavelength": np.asarray(obj[wl_k], dtype=float),
                "flux": np.asarray(obj[fx_k], dtype=float)
            })

        raise ValueError(f"Unsupported pickle format: {type(obj)}")

    def _preprocess(self, df: pd.DataFrame):
        """Clean and smooth spectrum."""
        w = df["wavelength"].to_numpy(dtype=float)
        f = df["flux"].to_numpy(dtype=float)

        # Remove NaN/inf
        mask = np.isfinite(w) & np.isfinite(f)
        w, f = w[mask], f[mask]

        # Sort by wavelength
        order = np.argsort(w)
        w, f = w[order], f[order]

        # Remove duplicate wavelengths
        _, idx = np.unique(w, return_index=True)
        w, f = w[idx], f[idx]

        # Savitzky-Golay smoothing
        if len(f) > 7:
            window = min(31, len(f) - 1)
            if window % 2 == 0:
                window -= 1
            if window < 5:
                window = 5
            poly = 3
            if poly >= window:
                poly = max(2, window - 2)
            f = savgol_filter(f, window_length=window, polyorder=poly)

        return w, f

    def _norm01(self, x: np.ndarray):
        """Normalize to [0, 1] range."""
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    def _get_normalized_fingerprint(self, uv_df: pd.DataFrame, ir_df: pd.DataFrame):
        """Get combined normalized UV+IR fingerprint."""
        w_uv, f_uv = self._preprocess(uv_df)
        w_ir, f_ir = self._preprocess(ir_df)

        # Normalize UV and IR separately (different units/scales)
        n_uv = self._norm01(f_uv)
        n_ir = self._norm01(f_ir)

        # Merge
        w = np.concatenate([w_uv, w_ir])
        n = np.concatenate([n_uv, n_ir])

        # Sort by wavelength
        order = np.argsort(w)
        return w[order], n[order]

    def _create_combined_barcodes(self, planet_data: Dict) -> plt.Figure:
        """Create combined barcode display for all planets."""
        n_planets = len(planet_data)
        fig, axes = plt.subplots(n_planets, 1, figsize=(12, n_planets * 0.8))

        if n_planets == 1:
            axes = [axes]

        for ax, (planet, data) in zip(axes, planet_data.items()):
            w = data['wavelength']
            n = data['normalized_flux']

            # Create absorption barcode
            intensity = 1.0 - n
            strip_height = 60
            strip = np.tile(intensity, (strip_height, 1))

            ax.imshow(strip, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(planet.capitalize(), loc="left", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        return fig

    def _create_similarity_heatmap(self, planet_data: Dict) -> plt.Figure:
        """Create similarity heatmap between planets."""
        # Build common wavelength grid
        planet_list = list(planet_data.keys())
        w_min = max(data['wavelength'].min() for data in planet_data.values())
        w_max = min(data['wavelength'].max() for data in planet_data.values())
        grid = np.linspace(w_min, w_max, 2000)

        # Interpolate all planets onto common grid
        X = []
        for planet in planet_list:
            w = planet_data[planet]['wavelength']
            n = planet_data[planet]['normalized_flux']
            f = interp1d(w, n, kind="linear", bounds_error=False, fill_value="extrapolate")
            X.append(f(grid))
        X = np.vstack(X)

        # Compute cosine distance matrix
        D = squareform(pdist(X, metric="cosine"))

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(D, aspect="equal", cmap="YlOrRd")

        ax.set_xticks(range(len(planet_list)))
        ax.set_yticks(range(len(planet_list)))
        ax.set_xticklabels([p.capitalize() for p in planet_list], rotation=45, ha="right")
        ax.set_yticklabels([p.capitalize() for p in planet_list])

        ax.set_title("Spectral Fingerprint Distance (Cosine)", fontsize=12, pad=10)
        plt.colorbar(im, ax=ax, label="Distance")

        # Add distance values in lower triangle only
        n = len(planet_list)
        for i in range(n):
            for j in range(n):
                if i > j:  # lower triangle only
                    ax.text(j, i,
                           f"{D[i, j]:.2f}",
                           ha="center",
                           va="center",
                           color="black",
                           fontsize=9)

        plt.tight_layout()

        return fig

    def _create_dendrogram(self, planet_data: Dict) -> plt.Figure:
        """Create hierarchical clustering dendrogram."""
        # Build common wavelength grid (same as heatmap)
        planet_list = list(planet_data.keys())
        w_min = max(data['wavelength'].min() for data in planet_data.values())
        w_max = min(data['wavelength'].max() for data in planet_data.values())
        grid = np.linspace(w_min, w_max, 2000)

        # Interpolate all planets onto common grid
        X = []
        for planet in planet_list:
            w = planet_data[planet]['wavelength']
            n = planet_data[planet]['normalized_flux']
            f = interp1d(w, n, kind="linear", bounds_error=False, fill_value="extrapolate")
            X.append(f(grid))
        X = np.vstack(X)

        # Hierarchical clustering
        Z = linkage(X, method="average", metric="cosine")

        # Plot dendrogram
        fig, ax = plt.subplots(figsize=(8, 5))
        dendrogram(Z, labels=[p.capitalize() for p in planet_list], ax=ax)
        ax.set_title("Hierarchical Clustering of Spectral Fingerprints", fontsize=12, pad=10)
        ax.set_ylabel("Distance")
        plt.tight_layout()

        return fig

    def _create_spectral_overlay(self, planet_data: Dict) -> plt.Figure:
        """Create overlay plot of all normalized spectra."""
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(planet_data)))

        for (planet, data), color in zip(planet_data.items(), colors):
            w = data['wavelength']
            n = data['normalized_flux']
            ax.plot(w, n, label=planet.capitalize(), alpha=0.7, linewidth=1.5, color=color)

        ax.set_xlabel("Wavelength", fontsize=11)
        ax.set_ylabel("Normalized Flux", fontsize=11)
        ax.set_title("Normalized Spectral Overlay (All Planets)", fontsize=12, pad=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig
