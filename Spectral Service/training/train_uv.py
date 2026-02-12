"""Train UV spectral model with real data + physics-based augmentation."""
from __future__ import annotations

import json, pickle, time, sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import optuna
import mlflow

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mlflow_utils import setup_mlflow, safe_log_param, safe_log_metric
from augmentation import create_augmented_dataset
from expanded_species import UV_SPECIES_EXPANDED, get_labels_for_planet

ROOT = Path(__file__).resolve().parents[1]
DATA_REAL = ROOT / "data" / "real"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# CONFIGURATION - ADJUST THESE!
# ============================================================================
N_AUGMENT_PER_PLANET = 75  # Number of augmented versions per planet (50-150)
N_RESAMPLE = 1024
BATCH = 128

N_TRIALS = 20  # Optuna trials
EPOCHS_TUNE = 10
PATIENCE_TUNE = 3

EPOCHS_FINAL = 25
PATIENCE_FINAL = 6


# ============================================================================
# DATA LOADING
# ============================================================================

def load_pkl_spectrum(p: Path) -> Tuple[str, np.ndarray, np.ndarray]:
    """Load spectrum from pickle file."""
    with open(p, "rb") as f:
        d = pickle.load(f)
    target = str(d.get("target", p.stem)).upper()
    w = np.asarray(d.get("wavelength", d.get("wave")), dtype=float)
    y = np.asarray(d["flux"], dtype=float)
    m = np.isfinite(w) & np.isfinite(y)
    w, y = w[m], y[m]
    idx = np.argsort(w)
    return target, w[idx], y[idx]


def detect_uv_files() -> Dict[str, Path]:
    """Auto-detect UV spectra files."""
    uv_files = {}
    for pkl_file in DATA_REAL.glob("*_uv.pkl"):
        planet_name = pkl_file.stem.replace("_uv", "").upper()
        uv_files[planet_name] = pkl_file
    return uv_files


# ============================================================================
# PREPROCESSING
# ============================================================================

def resample_to_fixed(wave: np.ndarray, flux: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resample spectrum to fixed number of points."""
    wave_new = np.linspace(wave.min(), wave.max(), n)
    flux_new = np.interp(wave_new, wave, flux)
    return wave_new.astype(np.float32), flux_new.astype(np.float32)


def compute_baseline(flux: np.ndarray, win: int = 151) -> np.ndarray:
    """Compute baseline using Savitzky-Golay filter."""
    from scipy.signal import savgol_filter
    n = len(flux)
    win = max(11, min(win, n - 1))
    if win % 2 == 0:
        win += 1
    baseline = savgol_filter(flux.astype(float), window_length=win, polyorder=3)
    baseline = np.clip(baseline, np.percentile(baseline, 1), np.percentile(baseline, 99))
    return (baseline + 1e-12).astype(np.float32)


def make_channels(wave: np.ndarray, flux: np.ndarray, baseline_win: int = 151) -> np.ndarray:
    """Create 3-channel representation: [normalized, 1st derivative, 2nd derivative]."""
    baseline = compute_baseline(flux, win=baseline_win)
    r = (flux / baseline) - 1.0
    r = (r - np.median(r)) / (np.std(r) + 1e-8)
    d1 = np.gradient(r, wave)
    d2 = np.gradient(d1, wave)
    return np.stack([r, d1, d2], axis=0).astype(np.float32)


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def build_augmented_dataset(win: int, n_augment: int, seed0: int):
    """Build training dataset with augmentation.

    IMPORTANT: Splits by planet, not by sample, to prevent data leakage.
    All augmentations of a planet stay together in either train or val.
    """
    uv_files = detect_uv_files()

    # Build dataset planet-by-planet
    planet_datasets = []

    for planet_idx, (planet_name, pkl_path) in enumerate(uv_files.items()):
        # Load real spectrum
        _, wave_real, flux_real = load_pkl_spectrum(pkl_path)

        # Get labels
        labels = get_labels_for_planet(planet_name, UV_SPECIES_EXPANDED, domain="UV")

        # Create augmented versions
        waves_aug, fluxes_aug = create_augmented_dataset(
            wave_real, flux_real,
            n_augmentations=n_augment,
            seed_offset=seed0 + planet_idx
        )

        # Preprocess each
        X_planet, Y_planet = [], []
        for wave_aug, flux_aug in zip(waves_aug, fluxes_aug):
            wave_fix, flux_fix = resample_to_fixed(wave_aug, flux_aug, N_RESAMPLE)
            X = make_channels(wave_fix, flux_fix, win)
            X_planet.append(X)
            Y_planet.append(labels)

        planet_datasets.append({
            'name': planet_name,
            'X': np.array(X_planet, dtype=np.float32),
            'Y': np.array(Y_planet, dtype=np.float32)
        })

    rng = np.random.default_rng(seed0 + 999)

    # If we have <=3 planets, use sample-level split (data leakage acceptable for now)
    # Otherwise use planet-level split
    if len(planet_datasets) <= 3:
        print(f"  WARNING: Only {len(planet_datasets)} UV planets detected.")
        print(f"  Using sample-level split (data leakage) until more planets are added.")
        print(f"  Recommendation: Add 4+ UV planets for proper planet-level validation.")

        # Combine all planets
        X_all = np.concatenate([p['X'] for p in planet_datasets], axis=0)
        Y_all = np.concatenate([p['Y'] for p in planet_datasets], axis=0)

        # Random shuffle and split
        perm = rng.permutation(len(X_all))
        X_all, Y_all = X_all[perm], Y_all[perm]
        n_train = int(0.85 * len(X_all))

        return X_all[:n_train], Y_all[:n_train], X_all[n_train:], Y_all[n_train:]

    else:
        # Planet-level train/val split (80/20 by planet count)
        rng.shuffle(planet_datasets)
        n_val_planets = max(1, len(planet_datasets) // 5)

        train_planets = planet_datasets[n_val_planets:]
        val_planets = planet_datasets[:n_val_planets]

        print(f"  Train planets: {[p['name'] for p in train_planets]}")
        print(f"  Val planets: {[p['name'] for p in val_planets]}")

        # Combine within each split
        X_train = np.concatenate([p['X'] for p in train_planets], axis=0)
        Y_train = np.concatenate([p['Y'] for p in train_planets], axis=0)
        X_val = np.concatenate([p['X'] for p in val_planets], axis=0)
        Y_val = np.concatenate([p['Y'] for p in val_planets], axis=0)

        # Shuffle within each split
        train_perm = rng.permutation(len(X_train))
        val_perm = rng.permutation(len(X_val))

        return X_train[train_perm], Y_train[train_perm], X_val[val_perm], Y_val[val_perm]


# ============================================================================
# MODEL
# ============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron for spectral classification."""
    def __init__(self, n_resample: int, h1: int, h2: int, drop1: float, drop2: float, k: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * n_resample, h1),
            nn.ReLU(),
            nn.Dropout(drop1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(drop2),
            nn.Linear(h2, k),
        )

    def forward(self, x):
        return self.net(x)


def train_one(
    X_train, Y_train, X_val, Y_val,
    params: Dict, n_resample: int,
    epochs: int, patience: int
) -> Tuple[float, Dict[str, torch.Tensor], np.ndarray]:
    """Train model with early stopping."""
    model = MLP(
        n_resample=n_resample,
        h1=params["h1"], h2=params["h2"],
        drop1=params["drop1"], drop2=params["drop2"],
        k=len(UV_SPECIES_EXPANDED)
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)),
        batch_size=BATCH, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(Y_val)),
        batch_size=BATCH, shuffle=False
    )

    best_loss = 1e9
    best_state = None
    bad = 0

    @torch.no_grad()
    def eval_val():
        model.eval()
        tot = 0.0
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            tot += loss_fn(model(xb), yb).item() * xb.size(0)
        return tot / len(val_loader.dataset)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        v = eval_val()
        if v < best_loss - 1e-4:
            best_loss = v
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_loss = eval_val()

    # Dummy logits return for compatibility
    logits = np.zeros(len(UV_SPECIES_EXPANDED), dtype=float)

    return float(best_loss), best_state, logits


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function."""
    params = {
        "h1": trial.suggest_categorical("h1", [512, 768, 1024]),
        "h2": trial.suggest_categorical("h2", [256, 384, 512]),
        "drop1": trial.suggest_float("drop1", 0.10, 0.30),
        "drop2": trial.suggest_float("drop2", 0.10, 0.30),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-4, log=True),
        "baseline_win": trial.suggest_categorical("baseline_win", [151, 201, 251]),
    }

    X_train, Y_train, X_val, Y_val = build_augmented_dataset(
        win=params["baseline_win"],
        n_augment=N_AUGMENT_PER_PLANET,
        seed0=1000 + trial.number
    )

    vloss, _, _ = train_one(
        X_train, Y_train, X_val, Y_val,
        params=params,
        n_resample=N_RESAMPLE,
        epochs=EPOCHS_TUNE,
        patience=PATIENCE_TUNE
    )

    return float(vloss)


def main():
    """Main training pipeline."""
    print("="*60)
    print("  UV Model Training (Real Data + Augmentation)")
    print("="*60)
    print(f"DEVICE: {DEVICE}")

    uv_files = detect_uv_files()
    print(f"UV planets: {list(uv_files.keys())}")
    print(f"Augmentations per planet: {N_AUGMENT_PER_PLANET}")
    print(f"Total samples: ~{len(uv_files) * N_AUGMENT_PER_PLANET}")
    print(f"Species: {len(UV_SPECIES_EXPANDED)} (expanded)")

    with mlflow.start_run(run_name="train-uv-mlp-augmented"):
        # Log params
        try:
            mlflow.log_param("domain", "UV")
            mlflow.log_param("device", DEVICE)
            mlflow.log_param("n_augment_per_planet", N_AUGMENT_PER_PLANET)
            mlflow.log_param("n_resample", N_RESAMPLE)
            mlflow.log_param("n_trials", N_TRIALS)
            mlflow.log_param("species_count", len(UV_SPECIES_EXPANDED))
            mlflow.log_param("planets", ",".join(uv_files.keys()))
        except Exception:
            pass

        # Optuna tuning
        print("\n[STEP 1] Hyperparameter tuning...")
        t_opt = time.time()
        with mlflow.start_span(name="optuna_tuning"):
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
            best = study.best_params

        print(f"\nBest params: {best}")
        print(f"Tuning time: {time.time() - t_opt:.1f}s")

        # Final training
        print("\n[STEP 2] Final training...")
        t0 = time.time()
        with mlflow.start_span(name="final_training"):
            X_train, Y_train, X_val, Y_val = build_augmented_dataset(
                win=best["baseline_win"],
                n_augment=N_AUGMENT_PER_PLANET,
                seed0=999
            )

            print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

            vloss, state, _ = train_one(
                X_train, Y_train, X_val, Y_val,
                params=best,
                n_resample=N_RESAMPLE,
                epochs=EPOCHS_FINAL,
                patience=PATIENCE_FINAL
            )

        print(f"\nFinal val_loss: {vloss:.4f}")
        print(f"Training time: {time.time() - t0:.1f}s")

        # Save model
        print("\n[STEP 3] Saving model...")
        uv_pt = MODEL_DIR / "uv_mlp.pt"
        uv_cfg = MODEL_DIR / "uv_config.json"

        torch.save({"state_dict": state}, uv_pt)
        uv_cfg.write_text(json.dumps({
            "domain": "UV",
            "species": UV_SPECIES_EXPANDED,
            "best_params": best,
            "n_resample": N_RESAMPLE,
            "val_loss": float(vloss),
        }, indent=2))

        print(f"Saved: {uv_pt}")
        print(f"Saved: {uv_cfg}")

        try:
            mlflow.log_artifact(str(uv_pt))
            mlflow.log_artifact(str(uv_cfg))
            mlflow.log_metric("final_val_loss", float(vloss))
        except Exception:
            pass

    print("\n" + "="*60)
    print("  Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
