# spectral_service/training/train_ir.py
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

# Add the training directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mlflow_utils import setup_mlflow, safe_log_param, safe_log_metric
from synthetic_generator import (
    IR_SPECIES, IR_BANDS, DEFAULT_PRIORS_IR,
    build_synthetic_from_planet,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_REAL = ROOT / "data" / "real"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IR_PKLS = {
    "SATURN": DATA_REAL / "saturn_ir.pkl",
    "URANUS": DATA_REAL / "uranus_ir.pkl",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_RESAMPLE = 1024
BATCH = 128

N_TRIALS = 20
N_SYNTH_PER_PLANET_TUNE = 900
EPOCHS_TUNE = 10
PATIENCE_TUNE = 3

N_SYNTH_PER_PLANET_FINAL = 2500
EPOCHS_FINAL = 22
PATIENCE_FINAL = 6


def load_pkl_spectrum(p: Path) -> Tuple[str, np.ndarray, np.ndarray]:
    with open(p, "rb") as f:
        d = pickle.load(f)
    target = str(d.get("target", p.stem)).upper()
    # Handle both 'wavelength' and 'wave' keys
    w = np.asarray(d.get("wavelength", d.get("wave")), dtype=float)
    y = np.asarray(d["flux"], dtype=float)
    m = np.isfinite(w) & np.isfinite(y)
    w, y = w[m], y[m]
    idx = np.argsort(w)
    return target, w[idx], y[idx]


class MLP(nn.Module):
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
    model = MLP(
        n_resample=n_resample,
        h1=params["h1"], h2=params["h2"],
        drop1=params["drop1"], drop2=params["drop2"],
        k=len(IR_SPECIES)
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

    # If best_state is None (no improvement), use current model state
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_loss = eval_val()

    model.load_state_dict(best_state)
    model.eval()

    # reference logits on Saturn if available
    if IR_PKLS["SATURN"].exists():
        _, ws, fs = load_pkl_spectrum(IR_PKLS["SATURN"])
        from synthetic_generator import resample_to_fixed, make_channels
        ws_fix, fs_fix = resample_to_fixed(ws, fs, n_resample)
        Xs = make_channels(ws_fix, fs_fix, win=params["baseline_win"])
        xt = torch.tensor(Xs[None, ...], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = model(xt)[0].cpu().numpy()
    else:
        logits = np.zeros(len(IR_SPECIES), dtype=float)

    return float(best_loss), best_state, logits


def build_multi_planet_dataset(win: int, n_per_planet: int, seed0: int):
    Xs, Ys = [], []
    for i, (planet, pkl_path) in enumerate(IR_PKLS.items()):
        if not pkl_path.exists():
            continue
        _, w, f = load_pkl_spectrum(pkl_path)
        priors = DEFAULT_PRIORS_IR.get(planet, {})
        Xp, Yp = build_synthetic_from_planet(
            w, f,
            species=IR_SPECIES,
            bands=IR_BANDS,
            priors=priors,
            n_resample=N_RESAMPLE,
            n=n_per_planet,
            win=win,
            seed=seed0 + i * 100
        )
        Xs.append(Xp)
        Ys.append(Yp)

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)

    rng = np.random.default_rng(seed0 + 999)
    perm = rng.permutation(len(X))
    X, Y = X[perm], Y[perm]
    n_train = int(0.9 * len(X))
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def objective(trial: optuna.Trial) -> float:
    params = {
        "h1": trial.suggest_categorical("h1", [256, 384, 512, 768]),
        "h2": trial.suggest_categorical("h2", [128, 192, 256, 384]),
        "drop1": trial.suggest_float("drop1", 0.10, 0.35),
        "drop2": trial.suggest_float("drop2", 0.00, 0.25),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 3e-4, log=True),
        "baseline_win": trial.suggest_categorical("baseline_win", [101, 151, 201, 251]),
        "temp": trial.suggest_float("temp", 1.0, 1.7),
    }

    X_train, Y_train, X_val, Y_val = build_multi_planet_dataset(
        win=params["baseline_win"],
        n_per_planet=N_SYNTH_PER_PLANET_TUNE,
        seed0=2000 + trial.number
    )

    vloss, _, logits = train_one(
        X_train, Y_train, X_val, Y_val,
        params=params,
        n_resample=N_RESAMPLE,
        epochs=EPOCHS_TUNE,
        patience=PATIENCE_TUNE
    )

    probs = 1.0 / (1.0 + np.exp(-logits / params["temp"]))
    nonzero = float(np.max(probs))
    penalty = 0.0
    if nonzero < 0.20:
        penalty += (0.20 - nonzero) * 1.0

    trial.set_user_attr("ref_probs", {sp: float(p) for sp, p in zip(IR_SPECIES, probs)})
    return float(vloss + penalty)


def main():
    import mlflow

    print("DEVICE:", DEVICE)
    active_planets = [k for k, v in IR_PKLS.items() if v.exists()]
    print("IR planets:", active_planets)

    # Optional: set once globally in the file
    # mlflow.set_experiment("spectral-service")

    with mlflow.start_run(run_name="train-ir-mlp"):

        # ---------- params (static) ----------
        try:
            mlflow.log_param("domain", "IR")
            mlflow.log_param("device", DEVICE)
            mlflow.log_param("n_resample", N_RESAMPLE)
            mlflow.log_param("n_trials", N_TRIALS)
            mlflow.log_param("n_synth_per_planet_final", N_SYNTH_PER_PLANET_FINAL)
            mlflow.log_param("epochs_final", EPOCHS_FINAL)
            mlflow.log_param("patience_final", PATIENCE_FINAL)
            mlflow.log_param("planets_used", ",".join(active_planets))
            mlflow.log_param("species_count", len(IR_SPECIES))
            mlflow.log_param("species", ",".join(IR_SPECIES))
        except Exception:
            pass

        # ---------- Optuna tuning span ----------
        t_opt = time.time()
        with mlflow.start_span(name="optuna_tuning") as span:
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=N_TRIALS)

            opt_time = time.time() - t_opt
            best = study.best_params

            span.set_attribute("best_objective", float(study.best_value))
            span.set_attribute("optuna_time_sec", float(opt_time))

            try:
                mlflow.log_metric("best_objective", float(study.best_value))
                mlflow.log_metric("optuna_time_sec", float(opt_time))

                for k, v in best.items():
                    mlflow.log_param(f"best_{k}", v)

                ref_probs = study.best_trial.user_attrs.get("ref_probs", {})
                if ref_probs:
                    mlflow.log_param(
                        "ref_probs_top",
                        ", ".join([f"{k}:{round(float(v),3)}" for k, v in sorted(ref_probs.items(), key=lambda x: -x[1])[:8]])
                    )
            except Exception:
                pass

        print("\nBest params:", best)
        print("Ref probs:", study.best_trial.user_attrs.get("ref_probs", {}))

        # ---------- Final training span ----------
        t0 = time.time()
        with mlflow.start_span(name="final_training") as span:
            X_train, Y_train, X_val, Y_val = build_multi_planet_dataset(
                win=best["baseline_win"],
                n_per_planet=N_SYNTH_PER_PLANET_FINAL,
                seed0=2999
            )

            try:
                mlflow.log_metric("train_samples", int(len(X_train)))
                mlflow.log_metric("val_samples", int(len(X_val)))
            except Exception:
                pass

            vloss, state, _ = train_one(
                X_train, Y_train, X_val, Y_val,
                params=best,
                n_resample=N_RESAMPLE,
                epochs=EPOCHS_FINAL,
                patience=PATIENCE_FINAL
            )

            train_time = time.time() - t0
            span.set_attribute("final_val_loss", float(vloss))
            span.set_attribute("train_time_sec", float(train_time))

            try:
                mlflow.log_metric("final_val_loss", float(vloss))
                mlflow.log_metric("train_time_sec", float(train_time))
            except Exception:
                pass

        # ---------- Save artifacts ----------
        ir_pt = MODEL_DIR / "ir_mlp.pt"
        ir_cfg = MODEL_DIR / "ir_config.json"

        torch.save({"state_dict": state}, ir_pt)
        ir_cfg.write_text(json.dumps({
            "domain": "IR",
            "species": IR_SPECIES,
            "bands": IR_BANDS,
            "best_params": best,
            "n_resample": N_RESAMPLE,
            "val_loss": float(vloss),
        }, indent=2))

        print("\nSaved:", ir_pt)
        print("Saved:", ir_cfg)
        print("Final val_loss:", round(float(vloss), 4), "| time:", round(time.time() - t0, 2), "s")

        try:
            mlflow.log_artifact(str(ir_pt))
            mlflow.log_artifact(str(ir_cfg))
        except Exception:
            pass

if __name__ == "__main__":
    main()
