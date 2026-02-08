# training/train_uv.py
import json, pickle, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import savgol_filter
import optuna
from pathlib import Path

# -------------------------
# Paths
# -------------------------
DATA_DIR = Path("Spectral Service/data/real") 
MODEL_DIR = Path("Spectral Service/models")
MODEL_DIR.mkdir(exist_ok=True)

UV_PKLS = {
    "JUPITER": DATA_DIR / "jupiter_uv.pkl",
    "SATURN":  DATA_DIR / "saturn_uv.pkl",
    "URANUS":  DATA_DIR / "uranus_uv.pkl",
}

# -------------------------
# Global config
# -------------------------
SEED = 7
N_RESAMPLE = 1024
BATCH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_SYNTH_PER_PLANET = 1200
EPOCHS_TUNE = 12
PATIENCE = 4
N_TRIALS = 25

torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

SPECIES = ["CH4", "NH3", "C2H2", "C2H6"]

BANDS = {
    "CH4": [(2350, 260), (2750, 320)],
    "NH3": [(2050, 140), (2150, 140)],
    "C2H2": [(2700, 140), (2810, 130)],
    "C2H6": [(2400, 170), (2550, 150)],
}

# -------------------------
# Utilities
# -------------------------
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def resample(w, f, n=N_RESAMPLE):
    w_new = np.linspace(w.min(), w.max(), n)
    f_new = np.interp(w_new, w, f)
    return w_new.astype(np.float32), f_new.astype(np.float32)

def compute_baseline(flux, win):
    if win % 2 == 0:
        win += 1
    win = max(11, min(win, len(flux) - 1))
    b = savgol_filter(flux, win, 3)
    b = np.clip(b, np.percentile(b, 1), np.percentile(b, 99))
    return b.astype(np.float32) + 1e-12

def make_channels(w, f, win):
    base = compute_baseline(f, win)
    r = (f / base) - 1.0
    r = (r - np.median(r)) / (np.std(r) + 1e-8)
    d1 = np.gradient(r, w)
    d2 = np.gradient(d1, w)
    return np.stack([r, d1, d2], axis=0).astype(np.float32)

# -------------------------
# Synthetic generator
# -------------------------
def build_synthetic_from_planet(w, f, win, n, seed):
    rng = np.random.default_rng(seed)
    base = compute_baseline(f, win)

    X, Y = [], []
    for _ in range(n):
        y = {
            "CH4": 1 if rng.random() < 0.85 else 0,
            "NH3": 1 if rng.random() < 0.25 else 0,
            "C2H2": 1 if rng.random() < 0.20 else 0,
            "C2H6": 1 if rng.random() < 0.20 else 0,
        }
        if sum(y.values()) == 0:
            y["CH4"] = 1

        spec = base.copy()
        x = (w - w.min()) / (w.max() - w.min())
        spec *= 1.0 + rng.normal(0, 0.01) + rng.normal(0, 0.01) * (x - 0.5)

        for sp, present in y.items():
            if not present:
                continue
            for c, w0 in BANDS[sp]:
                depth = rng.uniform(0.05, 0.25)
                width = w0 * rng.uniform(0.8, 1.3)
                spec *= (1.0 - depth * gaussian(w, c + rng.normal(0, 15), width))

        noise = rng.normal(0, 0.01 * spec.ptp(), size=len(spec))
        spec += np.convolve(noise, np.ones(7)/7, mode="same")

        X.append(make_channels(w, spec, win))
        Y.append([y[s] for s in SPECIES])

    return np.stack(X), np.array(Y, dtype=np.float32)

# -------------------------
# Model
# -------------------------
class MLP(nn.Module):
    def __init__(self, h1, h2, drop1, drop2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * N_RESAMPLE, h1),
            nn.ReLU(),
            nn.Dropout(drop1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(drop2),
            nn.Linear(h2, len(SPECIES))
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# Objective
# -------------------------
def objective(trial):
    h1 = trial.suggest_categorical("h1", [256, 512, 768])
    h2 = trial.suggest_categorical("h2", [128, 256, 384])
    drop1 = trial.suggest_float("drop1", 0.1, 0.3)
    drop2 = trial.suggest_float("drop2", 0.05, 0.25)
    lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-4, log=True)
    win = trial.suggest_categorical("baseline_win", [101, 151, 201])

    Xs, Ys = [], []
    for i, (planet, pkl) in enumerate(UV_PKLS.items()):
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        w, f = resample(d["wavelength"], d["flux"])
        Xp, Yp = build_synthetic_from_planet(
            w, f, win, N_SYNTH_PER_PLANET, seed=1000 + trial.number + i
        )
        Xs.append(Xp)
        Ys.append(Yp)

    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)

    perm = np.random.permutation(len(X))
    X, Y = X[perm], Y[perm]
    n_train = int(0.85 * len(X))

    train = DataLoader(
        TensorDataset(torch.tensor(X[:n_train]), torch.tensor(Y[:n_train])),
        batch_size=BATCH, shuffle=True
    )
    val = DataLoader(
        TensorDataset(torch.tensor(X[n_train:]), torch.tensor(Y[n_train:])),
        batch_size=BATCH
    )

    model = MLP(h1, h2, drop1, drop2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()

    best, bad = 1e9, 0
    for _ in range(EPOCHS_TUNE):
        model.train()
        for xb, yb in train:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        tot = 0
        with torch.no_grad():
            for xb, yb in val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                tot += loss_fn(model(xb), yb).item() * xb.size(0)
        tot /= len(val.dataset)

        if tot < best - 1e-4:
            best, bad = tot, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    return float(best)

# -------------------------
# Train
# -------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)

print("Best UV params:", study.best_params)

# Save config (model weights will be trained next step)
with open(MODEL_DIR / "uv_config.json", "w") as f:
    json.dump(study.best_params, f, indent=2)