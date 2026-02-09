# spectral_service/training/evaluate_real.py
from __future__ import annotations

import json, pickle, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add the training directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent))

from synthetic_generator import resample_to_fixed, make_channels

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_REAL = ROOT / "data" / "real"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def load_pkl(p: Path):
    with open(p, "rb") as f:
        d = pickle.load(f)
    # Handle both 'wavelength' and 'wave' keys
    w = np.asarray(d.get("wavelength", d.get("wave")), dtype=float)
    y = np.asarray(d["flux"], dtype=float)
    m = np.isfinite(w) & np.isfinite(y)
    w, y = w[m], y[m]
    idx = np.argsort(w)
    return str(d.get("target", p.stem)).upper(), w[idx], y[idx]


def run_model(domain: str, pkl_path: Path):
    cfg_path = MODEL_DIR / f"{domain.lower()}_config.json"
    pt_path  = MODEL_DIR / f"{domain.lower()}_mlp.pt"

    cfg = json.loads(cfg_path.read_text())
    species = cfg["species"]
    bp = cfg["best_params"]
    n_resample = int(cfg["n_resample"])

    ckpt = torch.load(pt_path, map_location="cpu")
    state = ckpt["state_dict"]

    model = MLP(
        n_resample=n_resample,
        h1=bp["h1"], h2=bp["h2"],
        drop1=bp["drop1"], drop2=bp["drop2"],
        k=len(species),
    )
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    tgt, w, f = load_pkl(pkl_path)
    w_fix, f_fix = resample_to_fixed(w, f, n_resample)
    X = make_channels(w_fix, f_fix, win=bp["baseline_win"])
    xt = torch.tensor(X[None, ...], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(xt)[0].cpu().numpy()

    T = float(bp.get("temp", 1.0))
    probs = 1.0 / (1.0 + np.exp(-logits / T))

    top = sorted(zip(species, probs), key=lambda x: -x[1])[:8]
    return tgt, top


def main():
    uv_files = [
        DATA_REAL / "jupiter_uv.pkl",
        DATA_REAL / "saturn_uv.pkl",
        DATA_REAL / "uranus_uv.pkl",
        DATA_REAL / "mars_uv.pkl",
    ]
    ir_files = [
        DATA_REAL / "saturn_ir.pkl",
        DATA_REAL / "uranus_ir.pkl",
    ]

    print("\n=== UV real inference ===")
    for p in uv_files:
        if not p.exists():
            continue
        tgt, top = run_model("uv", p)
        print("\n", tgt, "|", p.name)
        for sp, pr in top:
            print(f"  {sp:>5}: {pr:.3f}")

    print("\n=== IR real inference ===")
    for p in ir_files:
        if not p.exists():
            continue
        tgt, top = run_model("ir", p)
        print("\n", tgt, "|", p.name)
        for sp, pr in top:
            print(f"  {sp:>5}: {pr:.3f}")


if __name__ == "__main__":
    main()
