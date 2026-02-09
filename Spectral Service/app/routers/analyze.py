from __future__ import annotations
import json
import time
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter, UploadFile, File, HTTPException
import mlflow

# Add the app directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas import AnalyzeSpectrumResponse, Prediction
from utils.io import load_spectrum
from utils.preprocess import resample_to_fixed, make_channels
from utils.fusion import fuse_uv_ir

router = APIRouter()

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

# ---------- MLP definition must match training ----------
class MLP(nn.Module):
    def __init__(self, n_resample: int, h1: int, h2: int, drop1: float, drop2: float, k: int):
        super().__init__()
        C, N = 3, n_resample
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C * N, h1),
            nn.ReLU(),
            nn.Dropout(drop1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(drop2),
            nn.Linear(h2, k),
        )

    def forward(self, x):
        return self.net(x)

def _load_model(domain: str):
    cfg_path = MODEL_DIR / f"{domain.lower()}_config.json"
    pt_path  = MODEL_DIR / f"{domain.lower()}_mlp.pt"
    if not cfg_path.exists() or not pt_path.exists():
        return None, None

    cfg = json.loads(cfg_path.read_text())
    bp = cfg["best_params"]
    species = cfg["species"]
    n_resample = int(cfg["n_resample"])
    k = len(species)

    model = MLP(
        n_resample=n_resample,
        h1=int(bp["h1"]),
        h2=int(bp["h2"]),
        drop1=float(bp["drop1"]),
        drop2=float(bp["drop2"]),
        k=k,
    )

    blob = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, cfg

UV_MODEL, UV_CFG = _load_model("uv")
IR_MODEL, IR_CFG = _load_model("ir")

def _infer(model, cfg, wave: np.ndarray, flux: np.ndarray) -> list[float]:
    n_resample = int(cfg["n_resample"])
    win = int(cfg["best_params"]["baseline_win"])

    w_fix, f_fix = resample_to_fixed(wave, flux, n_resample)
    X = make_channels(w_fix, f_fix, baseline_win=win)  # (3,N)
    xt = torch.tensor(X[None, ...], dtype=torch.float32)

    with torch.no_grad():
        logits = model(xt)[0]
        probs = torch.sigmoid(logits).cpu().numpy().astype(float).tolist()
    return probs

@router.post("/analyze_spectrum", response_model=AnalyzeSpectrumResponse)
async def analyze_spectrum(file: UploadFile = File(...), top_k: int = 8):
    with mlflow.start_span(name="api_analyze_spectrum") as span:
        t0 = time.time()

        # ---- save upload to temp ----
        suffix = Path(file.filename).suffix.lower()
        tmp = Path("._tmp_upload" + suffix)
        data = await file.read()
        tmp.write_bytes(data)

        try:
            mlflow.log_param("spectral_input_file", file.filename)
            mlflow.log_param("spectral_file_extension", suffix)
        except Exception:
            pass

        # ---- load ----
        try:
            with mlflow.start_span(name="load_spectrum") as sp2:
                w, f, meta = load_spectrum(str(tmp))
                sp2.set_attribute("points", int(len(w)))
                try:
                    mlflow.log_metric("spectral_points", int(len(w)))
                except Exception:
                    pass
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Load failed: {e}")
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

        # ---- infer ----
        uv_probs = None
        ir_probs = None

        with mlflow.start_span(name="inference") as sp3:
            if UV_MODEL is not None and UV_CFG is not None:
                uv_probs = _infer(UV_MODEL, UV_CFG, w, f)

            if IR_MODEL is not None and IR_CFG is not None:
                ir_probs = _infer(IR_MODEL, IR_CFG, w, f)

            sp3.set_attribute("uv_loaded", UV_MODEL is not None)
            sp3.set_attribute("ir_loaded", IR_MODEL is not None)

        # ---- fuse ----
        fused = fuse_uv_ir(
            uv_species=UV_CFG["species"] if UV_CFG else [],
            uv_probs=uv_probs,
            ir_species=IR_CFG["species"] if IR_CFG else [],
            ir_probs=ir_probs,
            top_k=top_k,
        )

        domain_used = fused["domain_used"]
        final_pairs = fused["final"]
        debug = fused.get("debug", {})

        # ---- format predictions ----
        preds = []
        for elem, prob in final_pairs:
            preds.append(Prediction(
                element=str(elem),
                probability=float(prob),
                rationale=None,
                color=None
            ))

        dt_ms = (time.time() - t0) * 1000
        span.set_attribute("inference_time_ms", int(dt_ms))
        span.set_attribute("predictions", len(preds))

        try:
            mlflow.log_metric("spectral_inference_time_ms", dt_ms)
            mlflow.log_metric("spectral_prediction_count", len(preds))
            mlflow.log_param("spectral_domain_used", domain_used)
            if preds:
                mlflow.log_param("spectral_top", ", ".join([f"{p.element}:{round(p.probability,3)}" for p in preds[:8]]))
        except Exception:
            pass

        return AnalyzeSpectrumResponse(
            domain_used=domain_used,
            predictions=preds,
            debug=debug
        )
