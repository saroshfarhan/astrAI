from __future__ import annotations
from typing import Dict, List, Tuple

def topk_from_probs(species: List[str], probs: List[float], k: int) -> List[Tuple[str, float]]:
    pairs = sorted(zip(species, probs), key=lambda x: -x[1])
    return pairs[:k]

def fuse_uv_ir(
    uv_species: List[str],
    uv_probs: List[float] | None,
    ir_species: List[str],
    ir_probs: List[float] | None,
    top_k: int = 8,
) -> Dict:
    """
    Simple fusion:
    - If only one domain is present, return that.
    - If both present, combine on union by max(prob_uv, prob_ir).
    """
    if uv_probs is None and ir_probs is None:
        return {"domain_used": "AUTO", "final": [], "debug": {"reason": "no_probs"}}

    if ir_probs is None:
        top = topk_from_probs(uv_species, uv_probs, top_k)
        return {"domain_used": "UV", "final": top, "debug": {"fusion_rule": "uv_only", "uv_top": top}}

    if uv_probs is None:
        top = topk_from_probs(ir_species, ir_probs, top_k)
        return {"domain_used": "IR", "final": top, "debug": {"fusion_rule": "ir_only", "ir_top": top}}

    uv_map = {s: float(p) for s, p in zip(uv_species, uv_probs)}
    ir_map = {s: float(p) for s, p in zip(ir_species, ir_probs)}
    all_keys = sorted(set(uv_map) | set(ir_map))

    fused = []
    for s in all_keys:
        fused.append((s, max(uv_map.get(s, 0.0), ir_map.get(s, 0.0))))
    fused.sort(key=lambda x: -x[1])

    return {
        "domain_used": "AUTO",
        "final": fused[:top_k],
        "debug": {
            "fusion_rule": "max_union",
            "uv_top": topk_from_probs(uv_species, uv_probs, min(top_k, 10)),
            "ir_top": topk_from_probs(ir_species, ir_probs, min(top_k, 10)),
            "final_top": fused[:min(top_k, 10)],
        },
    }
