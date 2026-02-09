from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

Domain = Literal["UV", "IR", "AUTO"]

class Prediction(BaseModel):
    element: str
    probability: float
    rationale: Optional[str] = None
    color: Optional[str] = None

class AnalyzeSpectrumResponse(BaseModel):
    domain_used: Domain
    predictions: List[Prediction]
    debug: Optional[Dict[str, Any]] = None

class AnalyzePathRequest(BaseModel):
    input_path: str = Field(..., description="Local path on server/container to FITS/PKL")
    domain: Domain = "AUTO"
    top_k: int = 8
