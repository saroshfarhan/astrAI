"""Shared state definition for LangGraph pipeline.

This stays small on purpose; downstream nodes can append to the dict.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict


Modality = Literal["spectral", "image", "both"]


class Prediction(TypedDict, total=False):
    element: str
    probability: float
    rationale: str
    color: str  # Hex color code from model output


class PipelineState(TypedDict, total=False):
    # Input and routing
    modality: Modality
    input_path: str
    route: str  # Orchestrator decision: "spectral", "image", or "both"
    active_agents: List[str]  # List of agents to execute
    
    # Model outputs (separate for each specialist agent)
    spectral_predictions: List[Prediction]
    image_predictions: List[Prediction]
    
    # Legacy field for backward compatibility (will be deprecated)
    predictions: List[Prediction]
    model_name: str
    
    # Inference and consolidation
    inference_results: Dict[str, Any]  # Consolidated multi-modal results
    knowledge_base: Dict[str, Dict[str, Any]]  # Session-scoped KB: {element: {name, color, sources, confidence}}
    
    # Validation
    validated_results: List[Dict[str, Any]]  # Results after validation
    validation_flags: List[str]  # Quality flags (e.g., "LOW_CONFIDENCE")
    
    # Metadata and logging
    metadata: Dict[str, Any]  # Now includes spectral_summary and spectral_features_description
    log_refs: Dict[str, str]
    errors: List[str]
    prompt_version: str


DEFAULT_STATE: PipelineState = {
    # Routing
    "active_agents": [],
    
    # Model outputs
    "spectral_predictions": [],
    "image_predictions": [],
    "predictions": [],  # Legacy
    
    # Inference
    "inference_results": {},
    "knowledge_base": {},
    
    # Validation
    "validated_results": [],
    "validation_flags": [],
    
    # Metadata
    "metadata": {},
    "log_refs": {},
    "errors": [],
    "prompt_version": "v2",  # Updated for multi-agent architecture
}

