"""Public entry to run the LangGraph pipeline with injected params."""

import os
from .graph import RUN_GRAPH
from .state import PipelineState


def _ensure_api_key() -> None:
    if os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return
    raise RuntimeError(
        "Missing API key. Set GOOGLE_API_KEY for Gemini or OPENAI_API_KEY for OpenAI in your environment or .env file."
    )


def run_pipeline(input_path: str, modality: str | None = None) -> PipelineState:
    _ensure_api_key()
    return RUN_GRAPH({"input_path": input_path, "modality": modality or "spectral"})


__all__ = ["run_pipeline"]
