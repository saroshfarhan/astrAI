"""Agent modules for the multi-agent pipeline.

This package contains all specialist agents:
- orchestrator: Routes to appropriate model agents
- spectral_model: Processes spectral data (FITS/CSV)
- image_model: Processes image data (PNG/JPG)
- inference: Consolidates predictions and builds KB
- validator: Validates results and flags issues
- reporter: Generates natural language reports
- feedback: Captures user feedback (to be implemented)
"""

# Import all implemented agents
from .orchestrator import OrchestratorAgent
from .spectral_model import SpectralModelAgent
from .image_model import ImageModelAgent
from .inference import InferenceAgent
from .validator import ValidatorAgent
from .reporter import ReporterAgent
# from .feedback import FeedbackAgent  # To be implemented

__all__ = [
    "OrchestratorAgent",
    "SpectralModelAgent",
    "ImageModelAgent",
    "InferenceAgent",
    "ValidatorAgent",
    "ReporterAgent",
    # "FeedbackAgent",  # To be implemented
]

