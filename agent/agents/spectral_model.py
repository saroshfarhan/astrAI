"""Spectral Model Agent - Processes spectral data (FITS/CSV files).

This agent handles spectral analysis by loading astronomical spectral data
and running ML model inference (mocked initially, will integrate real model via tool calls).
"""

import time
from pathlib import Path
from typing import List, Dict, Any
import mlflow

from ..base import BaseAgent, validate_state_fields
from ..state import PipelineState, Prediction


class SpectralModelAgent(BaseAgent):
    """Processes spectral data and returns element predictions.
    
    Responsibilities:
    - Load FITS/CSV spectral data
    - Preprocess data (normalization, feature extraction)
    - Call spectral analysis tool (mocked initially)
    - Return structured predictions with elements, probabilities, rationales, colors
    """
    
    def __init__(self, model_name: str = "mock-spectral-v1"):
        """Initialize spectral model agent.
        
        Args:
            model_name: Name of the spectral analysis model
        """
        super().__init__("SpectralModelAgent")
        self.model_name = model_name
    
    def process(self, state: PipelineState) -> PipelineState:
        """Process spectral data and generate predictions.
        
        Args:
            state: Pipeline state with input_path
            
        Returns:
            Updated state with spectral_predictions
        """
        # Validate required fields
        validate_state_fields(state, ["input_path"])
        
        input_path = state["input_path"]
        file_path = Path(input_path)
        
        # Log input parameters
        try:
            mlflow.log_param("spectral_input_file", file_path.name)
            mlflow.log_param("spectral_file_extension", file_path.suffix)
            mlflow.log_param("spectral_model_name", self.model_name)
        except Exception:
            pass
        
        # Load and preprocess spectral data
        try:
            with mlflow.start_span(name="load_spectral_data") as span:
                spectral_data = self._load_spectral_data(input_path)
                
                span.set_attribute("file_exists", spectral_data["exists"])
                span.set_attribute("file_size_bytes", spectral_data.get("size", 0))
                
                if spectral_data["exists"]:
                    mlflow.log_metric("spectral_file_load_success", 1)
                else:
                    mlflow.log_metric("spectral_file_load_success", 0)
                    
        except Exception as exc:
            mlflow.log_metric("spectral_load_error", 1)
            state.setdefault("errors", []).append(f"Spectral data load failed: {exc}")
            state["spectral_predictions"] = []
            return state
        
        # Preprocess data
        try:
            with mlflow.start_span(name="preprocess_spectral_data") as span:
                preprocessed = self._preprocess_data(spectral_data)
                
                span.set_attribute("num_wavelengths", preprocessed.get("num_wavelengths", 0))
                span.set_attribute("wavelength_range", preprocessed.get("wavelength_range", "unknown"))
                
                mlflow.log_param("spectral_num_wavelengths", preprocessed.get("num_wavelengths", 0))
                mlflow.log_metric("spectral_preprocess_success", 1)
                
        except Exception as exc:
            mlflow.log_metric("spectral_preprocess_error", 1)
            state.setdefault("errors", []).append(f"Spectral preprocessing failed: {exc}")
            state["spectral_predictions"] = []
            return state
        
        # Run model inference (mocked)
        try:
            with mlflow.start_span(name="spectral_model_inference") as span:
                start_time = time.time()
                
                predictions = self._run_model_inference(preprocessed)
                
                inference_time = time.time() - start_time
                span.set_attribute("inference_time_ms", int(inference_time * 1000))
                span.set_attribute("num_predictions", len(predictions))
                
                mlflow.log_metric("spectral_inference_time_ms", inference_time * 1000)
                mlflow.log_metric("spectral_prediction_count", len(predictions))
                mlflow.log_metric("spectral_inference_success", 1)
                
        except Exception as exc:
            mlflow.log_metric("spectral_inference_error", 1)
            state.setdefault("errors", []).append(f"Spectral model inference failed: {exc}")
            state["spectral_predictions"] = []
            return state
        
        # Format predictions
        try:
            with mlflow.start_span(name="format_spectral_predictions") as span:
                formatted_predictions = self._format_predictions(predictions)
                
                # Calculate metrics
                confidences = [p["probability"] for p in formatted_predictions]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                low_confidence_count = sum(1 for c in confidences if c < 0.4)
                
                span.set_attribute("avg_confidence", round(avg_confidence, 3))
                span.set_attribute("low_confidence_count", low_confidence_count)
                
                mlflow.log_metric("spectral_avg_confidence", avg_confidence)
                mlflow.log_metric("spectral_low_confidence_count", low_confidence_count)
                mlflow.log_metric("spectral_high_confidence_count", len(confidences) - low_confidence_count)
                
                # Log individual element detections
                elements = [p["element"] for p in formatted_predictions]
                mlflow.log_param("spectral_elements_detected", ", ".join(elements[:10]))  # First 10
                
        except Exception as exc:
            mlflow.log_metric("spectral_format_error", 1)
            state.setdefault("errors", []).append(f"Spectral prediction formatting failed: {exc}")
            state["spectral_predictions"] = []
            return state
        
        # Update state
        state["spectral_predictions"] = formatted_predictions
        
        # Log final summary
        try:
            mlflow.log_param("spectral_final_prediction_count", len(formatted_predictions))
            mlflow.log_metric("spectral_agent_success", 1)
        except Exception:
            pass
        
        return state
    
    def _load_spectral_data(self, input_path: str) -> Dict[str, Any]:
        """Load spectral data from FITS or CSV file.
        
        Args:
            input_path: Path to spectral data file
            
        Returns:
            Dictionary with spectral data metadata
        """
        file_path = Path(input_path)
        
        # Check if file exists
        if not file_path.exists():
            return {"exists": False, "path": input_path}
        
        # Mock loading - in real implementation, use astropy for FITS
        # from astropy.io import fits
        # hdul = fits.open(input_path)
        # spectral_data = hdul[1].data
        
        return {
            "exists": True,
            "path": input_path,
            "size": file_path.stat().st_size if file_path.exists() else 0,
            "format": file_path.suffix
        }
    
    def _preprocess_data(self, spectral_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess spectral data for model inference.
        
        Args:
            spectral_data: Raw spectral data
            
        Returns:
            Preprocessed data ready for model
        """
        # Mock preprocessing - in real implementation:
        # - Normalize flux values
        # - Extract features
        # - Handle missing data
        
        return {
            "num_wavelengths": 1000,  # Mock value
            "wavelength_range": "400-700nm",  # Mock value
            "preprocessed": True
        }
    
    def _run_model_inference(self, preprocessed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run spectral analysis model (mocked).
        
        This will be replaced with real model tool call when available.
        
        Args:
            preprocessed_data: Preprocessed spectral data
            
        Returns:
            List of raw predictions from model
        """
        # Mock predictions - realistic structure for astronomical spectral analysis
        # In real implementation, this will call the actual ML model via tool
        
        mock_predictions = [
            {
                "element": "H",
                "probability": 0.89,
                "rationale": "Strong Balmer series lines detected at 656.3nm (H-alpha), 486.1nm (H-beta)",
                "color": "#FF0000",
                "wavelengths": [656.3, 486.1, 434.0]
            },
            {
                "element": "He",
                "probability": 0.72,
                "rationale": "Helium D3 line present at 587.6nm, He I line at 667.8nm",
                "color": "#FFFF00",
                "wavelengths": [587.6, 667.8]
            },
            {
                "element": "O",
                "probability": 0.65,
                "rationale": "Oxygen forbidden lines [O III] at 495.9nm and 500.7nm",
                "color": "#00FF00",
                "wavelengths": [495.9, 500.7]
            },
            {
                "element": "N",
                "probability": 0.48,
                "rationale": "Nitrogen lines detected but weak signal",
                "color": "#0000FF",
                "wavelengths": [658.4, 654.8]
            },
            {
                "element": "Fe",
                "probability": 0.35,
                "rationale": "Iron absorption lines present, low confidence due to noise",
                "color": "#CC5500",
                "wavelengths": [438.4, 527.0]
            }
        ]
        
        return mock_predictions
    
    def _format_predictions(self, raw_predictions: List[Dict[str, Any]]) -> List[Prediction]:
        """Format raw model predictions to standard Prediction format.
        
        Args:
            raw_predictions: Raw predictions from model
            
        Returns:
            List of formatted Prediction objects
        """
        formatted = []
        
        for pred in raw_predictions:
            formatted.append({
                "element": pred["element"],
                "probability": pred["probability"],
                "rationale": pred["rationale"],
                "color": pred["color"]
            })
        
        return formatted
