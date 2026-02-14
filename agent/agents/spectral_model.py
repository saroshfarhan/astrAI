"""Spectral Model Agent - Processes spectral data (FITS/CSV files).

This agent handles spectral analysis by loading astronomical spectral data
and running ML model inference via Spectral Service FastAPI endpoint.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any
import mlflow
import requests

from ..base import BaseAgent, validate_state_fields
from ..state import PipelineState, Prediction


class SpectralModelAgent(BaseAgent):
    """Processes spectral data and returns element predictions.

    Responsibilities:
    - Load FITS/CSV/PKL spectral data
    - Call Spectral Service FastAPI endpoint
    - Return structured predictions with elements, probabilities, rationales, colors
    """

    def __init__(self, model_name: str = "spectral-mlp-uv-ir", service_url: str = None):
        """Initialize spectral model agent.

        Args:
            model_name: Name of the spectral analysis model
            service_url: URL of the Spectral Service (default: env var or http://localhost:8001)
        """
        super().__init__("SpectralModelAgent")
        self.model_name = model_name
        self.service_url = service_url or os.getenv("SPECTRAL_SERVICE_URL", "http://localhost:8001")

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
            mlflow.log_param("spectral_service_url", self.service_url)
        except Exception:
            pass

        # Validate file exists
        if not file_path.exists():
            error_msg = f"Input file not found: {input_path}"
            mlflow.log_metric("spectral_file_load_success", 0)
            state.setdefault("errors", []).append(error_msg)
            state["spectral_predictions"] = []
            return state

        # Log file metadata
        try:
            with mlflow.start_span(name="load_spectral_data") as span:
                file_size = file_path.stat().st_size
                span.set_attribute("file_exists", True)
                span.set_attribute("file_size_bytes", file_size)
                mlflow.log_metric("spectral_file_load_success", 1)
        except Exception:
            pass

        # Call Spectral Service endpoint
        spectral_summary = None
        try:
            with mlflow.start_span(name="spectral_service_request") as span:
                start_time = time.time()

                predictions, spectral_summary = self._call_spectral_service(input_path)

                request_time = time.time() - start_time
                span.set_attribute("request_time_ms", int(request_time * 1000))
                span.set_attribute("num_predictions", len(predictions))

                mlflow.log_metric("spectral_inference_time_ms", request_time * 1000)
                mlflow.log_metric("spectral_prediction_count", len(predictions))
                mlflow.log_metric("spectral_inference_success", 1)

        except Exception as exc:
            mlflow.log_metric("spectral_inference_error", 1)
            state.setdefault("errors", []).append(f"Spectral Service call failed: {exc}")
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

        # Store spectral summary in metadata for LLM visibility
        if spectral_summary:
            state.setdefault("metadata", {})["spectral_summary"] = spectral_summary
            # Also store the natural language description separately for easy access
            state["metadata"]["spectral_features_description"] = spectral_summary.get("natural_language", "")

        # Log final summary
        try:
            mlflow.log_param("spectral_final_prediction_count", len(formatted_predictions))
            mlflow.log_metric("spectral_agent_success", 1)
        except Exception:
            pass

        return state

    def _call_spectral_service(self, input_path: str) -> List[Dict[str, Any]]:
        """Call Spectral Service FastAPI endpoint.

        Args:
            input_path: Path to spectral data file

        Returns:
            List of predictions from the service

        Raises:
            Exception: If service call fails
        """
        endpoint = f"{self.service_url}/analyze_spectrum"
        file_path = Path(input_path)

        # Check service health first
        try:
            health_response = requests.get(f"{self.service_url}/", timeout=2)
            if health_response.status_code != 200:
                raise Exception(f"Spectral Service unhealthy: {health_response.status_code}")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Spectral Service at {self.service_url}. Is it running?")
        except requests.exceptions.Timeout:
            raise Exception(f"Spectral Service timeout at {self.service_url}")

        # Upload file and get predictions
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, self._get_mime_type(file_path))}
                params = {'top_k': 8}

                response = requests.post(
                    endpoint,
                    files=files,
                    params=params,
                    timeout=60  # 60 second timeout for model inference
                )

            response.raise_for_status()
            result = response.json()

            # Extract predictions from response
            predictions = result.get("predictions", [])

            # Log domain used for debugging
            domain_used = result.get("domain_used", "UNKNOWN")
            try:
                mlflow.log_param("spectral_domain_used", domain_used)
            except Exception:
                pass

            # Extract spectral summary (if available)
            spectral_summary = result.get("spectral_summary")
            if spectral_summary:
                try:
                    mlflow.log_param("spectral_summary_available", True)
                    mlflow.log_param("spectral_features_natural_language",
                                   spectral_summary.get("natural_language", "")[:500])
                except Exception:
                    pass

            return predictions, spectral_summary

        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP error from Spectral Service: {e.response.status_code} - {e.response.text}")
        except requests.exceptions.Timeout:
            raise Exception("Spectral Service request timeout (60s)")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request error: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error calling Spectral Service: {str(e)}")

    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for file upload.

        Args:
            file_path: Path to file

        Returns:
            MIME type string
        """
        suffix = file_path.suffix.lower()
        mime_types = {
            '.fits': 'application/fits',
            '.fit': 'application/fits',
            '.fts': 'application/fits',
            '.pkl': 'application/octet-stream',
            '.pickle': 'application/octet-stream',
            '.csv': 'text/csv'
        }
        return mime_types.get(suffix, 'application/octet-stream')

    def _format_predictions(self, raw_predictions: List[Dict[str, Any]]) -> List[Prediction]:
        """Format Spectral Service predictions to agent Prediction format.

        Args:
            raw_predictions: Raw predictions from Spectral Service

        Returns:
            List of formatted Prediction objects
        """
        formatted = []

        for pred in raw_predictions:
            # Map Spectral Service response to agent format
            # Service returns: {"element": str, "probability": float, "rationale": str|None, "color": str|None}
            formatted.append({
                "element": pred.get("element", "Unknown"),
                "probability": pred.get("probability", 0.0),
                "rationale": pred.get("rationale") or f"Detected {pred.get('element', 'element')} with {pred.get('probability', 0.0):.1%} confidence",
                "color": pred.get("color") or self._generate_default_color(pred.get("element", ""))
            })

        return formatted

    def _generate_default_color(self, element: str) -> str:
        """Generate a default color for an element if not provided.

        Args:
            element: Element symbol

        Returns:
            Hex color code
        """
        # Simple hash-based color generation for consistency
        hash_val = sum(ord(c) for c in element.upper())
        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A",
            "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E2"
        ]
        return colors[hash_val % len(colors)]
