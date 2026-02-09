"""Image Model Agent - Processes image data (PNG/JPG/FITS images).

This agent handles image analysis by loading astronomical images
and running ML model inference (mocked initially, will integrate real model via tool calls).
"""

import time
from pathlib import Path
from typing import List, Dict, Any
import mlflow

from ..base import BaseAgent, validate_state_fields
from ..state import PipelineState, Prediction


class ImageModelAgent(BaseAgent):
    """Processes image data and returns element/feature predictions.
    
    Responsibilities:
    - Load and preprocess image files
    - Call image analysis tool (mocked initially)
    - Extract color and morphology features
    - Return structured predictions with elements, probabilities, rationales, colors
    """
    
    def __init__(self, model_name: str = "mock-image-v1"):
        """Initialize image model agent.
        
        Args:
            model_name: Name of the image analysis model
        """
        super().__init__("ImageModelAgent")
        self.model_name = model_name
    
    def process(self, state: PipelineState) -> PipelineState:
        """Process image data and generate predictions.
        
        Args:
            state: Pipeline state with input_path
            
        Returns:
            Updated state with image_predictions
        """
        # Validate required fields
        validate_state_fields(state, ["input_path"])
        
        input_path = state["input_path"]
        file_path = Path(input_path)
        
        # Log input parameters
        try:
            mlflow.log_param("image_input_file", file_path.name)
            mlflow.log_param("image_file_extension", file_path.suffix)
            mlflow.log_param("image_model_name", self.model_name)
        except Exception:
            pass
        
        # Load image
        try:
            with mlflow.start_span(name="load_image") as span:
                image_data = self._load_image(input_path)
                
                span.set_attribute("file_exists", image_data["exists"])
                span.set_attribute("file_size_bytes", image_data.get("size", 0))
                
                if image_data["exists"]:
                    mlflow.log_metric("image_file_load_success", 1)
                else:
                    mlflow.log_metric("image_file_load_success", 0)
                    
        except Exception as exc:
            mlflow.log_metric("image_load_error", 1)
            state.setdefault("errors", []).append(f"Image load failed: {exc}")
            state["image_predictions"] = []
            return state
        
        # Preprocess image
        try:
            with mlflow.start_span(name="preprocess_image") as span:
                preprocessed = self._preprocess_image(image_data)
                
                span.set_attribute("image_width", preprocessed.get("width", 0))
                span.set_attribute("image_height", preprocessed.get("height", 0))
                span.set_attribute("image_format", preprocessed.get("format", "unknown"))
                
                mlflow.log_param("image_dimensions", f"{preprocessed.get('width', 0)}x{preprocessed.get('height', 0)}")
                mlflow.log_param("image_format", preprocessed.get("format", "unknown"))
                mlflow.log_metric("image_preprocess_success", 1)
                
        except Exception as exc:
            mlflow.log_metric("image_preprocess_error", 1)
            state.setdefault("errors", []).append(f"Image preprocessing failed: {exc}")
            state["image_predictions"] = []
            return state
        
        # Run model inference (mocked)
        try:
            with mlflow.start_span(name="image_model_inference") as span:
                start_time = time.time()
                
                predictions = self._run_model_inference(preprocessed)
                
                inference_time = time.time() - start_time
                span.set_attribute("inference_time_ms", int(inference_time * 1000))
                span.set_attribute("num_predictions", len(predictions))
                
                mlflow.log_metric("image_inference_time_ms", inference_time * 1000)
                mlflow.log_metric("image_prediction_count", len(predictions))
                mlflow.log_metric("image_inference_success", 1)
                
        except Exception as exc:
            mlflow.log_metric("image_inference_error", 1)
            state.setdefault("errors", []).append(f"Image model inference failed: {exc}")
            state["image_predictions"] = []
            return state
        
        # Analyze colors
        try:
            with mlflow.start_span(name="color_analysis") as span:
                color_features = self._analyze_colors(preprocessed)
                
                span.set_attribute("dominant_colors_count", len(color_features.get("dominant_colors", [])))
                span.set_attribute("color_extraction_success", color_features.get("success", False))
                
                mlflow.log_metric("image_color_extraction_success", 1 if color_features.get("success") else 0)
                mlflow.log_param("image_dominant_colors", ", ".join(color_features.get("dominant_colors", [])[:5]))
                
        except Exception as exc:
            mlflow.log_metric("image_color_analysis_error", 1)
            # Continue even if color analysis fails
            color_features = {"success": False}
        
        # Format predictions
        try:
            with mlflow.start_span(name="format_image_predictions") as span:
                formatted_predictions = self._format_predictions(predictions)
                
                # Calculate metrics
                confidences = [p["probability"] for p in formatted_predictions]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                low_confidence_count = sum(1 for c in confidences if c < 0.4)
                
                span.set_attribute("avg_confidence", round(avg_confidence, 3))
                span.set_attribute("low_confidence_count", low_confidence_count)
                
                mlflow.log_metric("image_avg_confidence", avg_confidence)
                mlflow.log_metric("image_low_confidence_count", low_confidence_count)
                mlflow.log_metric("image_high_confidence_count", len(confidences) - low_confidence_count)
                
                # Log individual element detections
                elements = [p["element"] for p in formatted_predictions]
                mlflow.log_param("image_elements_detected", ", ".join(elements[:10]))  # First 10
                
        except Exception as exc:
            mlflow.log_metric("image_format_error", 1)
            state.setdefault("errors", []).append(f"Image prediction formatting failed: {exc}")
            state["image_predictions"] = []
            return state
        
        # Update state
        state["image_predictions"] = formatted_predictions
        
        # Log final summary
        try:
            mlflow.log_param("image_final_prediction_count", len(formatted_predictions))
            mlflow.log_metric("image_agent_success", 1)
        except Exception:
            pass
        
        return state
    
    def _load_image(self, input_path: str) -> Dict[str, Any]:
        """Load image file.
        
        Args:
            input_path: Path to image file
            
        Returns:
            Dictionary with image metadata
        """
        file_path = Path(input_path)
        
        # Check if file exists
        if not file_path.exists():
            return {"exists": False, "path": input_path}
        
        # Mock loading - in real implementation, use PIL or opencv
        # from PIL import Image
        # img = Image.open(input_path)
        
        return {
            "exists": True,
            "path": input_path,
            "size": file_path.stat().st_size if file_path.exists() else 0,
            "format": file_path.suffix
        }
    
    def _preprocess_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess image for model inference.
        
        Args:
            image_data: Raw image data
            
        Returns:
            Preprocessed image ready for model
        """
        # Mock preprocessing - in real implementation:
        # - Resize to model input size
        # - Normalize pixel values
        # - Convert color spaces if needed
        
        return {
            "width": 1024,  # Mock value
            "height": 768,  # Mock value
            "format": "RGB",
            "preprocessed": True
        }
    
    def _analyze_colors(self, preprocessed_image: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dominant colors in image.
        
        Args:
            preprocessed_image: Preprocessed image data
            
        Returns:
            Color analysis results
        """
        # Mock color analysis - in real implementation:
        # - Extract dominant colors using k-means
        # - Map colors to elements/minerals
        
        return {
            "success": True,
            "dominant_colors": ["#CC5500", "#8B4513", "#CD853F"],  # Reddish/brown tones
            "color_distribution": {"red": 0.45, "brown": 0.35, "tan": 0.20}
        }
    
    def _run_model_inference(self, preprocessed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run image analysis model (mocked).
        
        This will be replaced with real model tool call when available.
        
        Args:
            preprocessed_data: Preprocessed image data
            
        Returns:
            List of raw predictions from model
        """
        # Mock predictions - realistic structure for astronomical image analysis
        # In real implementation, this will call the actual ML model via tool
        
        mock_predictions = [
            {
                "element": "Fe",
                "probability": 0.78,
                "rationale": "Reddish-brown coloration indicates iron oxide presence in surface materials",
                "color": "#CC5500"
            },
            {
                "element": "Si",
                "probability": 0.65,
                "rationale": "Tan/beige regions suggest silicate minerals",
                "color": "#CD853F"
            },
            {
                "element": "Mg",
                "probability": 0.52,
                "rationale": "Dark patches may indicate magnesium-rich basaltic composition",
                "color": "#8B4513"
            },
            {
                "element": "Al",
                "probability": 0.41,
                "rationale": "Lighter regions could contain aluminum silicates, low confidence",
                "color": "#D3D3D3"
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
