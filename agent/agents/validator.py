"""Validator Agent - Validates predictions and flags quality issues.

This agent performs quality checks on predictions, validates confidence thresholds,
checks for consistency, and flags potential issues for user review.
"""

from typing import List, Dict, Any
import mlflow

from ..base import BaseAgent
from ..state import PipelineState


class ValidatorAgent(BaseAgent):
    """Validates predictions and applies quality checks.
    
    Responsibilities:
    - Validate confidence thresholds
    - Check prediction consistency
    - Detect anomalies
    - Flag low-quality predictions
    - Store validation results
    """
    
    def __init__(self, confidence_threshold: float = 0.4):
        """Initialize validator agent.
        
        Args:
            confidence_threshold: Minimum confidence for valid predictions
        """
        super().__init__("ValidatorAgent")
        self.confidence_threshold = confidence_threshold
    
    def process(self, state: PipelineState) -> PipelineState:
        """Validate predictions and flag issues.
        
        Args:
            state: Pipeline state with inference_results
            
        Returns:
            Updated state with validated_results and validation_flags
        """
        # Get predictions from inference results
        inference_results = state.get("inference_results", {})
        predictions = inference_results.get("predictions", [])
        
        # Log input
        try:
            mlflow.log_param("validator_num_predictions", len(predictions))
            mlflow.log_param("validator_confidence_threshold", self.confidence_threshold)
        except Exception:
            pass
        
        if not predictions:
            state["validated_results"] = []
            state["validation_flags"] = []
            return state
        
        # Confidence validation
        try:
            with mlflow.start_span(name="confidence_check") as span:
                confidence_results = self._check_confidence(predictions)
                
                span.set_attribute("low_confidence_count", confidence_results["low_count"])
                span.set_attribute("high_confidence_count", confidence_results["high_count"])
                
                mlflow.log_metric("validator_low_confidence_count", confidence_results["low_count"])
                mlflow.log_metric("validator_high_confidence_count", confidence_results["high_count"])
                
        except Exception as exc:
            mlflow.log_metric("validator_confidence_check_error", 1)
            confidence_results = {"low_count": 0, "high_count": 0, "flags": []}
        
        # Consistency validation
        try:
            with mlflow.start_span(name="consistency_validation") as span:
                consistency_results = self._check_consistency(predictions)
                
                span.set_attribute("consistency_issues", len(consistency_results["issues"]))
                
                mlflow.log_metric("validator_consistency_issues", len(consistency_results["issues"]))
                
        except Exception as exc:
            mlflow.log_metric("validator_consistency_check_error", 1)
            consistency_results = {"issues": [], "flags": []}
        
        # Anomaly detection
        try:
            with mlflow.start_span(name="anomaly_detection") as span:
                anomaly_results = self._detect_anomalies(predictions)
                
                span.set_attribute("anomalies_detected", len(anomaly_results["anomalies"]))
                
                mlflow.log_metric("validator_anomaly_count", len(anomaly_results["anomalies"]))
                
        except Exception as exc:
            mlflow.log_metric("validator_anomaly_detection_error", 1)
            anomaly_results = {"anomalies": [], "flags": []}
        
        # Combine all validation results
        validated_results = []
        all_flags = []
        
        for pred in predictions:
            validated_pred = pred.copy()
            
            # Add confidence flag
            if pred["probability"] < self.confidence_threshold:
                validated_pred["confidence_flag"] = "LOW"
            else:
                validated_pred["confidence_flag"] = "OK"
            
            # Add validation status
            validated_pred["validated"] = True
            validated_pred["validation_issues"] = []
            
            # Check for specific issues
            element = pred["element"]
            if element in [a["element"] for a in anomaly_results["anomalies"]]:
                validated_pred["validation_issues"].append("ANOMALY_DETECTED")
            
            validated_results.append(validated_pred)
        
        # Collect all flags
        all_flags.extend(confidence_results.get("flags", []))
        all_flags.extend(consistency_results.get("flags", []))
        all_flags.extend(anomaly_results.get("flags", []))
        
        # Update state
        state["validated_results"] = validated_results
        state["validation_flags"] = all_flags
        
        # Calculate validation pass rate
        passed = len([p for p in validated_results if p["confidence_flag"] == "OK" and not p["validation_issues"]])
        pass_rate = passed / len(validated_results) if validated_results else 0
        
        # Log final metrics
        try:
            mlflow.log_param("validator_validation_rules_applied", 3)  # confidence, consistency, anomaly
            mlflow.log_metric("validator_validation_pass_rate", pass_rate)
            mlflow.log_metric("validator_total_flags", len(all_flags))
            mlflow.log_metric("validator_agent_success", 1)
        except Exception:
            pass
        
        return state
    
    def _check_confidence(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check confidence thresholds.
        
        Args:
            predictions: List of predictions to check
            
        Returns:
            Confidence check results
        """
        low_count = 0
        high_count = 0
        flags = []
        
        for pred in predictions:
            confidence = pred.get("probability", 0)
            
            if confidence < self.confidence_threshold:
                low_count += 1
                flags.append(f"LOW_CONFIDENCE: {pred['element']} ({confidence:.1%})")
            else:
                high_count += 1
        
        return {
            "low_count": low_count,
            "high_count": high_count,
            "flags": flags
        }
    
    def _check_consistency(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for consistency issues in predictions.
        
        Args:
            predictions: List of predictions to check
            
        Returns:
            Consistency check results
        """
        issues = []
        flags = []
        
        # Check for duplicate elements with conflicting data
        element_groups: Dict[str, List[Dict[str, Any]]] = {}
        for pred in predictions:
            element = pred["element"]
            if element not in element_groups:
                element_groups[element] = []
            element_groups[element].append(pred)
        
        for element, preds in element_groups.items():
            if len(preds) > 1:
                # Check if confidences are very different
                confidences = [p["probability"] for p in preds]
                if max(confidences) - min(confidences) > 0.3:
                    issues.append({
                        "element": element,
                        "issue": "CONFIDENCE_MISMATCH",
                        "details": f"Confidence range: {min(confidences):.1%} - {max(confidences):.1%}"
                    })
                    flags.append(f"CONSISTENCY: {element} has conflicting confidence scores")
        
        return {
            "issues": issues,
            "flags": flags
        }
    
    def _detect_anomalies(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect anomalous predictions.
        
        Args:
            predictions: List of predictions to check
            
        Returns:
            Anomaly detection results
        """
        anomalies = []
        flags = []
        
        if not predictions:
            return {"anomalies": anomalies, "flags": flags}
        
        # Calculate average confidence
        avg_confidence = sum(p["probability"] for p in predictions) / len(predictions)
        
        # Flag predictions that are significantly below average
        for pred in predictions:
            if pred["probability"] < avg_confidence * 0.5:  # Less than 50% of average
                anomalies.append({
                    "element": pred["element"],
                    "confidence": pred["probability"],
                    "reason": "Significantly below average confidence"
                })
                flags.append(f"ANOMALY: {pred['element']} confidence unusually low")
        
        return {
            "anomalies": anomalies,
            "flags": flags
        }
