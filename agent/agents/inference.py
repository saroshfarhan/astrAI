"""Inference Agent - Consolidates predictions and builds Knowledge Base.

This agent merges predictions from multiple model agents (spectral and/or image),
builds a session-scoped Knowledge Base dynamically from the results,
and enriches predictions with cross-referenced metadata.
"""

from typing import List, Dict, Any
import mlflow

from ..base import BaseAgent
from ..state import PipelineState, Prediction


class InferenceAgent(BaseAgent):
    """Consolidates multi-modal predictions and builds dynamic KB.
    
    Responsibilities:
    - Merge spectral and image predictions
    - Build/update Knowledge Base dynamically from model outputs
    - Cross-reference and validate multi-modal data
    - Format results as JSON
    - Store KB entries for current session retrieval
    """
    
    def __init__(self):
        """Initialize inference agent."""
        super().__init__("InferenceAgent")
    
    def process(self, state: PipelineState) -> PipelineState:
        """Consolidate predictions and build KB.
        
        Args:
            state: Pipeline state with spectral_predictions and/or image_predictions
            
        Returns:
            Updated state with inference_results and knowledge_base
        """
        # Get predictions from both models
        spectral_preds = state.get("spectral_predictions", [])
        image_preds = state.get("image_predictions", [])
        
        # Log input counts
        try:
            mlflow.log_param("inference_num_spectral_preds", len(spectral_preds))
            mlflow.log_param("inference_num_image_preds", len(image_preds))
            mlflow.log_param("inference_total_input_preds", len(spectral_preds) + len(image_preds))
        except Exception:
            pass
        
        # Merge predictions
        try:
            with mlflow.start_span(name="merge_predictions") as span:
                merged = self._merge_predictions(spectral_preds, image_preds)
                
                span.set_attribute("merged_count", len(merged))
                span.set_attribute("unique_elements", len(set(p["element"] for p in merged)))
                
                mlflow.log_metric("inference_merged_count", len(merged))
                mlflow.log_metric("inference_unique_elements", len(set(p["element"] for p in merged)))
                
        except Exception as exc:
            mlflow.log_metric("inference_merge_error", 1)
            state.setdefault("errors", []).append(f"Prediction merging failed: {exc}")
            state["inference_results"] = {}
            return state
        
        # Build Knowledge Base
        try:
            with mlflow.start_span(name="build_knowledge_base") as span:
                kb = self._build_knowledge_base(merged, spectral_preds, image_preds)
                
                span.set_attribute("kb_entries", len(kb))
                span.set_attribute("kb_elements", ", ".join(list(kb.keys())[:10]))
                
                mlflow.log_param("inference_kb_entries_created", len(kb))
                mlflow.log_param("inference_kb_elements", ", ".join(list(kb.keys())[:10]))
                mlflow.log_metric("inference_kb_size", len(kb))
                
        except Exception as exc:
            mlflow.log_metric("inference_kb_build_error", 1)
            state.setdefault("errors", []).append(f"KB building failed: {exc}")
            kb = {}
        
        # Cross-reference predictions
        try:
            with mlflow.start_span(name="cross_reference") as span:
                cross_ref_results = self._cross_reference_predictions(merged, kb)
                
                span.set_attribute("cross_ref_matches", cross_ref_results["matches"])
                span.set_attribute("multi_modal_elements", cross_ref_results["multi_modal_count"])
                
                mlflow.log_metric("inference_cross_ref_matches", cross_ref_results["matches"])
                mlflow.log_metric("inference_multi_modal_elements", cross_ref_results["multi_modal_count"])
                
        except Exception as exc:
            mlflow.log_metric("inference_cross_ref_error", 1)
            cross_ref_results = {"matches": 0, "multi_modal_count": 0}
        
        # Enrich predictions with KB data
        try:
            with mlflow.start_span(name="enrich_predictions") as span:
                enriched = self._enrich_predictions(merged, kb)
                
                span.set_attribute("enriched_count", len(enriched))
                
                # Calculate enrichment success rate
                enrichment_rate = len([p for p in enriched if p.get("enriched", False)]) / len(enriched) if enriched else 0
                mlflow.log_metric("inference_enrichment_success_rate", enrichment_rate)
                
        except Exception as exc:
            mlflow.log_metric("inference_enrich_error", 1)
            enriched = merged  # Fall back to non-enriched
        
        # Format final results
        inference_results = {
            "predictions": enriched,
            "summary": {
                "total_elements": len(enriched),
                "unique_elements": len(set(p["element"] for p in enriched)),
                "spectral_sources": len(spectral_preds),
                "image_sources": len(image_preds),
                "multi_modal_elements": cross_ref_results["multi_modal_count"],
                "avg_confidence": sum(p["probability"] for p in enriched) / len(enriched) if enriched else 0
            }
        }
        
        # Update state
        state["inference_results"] = inference_results
        state["knowledge_base"] = kb
        
        # Log final metrics
        try:
            mlflow.log_param("inference_consolidation_strategy", "merge_and_enrich")
            mlflow.log_metric("inference_final_prediction_count", len(enriched))
            mlflow.log_metric("inference_avg_confidence", inference_results["summary"]["avg_confidence"])
            mlflow.log_metric("inference_agent_success", 1)
        except Exception:
            pass
        
        return state
    
    def _merge_predictions(
        self, 
        spectral_preds: List[Prediction], 
        image_preds: List[Prediction]
    ) -> List[Dict[str, Any]]:
        """Merge predictions from spectral and image models.
        
        If the same element appears in both, we keep both but mark them as multi-modal.
        
        Args:
            spectral_preds: Predictions from spectral model
            image_preds: Predictions from image model
            
        Returns:
            Merged list of predictions with source tags
        """
        merged = []
        
        # Add spectral predictions with source tag
        for pred in spectral_preds:
            merged.append({
                **pred,
                "source": "spectral",
                "multi_modal": False
            })
        
        # Add image predictions with source tag
        for pred in image_preds:
            merged.append({
                **pred,
                "source": "image",
                "multi_modal": False
            })
        
        # Mark elements that appear in both as multi-modal
        spectral_elements = {p["element"] for p in spectral_preds}
        image_elements = {p["element"] for p in image_preds}
        common_elements = spectral_elements & image_elements
        
        for pred in merged:
            if pred["element"] in common_elements:
                pred["multi_modal"] = True
        
        return merged
    
    def _build_knowledge_base(
        self,
        merged_predictions: List[Dict[str, Any]],
        spectral_preds: List[Prediction],
        image_preds: List[Prediction]
    ) -> Dict[str, Dict[str, Any]]:
        """Build session-scoped Knowledge Base from predictions.
        
        KB structure: {element: {name, color, sources, confidence, ...}}
        
        Args:
            merged_predictions: All merged predictions
            spectral_preds: Original spectral predictions
            image_preds: Original image predictions
            
        Returns:
            Knowledge Base dictionary
        """
        kb = {}
        
        # Group predictions by element
        element_groups: Dict[str, List[Dict[str, Any]]] = {}
        for pred in merged_predictions:
            element = pred["element"]
            if element not in element_groups:
                element_groups[element] = []
            element_groups[element].append(pred)
        
        # Build KB entry for each element
        for element, preds in element_groups.items():
            # Get highest confidence prediction for this element
            best_pred = max(preds, key=lambda p: p["probability"])
            
            # Determine sources
            sources = list(set(p["source"] for p in preds))
            
            # Calculate average confidence if multiple predictions
            avg_confidence = sum(p["probability"] for p in preds) / len(preds)
            
            kb[element] = {
                "name": self._get_element_name(element),
                "symbol": element,
                "color": best_pred.get("color", "#CCCCCC"),
                "sources": sources,
                "confidence": avg_confidence,
                "max_confidence": best_pred["probability"],
                "detection_count": len(preds),
                "multi_modal": len(sources) > 1,
                "rationale": best_pred.get("rationale", "")
            }
        
        return kb
    
    def _cross_reference_predictions(
        self,
        merged_predictions: List[Dict[str, Any]],
        kb: Dict[str, Dict[str, Any]]
    ) -> Dict[str, int]:
        """Cross-reference predictions using KB.
        
        Args:
            merged_predictions: Merged predictions
            kb: Knowledge Base
            
        Returns:
            Cross-reference statistics
        """
        matches = 0
        multi_modal_count = 0
        
        for pred in merged_predictions:
            element = pred["element"]
            if element in kb:
                matches += 1
                if kb[element]["multi_modal"]:
                    multi_modal_count += 1
        
        return {
            "matches": matches,
            "multi_modal_count": multi_modal_count
        }
    
    def _enrich_predictions(
        self,
        predictions: List[Dict[str, Any]],
        kb: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich predictions with KB metadata.
        
        Args:
            predictions: Predictions to enrich
            kb: Knowledge Base
            
        Returns:
            Enriched predictions
        """
        enriched = []
        
        for pred in predictions:
            element = pred["element"]
            
            # Copy prediction
            enriched_pred = pred.copy()
            
            # Add KB metadata if available
            if element in kb:
                enriched_pred["kb_name"] = kb[element]["name"]
                enriched_pred["kb_multi_modal"] = kb[element]["multi_modal"]
                enriched_pred["kb_avg_confidence"] = kb[element]["confidence"]
                enriched_pred["enriched"] = True
            else:
                enriched_pred["enriched"] = False
            
            enriched.append(enriched_pred)
        
        return enriched
    
    def _get_element_name(self, symbol: str) -> str:
        """Get full element name from symbol.
        
        Args:
            symbol: Element symbol (e.g., "H", "Fe")
            
        Returns:
            Full element name
        """
        # Common astronomical elements
        element_names = {
            "H": "Hydrogen",
            "He": "Helium",
            "O": "Oxygen",
            "N": "Nitrogen",
            "C": "Carbon",
            "Fe": "Iron",
            "Si": "Silicon",
            "Mg": "Magnesium",
            "Al": "Aluminum",
            "Ca": "Calcium",
            "Na": "Sodium",
            "S": "Sulfur"
        }
        
        return element_names.get(symbol, symbol)
