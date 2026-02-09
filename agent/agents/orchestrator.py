"""Orchestrator Agent - Routes to appropriate specialist agents.

The Orchestrator is the entry point of the pipeline. It analyzes the input file
and decides which specialist agents should process it.
"""

import json
from pathlib import Path
from typing import Dict

from langchain_google_genai import ChatGoogleGenerativeAI

from ..base import BaseAgent, validate_state_fields
from ..state import PipelineState


class OrchestratorAgent(BaseAgent):
    """Intelligent routing agent that determines processing path.
    
    Responsibilities:
    - Analyze input file type and metadata
    - Decide routing: "spectral", "image", or "both"
    - Select appropriate models
    - Handle error cases
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI = None):
        """Initialize orchestrator with LLM for intelligent routing.
        
        Args:
            llm: Language model for routing decisions (defaults to Gemini)
        """
        super().__init__("OrchestratorAgent")
        self.llm = llm or ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0
        )
        self.prompt = self._load_prompt()
    
    def _load_prompt(self) -> str:
        """Load orchestrator prompt from file."""
        prompt_path = Path(__file__).parent.parent / "prompts" / "orchestrator_v2_prompt.txt"
        if prompt_path.exists():
            return prompt_path.read_text().strip()
        
        # Fallback prompt if file doesn't exist yet
        return """You are the Orchestrator. Analyze the input file and decide the processing route.

Guidelines:
- If file extension is .fits or .csv → "spectral" (spectral analysis)
- If file extension is .png, .jpg, .jpeg → "image" (image analysis)  
- If the file could benefit from BOTH analyses → "both" (parallel processing)
- If unsupported or error → "error"

Consider:
- FITS files can contain both spectral data AND images
- For complex astronomical data, both modalities may provide complementary insights
- Default to single modality unless there's clear benefit to both

Respond with JSON only:
{
  "route": "spectral" | "image" | "both" | "error",
  "model_spectral": "mock-spectral-v1" (if spectral route),
  "model_image": "mock-image-v1" (if image route),
  "reasoning": "brief explanation of routing decision"
}"""
    
    def process(self, state: PipelineState) -> PipelineState:
        """Analyze input and determine routing.
        
        Args:
            state: Pipeline state with input_path
            
        Returns:
            Updated state with route and model selections
        """
        import mlflow
        
        # Validate required fields
        validate_state_fields(state, ["input_path"])
        
        input_path = state["input_path"]
        file_ext = Path(input_path).suffix.lower()
        
        # Log input parameters to MLflow
        try:
            mlflow.log_param("orchestrator_input_file", Path(input_path).name)
            mlflow.log_param("orchestrator_file_extension", file_ext)
            mlflow.log_param("orchestrator_modality", state.get("modality", "unknown"))
        except Exception:
            pass  # MLflow not configured
        
        # Prepare context for LLM
        context = {
            "file_path": input_path,
            "file_extension": file_ext,
            "modality": state.get("modality", "unknown")
        }
        
        try:
            # Create MLflow span for LLM routing decision
            with mlflow.start_span(name="llm_routing_decision") as span:
                span.set_attribute("file_extension", file_ext)
                span.set_attribute("modality", state.get("modality", "unknown"))
                
                # Get routing decision from LLM
                response = self.llm.invoke([
                    ("system", self.prompt),
                    ("human", json.dumps(context, default=str))
                ])
                
                # Parse LLM response
                decision = json.loads(response.content)
                
                # Log LLM decision to span
                span.set_attribute("llm_route_decision", decision.get("route", "unknown"))
                span.set_attribute("llm_reasoning", decision.get("reasoning", ""))
                
                # Log metrics
                try:
                    mlflow.log_metric("orchestrator_llm_success", 1)
                except Exception:
                    pass
            
        except Exception as exc:
            # Log LLM failure
            try:
                mlflow.log_metric("orchestrator_llm_failure", 1)
                mlflow.log_param("orchestrator_llm_error", str(exc)[:100])
            except Exception:
                pass
            
            # Fallback to rule-based routing if LLM fails
            state.setdefault("errors", []).append(f"LLM routing failed: {exc}, using fallback")
            
            with mlflow.start_span(name="fallback_routing") as span:
                decision = self._fallback_routing(file_ext, Path(input_path).name)
                span.set_attribute("fallback_route", decision.get("route", "unknown"))
                
                try:
                    mlflow.log_metric("orchestrator_fallback_used", 1)
                except Exception:
                    pass
        
        # Update state with routing decision
        route = decision.get("route", "error")
        state["route"] = route
        
        # Set model names
        if route in ["spectral", "both"]:
            state["model_name"] = decision.get("model_spectral", "mock-spectral-v1")
        elif route == "image":
            state["model_name"] = decision.get("model_image", "mock-image-v1")
        
        # Store reasoning in metadata
        state.setdefault("metadata", {})["orchestrator_reasoning"] = decision.get(
            "reasoning", "No reasoning provided"
        )
        
        # Log routing decision
        state["metadata"]["orchestrator_route"] = route
        
        # Log final routing metrics
        try:
            mlflow.log_param("orchestrator_final_route", route)
            mlflow.log_param("orchestrator_model_name", state.get("model_name", "unknown"))
            mlflow.log_param("orchestrator_reasoning", decision.get("reasoning", "")[:200])
            
            # Log route type as metric for aggregation
            mlflow.log_metric(f"orchestrator_route_{route}", 1)
            
            # Log whether parallel processing is needed
            if route == "both":
                mlflow.log_metric("orchestrator_parallel_processing", 1)
            else:
                mlflow.log_metric("orchestrator_parallel_processing", 0)
                
        except Exception:
            pass  # MLflow not configured
        
        return state
    
    def _fallback_routing(self, file_ext: str, filename: str = "") -> Dict[str, str]:
        """Rule-based fallback routing when LLM fails.
        
        Args:
            file_ext: File extension (e.g., ".fits", ".png")
            filename: Original filename (optional, for heuristic routing)
            
        Returns:
            Routing decision dictionary
        """
        # Heuristic: if filename indicates multi-modal/complex data, route to both
        if "complex" in filename.lower() or "multi" in filename.lower():
             return {
                "route": "both",
                "model_spectral": "mock-spectral-v1",
                "model_image": "mock-image-v1",
                "reasoning": "Fallback: Filename indicates multi-modal data"
            }

        if file_ext in [".fits", ".csv"]:
            return {
                "route": "spectral",
                "model_spectral": "mock-spectral-v1",
                "reasoning": "Fallback: FITS/CSV files routed to spectral analysis"
            }
        elif file_ext in [".png", ".jpg", ".jpeg"]:
            return {
                "route": "image",
                "model_image": "mock-image-v1",
                "reasoning": "Fallback: Image files routed to image analysis"
            }
        else:
            return {
                "route": "error",
                "reasoning": f"Fallback: Unsupported file type {file_ext}"
            }
