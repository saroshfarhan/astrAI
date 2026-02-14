"""LangGraph multi-agent pipeline with conditional routing.

This graph orchestrates the complete multi-agent workflow:
1. Orchestrator → routes to appropriate model agents
2. Spectral/Image Models → run in parallel or individually based on route
3. Inference → consolidates predictions and builds KB
4. Validator → validates results
5. Reporter → generates natural language report
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from mlflow import MlflowClient
import mlflow
from mlflow.exceptions import MlflowException

from .state import DEFAULT_STATE, PipelineState
from .agents.orchestrator import OrchestratorAgent
from .agents.spectral_model import SpectralModelAgent
from .agents.image_model import ImageModelAgent
from .agents.inference import InferenceAgent
from .agents.validator import ValidatorAgent
from .agents.reporter import ReporterAgent

PROMPT_DIR = Path(__file__).parent / "prompts"


def _normalize_prompt(raw: Any) -> str:
    """Return a plain string prompt from various prompt objects."""
    if isinstance(raw, str):
        return raw
    template = getattr(raw, "template", None)
    if template:
        return template
    definition = getattr(raw, "definition", None)
    if definition:
        if hasattr(definition, "template") and definition.template:
            return definition.template
        if isinstance(definition, dict):
            for key in ("template", "text", "prompt"):
                if key in definition:
                    return definition[key]
    if isinstance(raw, dict):
        for key in ("template", "text", "prompt"):
            if key in raw:
                return raw[key]
    return str(raw)


def load_prompt(name: str) -> str:
    """Load prompt text from local file or MLflow prompt registry."""
    path = PROMPT_DIR / name
    if path.exists():
        return path.read_text().strip()
    try:
        prompt_version = mlflow.genai.load_prompt(name)
        return _normalize_prompt(prompt_version)
    except Exception:
        return _normalize_prompt(name)


def configure_mlflow() -> None:
    """Ensure MLflow has a usable tracking URI and experiment.

    Defaults to http://127.0.0.1:5000 if MLflow server is running.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "ad-astrAI")

    # Set experiment (create if not exists)
    try:
        mlflow.set_experiment(experiment)
    except Exception:
        print(f"Warning: Could not set MLflow experiment '{experiment}'")

    # Enable LangChain autologging for GenAI monitoring
    # This makes token usage visible in the GenAI apps & agents overview
    try:
        mlflow.langchain.autolog(
            log_traces=True,  # Enable trace logging
            silent=True       # Suppress verbose logging
        )
    except Exception as e:
        print(f"Warning: Could not enable LangChain autologging: {e}")




@mlflow.trace
def log_mlflow(event: str, state: PipelineState, run_id: str) -> None:
    """Log pipeline events to MLflow."""
    mlflow_client = MlflowClient()
    mlflow_client.log_text(
        run_id, 
        json.dumps({"event": event, "state": state}, default=str), 
        f"events/{event}.json"
    )


# ============================================================================
# Agent Node Wrappers
# ============================================================================

# Initialize agents (singleton instances)
_orchestrator = OrchestratorAgent()
_spectral_model = SpectralModelAgent()
_image_model = ImageModelAgent()
_inference = InferenceAgent()
_validator = ValidatorAgent()
_reporter = ReporterAgent()


def orchestrator_node(state: PipelineState) -> PipelineState:
    """Orchestrator agent node."""
    return _orchestrator(state)


def spectral_model_node(state: PipelineState) -> PipelineState:
    """Spectral model agent node."""
    return _spectral_model(state)


def image_model_node(state: PipelineState) -> PipelineState:
    """Image model agent node."""
    return _image_model(state)


def inference_node(state: PipelineState) -> PipelineState:
    """Inference agent node."""
    return _inference(state)


def validator_node(state: PipelineState) -> PipelineState:
    """Validator agent node."""
    return _validator(state)


def reporter_node(state: PipelineState) -> PipelineState:
    """Reporter agent node."""
    return _reporter(state)


# ============================================================================
# Conditional Routing Functions
# ============================================================================

def route_after_orchestrator(state: PipelineState) -> Literal["spectral", "image", "both", "error"]:
    """Route based on orchestrator decision.
    
    Returns:
        - "spectral": Route to spectral model only
        - "image": Route to image model only
        - "both": Route to both models (parallel)
        - "error": Route to end (unsupported file)
    """
    route = state.get("route", "error")
    
    # Log routing decision
    try:
        mlflow.log_param("graph_route_decision", route)
    except Exception:
        pass
    
    return route


def route_after_models(state: PipelineState) -> str:
    """Route from model agents to inference.
    
    All model routes converge to inference agent.
    """
    return "inference"


# ============================================================================
# Graph Construction
# ============================================================================

@mlflow.trace
def build_graph() -> Callable[[PipelineState], PipelineState]:
    """Build the multi-agent LangGraph pipeline.
    
    Graph structure:
    
        START
          ↓
      Orchestrator
          ↓
        [Route Decision]
          ↓
        ┌─────┬─────┬─────┐
        │     │     │     │
     spectral image both error
        │     │     │     │
        │     │   ┌─┴─┐   │
        │     │   │   │   │
     Spectral Image │   END
        │     │     │
        └──┬──┴─────┘
           ↓
       Inference
           ↓
       Validator
           ↓
        Reporter
           ↓
          END
    
    Returns:
        Compiled graph runner function
    """
    graph = StateGraph(PipelineState)
    
    # Add all agent nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("spectral", spectral_model_node)
    graph.add_node("image", image_model_node)
    graph.add_node("inference", inference_node)
    graph.add_node("validator", validator_node)
    graph.add_node("reporter", reporter_node)
    
    # Set entry point
    graph.set_entry_point("orchestrator")
    
    # Conditional routing after orchestrator
    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "spectral": "spectral",
            "image": "image",
            "both": "spectral",  # Start with spectral, then image
            "error": END
        }
    )
    
    # For "both" route, run spectral then image
    graph.add_conditional_edges(
        "spectral",
        lambda state: "image" if state.get("route") == "both" else "inference",
        {
            "image": "image",
            "inference": "inference"
        }
    )
    
    # Image always goes to inference
    graph.add_edge("image", "inference")
    
    # Linear flow after inference
    graph.add_edge("inference", "validator")
    graph.add_edge("validator", "reporter")
    graph.add_edge("reporter", END)
    
    # Compile with memory
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    @mlflow.trace(name="pipeline_execution", span_type="CHAIN")
    def runner(initial: PipelineState) -> PipelineState:
        """Run the compiled graph with MLflow tracking.
        
        Args:
            initial: Initial pipeline state
            
        Returns:
            Final pipeline state after all agents
        """
        configure_mlflow()
        nested = mlflow.active_run() is not None
        
        with mlflow.start_run(run_name="multi-agent-pipeline", nested=nested) as run:
            # Merge with default state
            initial = {**DEFAULT_STATE, **initial}
            initial.setdefault("log_refs", {})["mlflow"] = run.info.run_id
            
            # Log start event
            log_mlflow("start", initial, run.info.run_id)
            
            # Log initial parameters
            try:
                mlflow.log_param("pipeline_input_path", initial.get("input_path", "unknown"))
                mlflow.log_param("pipeline_modality", initial.get("modality", "unknown"))
            except Exception:
                pass
            
            # Run graph
            config = {"configurable": {"thread_id": initial.get("input_path", "run")}}
            final_state = compiled.invoke(initial, config=config)
            
            # Log end event
            log_mlflow("end", final_state, run.info.run_id)

            # Log final metrics
            try:
                mlflow.log_metric("pipeline_total_agents", len(final_state.get("active_agents", [])))
                mlflow.log_metric("pipeline_total_errors", len(final_state.get("errors", [])))
                mlflow.log_metric("pipeline_success", 1 if not final_state.get("errors") else 0)
            except Exception:
                pass

            # Aggregate token usage across all agents
            try:
                # Get all metrics logged in this run
                mlflow_client = MlflowClient()
                run_data = mlflow_client.get_run(run.info.run_id)
                metrics = run_data.data.metrics

                # Sum tokens by category
                total_input_tokens = sum(v for k, v in metrics.items() if k.endswith('_input_tokens'))
                total_output_tokens = sum(v for k, v in metrics.items() if k.endswith('_output_tokens'))
                total_tokens = sum(v for k, v in metrics.items() if k.endswith('_total_tokens'))

                # Log aggregated metrics
                if total_tokens > 0:  # Only log if we actually used LLMs
                    mlflow.log_metric("pipeline_total_input_tokens", total_input_tokens)
                    mlflow.log_metric("pipeline_total_output_tokens", total_output_tokens)
                    mlflow.log_metric("pipeline_total_tokens", total_tokens)

                    # CRITICAL: Set token usage on root span for MLflow Overview tab
                    # MLflow's Overview expects tokenUsage attribute on the root trace span
                    try:
                        root_span = mlflow.get_current_active_span()
                        if root_span:
                            # Set token usage in the format MLflow Overview expects
                            root_span.set_attribute("tokenUsage", {
                                "input_tokens": int(total_input_tokens),
                                "output_tokens": int(total_output_tokens),
                                "total_tokens": int(total_tokens)
                            })
                            # Also set individual fields for compatibility
                            root_span.set_attribute("input_tokens", int(total_input_tokens))
                            root_span.set_attribute("output_tokens", int(total_output_tokens))
                            root_span.set_attribute("total_tokens", int(total_tokens))
                    except Exception:
                        pass  # Span operations are optional

            except Exception:
                pass  # Don't fail pipeline on metrics aggregation

            return final_state
    
    return runner


# Build and export the graph runner
RUN_GRAPH = build_graph()


if __name__ == "__main__":  # manual smoke test
    result = RUN_GRAPH({"input_path": "example.fits", "modality": "spectral"})
    print(json.dumps(result, indent=2, default=str))
