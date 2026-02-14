"""Base agent framework for multi-agent system.

Provides common utilities, MLflow tracing, and error handling for all agents.
"""

from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import mlflow

from .state import PipelineState


class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class BaseAgent(ABC):
    """Abstract base class for all agents in the pipeline.
    
    All agents must implement the `process()` method which takes a PipelineState
    and returns an updated PipelineState.
    
    The `__call__()` method wraps `process()` with automatic MLflow tracing
    and error handling.
    """
    
    def __init__(self, name: str):
        """Initialize agent with a name for logging and tracing.
        
        Args:
            name: Human-readable agent name (e.g., "OrchestratorAgent")
        """
        self.name = name
    
    @abstractmethod
    def process(self, state: PipelineState) -> PipelineState:
        """Process the state and return updated state.
        
        This is the main method that each agent must implement.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated pipeline state
            
        Raises:
            AgentError: If agent processing fails
        """
        pass
    
    @mlflow.trace(name="agent_call", span_type="AGENT")
    def __call__(self, state: PipelineState) -> PipelineState:
        """Execute agent with automatic tracing and error handling.
        
        This wrapper:
        1. Traces the agent execution in MLflow
        2. Logs the agent name
        3. Catches and logs errors to state
        4. Returns updated state
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated pipeline state with any errors logged
        """
        try:
            # Try to set MLflow tags if available
            try:
                mlflow.set_tag("agent_name", self.name)
            except Exception:
                pass  # MLflow not configured, skip tagging
            
            # Log agent activation
            state.setdefault("active_agents", []).append(self.name)
            
            # Process state
            updated_state = self.process(state)
            
            # Log success metric if MLflow is available
            try:
                mlflow.log_metric(f"{self.name}_success", 1)
            except Exception:
                pass  # MLflow not configured, skip metrics
            
            return updated_state
            
        except Exception as exc:
            # Log error to state
            error_msg = f"{self.name} error: {str(exc)}"
            state.setdefault("errors", []).append(error_msg)
            
            # Try to log to MLflow if available
            try:
                mlflow.log_metric(f"{self.name}_error", 1)
                mlflow.log_param(f"{self.name}_error_message", str(exc))
            except Exception:
                pass  # MLflow not configured, skip logging
            
            # Re-raise as AgentError for upstream handling
            raise AgentError(error_msg) from exc


def safe_agent_call(agent_func: Callable) -> Callable:
    """Decorator for agent methods that need error handling without full BaseAgent.
    
    Use this for standalone agent functions that don't inherit from BaseAgent.
    
    Example:
        @safe_agent_call
        def my_agent_function(state: PipelineState) -> PipelineState:
            # ... processing logic
            return state
    """
    @wraps(agent_func)
    def wrapper(state: PipelineState) -> PipelineState:
        try:
            return agent_func(state)
        except Exception as exc:
            error_msg = f"{agent_func.__name__} error: {str(exc)}"
            state.setdefault("errors", []).append(error_msg)
            return state
    return wrapper


def log_agent_metrics(
    agent_name: str,
    metrics: Dict[str, float],
    params: Optional[Dict[str, Any]] = None
) -> None:
    """Helper to log agent-specific metrics and parameters to MLflow.
    
    Args:
        agent_name: Name of the agent
        metrics: Dictionary of metrics to log (e.g., {"confidence": 0.85})
        params: Optional dictionary of parameters to log
    """
    for key, value in metrics.items():
        mlflow.log_metric(f"{agent_name}_{key}", value)
    
    if params:
        for key, value in params.items():
            mlflow.log_param(f"{agent_name}_{key}", value)


def validate_state_fields(state: PipelineState, required_fields: List[str]) -> None:
    """Validate that required fields are present in state.

    Args:
        state: Pipeline state to validate
        required_fields: List of required field names

    Raises:
        AgentError: If any required field is missing
    """
    missing = [field for field in required_fields if field not in state]
    if missing:
        raise AgentError(f"Missing required state fields: {missing}")


def log_llm_usage(response: Any, agent_name: str, operation: str = "llm_call") -> None:
    """Extract and log LLM token usage to MLflow.

    This function extracts token usage metadata from Gemini/LangChain responses
    and logs them as MLflow metrics for cost tracking and analysis.

    Args:
        response: LLM response object from langchain (must have usage_metadata)
        agent_name: Name of the agent (e.g., "orchestrator", "reporter")
        operation: Operation name for span attributes (default: "llm_call")

    Logs:
        - {agent_name}_input_tokens: Number of input/prompt tokens
        - {agent_name}_output_tokens: Number of output/generated tokens
        - {agent_name}_total_tokens: Total tokens used

    Example:
        >>> response = llm.invoke("Hello")
        >>> log_llm_usage(response, "orchestrator", "routing_decision")
        # Logs: orchestrator_input_tokens, orchestrator_output_tokens, etc.
    """
    try:
        # Extract usage metadata from Gemini response
        # Token usage is in response.usage_metadata (not response.response_metadata)
        usage = getattr(response, 'usage_metadata', {})

        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)

        # Log metrics with agent prefix
        mlflow.log_metric(f"{agent_name}_input_tokens", input_tokens)
        mlflow.log_metric(f"{agent_name}_output_tokens", output_tokens)
        mlflow.log_metric(f"{agent_name}_total_tokens", total_tokens)

        # Also log to span attributes for detailed tracing
        try:
            span = mlflow.get_current_active_span()
            if span:
                span.set_attribute("input_tokens", input_tokens)
                span.set_attribute("output_tokens", output_tokens)
                span.set_attribute("total_tokens", total_tokens)
                span.set_attribute("operation", operation)
        except Exception:
            pass  # Span operations are optional

    except Exception as e:
        # Don't fail on logging errors - just log the error
        try:
            mlflow.log_param(f"{agent_name}_token_logging_error", str(e)[:200])
        except Exception:
            pass  # Completely silent failure if MLflow not available
