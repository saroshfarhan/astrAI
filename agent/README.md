# Agent skeleton

This directory contains a minimal LangGraph-based multi-agent pipeline with mocked model execution and MLflow logging.

Key files
- `state.py` — shared PipelineState schema and defaults.
- `graph.py` — LangGraph definition, stubbed model runner, MLflow hooks.
- `prompts/` — LLM prompts kept separate for clarity and prompt testing.
- `runner.py` — helper for running the pipeline.

Replacing the mock model
- Implement real inference in `run_model` inside `graph.py` (load your pickle, return list of predictions with probabilities and rationales).
- Keep outputs JSON-serializable; avoid heavy tensors in state.

Prompt testing
- Prompts live in `prompts/*.txt`; use MLflow to log prompt versions and outcomes by updating `prompt_version` in state and logging in `log_mlflow`.

Observability
- Each run starts an MLflow run; state snapshots are logged at start and end under `events/` artifacts.
- Default tracking URI is `sqlite:///mlflow.db` to avoid the soon-to-be-deprecated file store. Override via `MLFLOW_TRACKING_URI` if using a remote server.
- When launched from Streamlit (`app.py`), a parent `ui-session` MLflow run logs upload metadata and links to the child pipeline run ID for visibility.
