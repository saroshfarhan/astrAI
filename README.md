# ad-astrAI

A multi-agent AI framework for autonomous spectro-visual analysis of astronomical data. Detect elements and molecules from spectral (FITS/CSV) and image (PNG/JPG) data using a coordinated team of specialized AI agents.

## Multi-Agent Architecture

This project uses **LangGraph** to coordinate 6 specialized agents:

1.  **Orchestrator Agent**: The brain. Analyzes input files and routes them to the correct model(s).
    *   *Routes*: `spectral`, `image`, or `both` (for multi-modal analysis).
2.  **Spectral Model Agent**: Specializes in 1D spectral data analysis (FITS/CSV).
3.  **Image Model Agent**: Specializes in 2D visual analysis (PNG/JPG).
4.  **Inference Agent**: The synthesizer. Consolidates predictions from all models and builds a dynamic **Knowledge Base**.
5.  **Validator Agent**: The quality control. Checks confidence thresholds and flags consistency issues.
6.  **Reporter Agent**: The communicator. Generates human-readable scientific reports.

## Getting Started

### Prerequisites

- Python 3.11+
- `uv` (Fast Python package manager)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1.  Clone the repository:
    ```bash
    git clone <repo-url>
    cd ad-astrAI
    ```

2.  Install dependencies:
    ```bash
    uv sync
    ```

3.  Configure API keys in `.env` (see `.env.example`):
    ```ini
    GOOGLE_API_KEY=your_gemini_key
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    ```

## Usage

### 1. Start MLflow Server (Required for Tracing)
Start the local MLflow server to track agent execution traces.

```bash
uv run mlflow server --port 5000 --backend-store-uri sqlite:///mlruns.db
```
*View Trace UI at: http://127.0.0.1:5000*

### 2. Run the Web UI
Launch the Streamlit interface for interactive analysis.

```bash
uv run streamlit run app.py
```

### 3. How to Use the UI

1.  **Upload Data**:
    *   `*.fits` / `*.csv` -> Triggers **Spectral Analysis**
    *   `*.png` / `*.jpg` -> Triggers **Image Analysis**
    *   **Multi-Modal Trick**: Include `complex` in the filename (e.g., `complex_nebula.fits`) to trigger **Both** spectral and image analysis.

2.  **View Results**:
    *   **Mission Report**: Read the AI-generated summary.
    *   **Agent Trace**: See the execution path (e.g., Orchestrator -> Spectral -> Inference...).
    *   **Predictions**: View the consolidated element detection table.

3.  **Chat with Data**:
    *   Use the chat interface at the bottom to ask questions about the findings (e.g., *"What elements were found with high confidence?"*).

## Project Structure

- `agent/`
    - `agents/`: Source code for all 6 agents.
    - `graph.py`: LangGraph definitions and routing logic.
    - `state.py`: Shared state schema.
- `app.py`: Streamlit frontend application.
- `experiments/`: Jupyter notebooks for prototyping.

## Observability

This project uses **MLflow Tracing** for deep observability.
- **Spans**: Track every agent's execution time and inputs/outputs.
- **Metrics**: Monitor token usage, latency, and tool calls.
- **Artifacts**: Store generated reports and data snapshots.
