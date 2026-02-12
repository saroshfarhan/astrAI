# ad-astrAI

A multi-agent AI framework for autonomous spectro-visual analysis of astronomical data. Detect elements and molecules from spectral (FITS/CSV) and image (PNG/JPG) data using a coordinated team of specialized AI agents.

## Quick Start

**For first-time users, follow these steps:**

1. **Install dependencies**: `uv sync`
2. **Configure API keys**: Copy `.env.example` to `.env` and add your `GOOGLE_API_KEY`
3. **Start MLflow** (Terminal 1): `uv run mlflow server --port 5000 --backend-store-uri sqlite:///mlruns.db`
4. **Start Spectral Service** (Terminal 2): `cd "Spectral Service" && uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload`
5. **Start Web UI** (Terminal 3): `uv run streamlit run app.py`
6. **Upload data** at http://localhost:8501 and explore results

**Note**: If models are missing, train them first using the instructions in the [Training Models](#training-models) section.

## Multi-Agent Architecture

This project uses **LangGraph** to coordinate 6 specialized agents:

1.  **Orchestrator Agent**: The brain. Analyzes input files and routes them to the correct model(s).
    *   *Routes*: `spectral`, `image`, or `both` (for multi-modal analysis).
2.  **Spectral Model Agent**: Specializes in 1D spectral data analysis (FITS/CSV).
3.  **Image Model Agent**: Specializes in 2D visual analysis (PNG/JPG).
4.  **Inference Agent**: The synthesizer. Consolidates predictions from all models and builds a dynamic **Knowledge Base**.
5.  **Validator Agent**: The quality control. Checks confidence thresholds and flags consistency issues.
6.  **Reporter Agent**: The communicator. Generates human-readable scientific reports.

### Architecture Diagrams

**Multi-Agent Architecture**
![Multi-Agent Architecture](Multi-Agent%20Architecture.png)

**Data-Model Flow Diagram**
![Data-Model Flow Diagram](Data-model%20flow%20diagram.png)

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
    SPECTRAL_SERVICE_URL=http://localhost:8001
    ```

### Environment Configuration

Create a `.env` file in the project root with the following variables:

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GOOGLE_API_KEY` | Gemini API key for LLM-powered agents | Yes | - |
| `MLFLOW_TRACKING_URI` | MLflow server URL for experiment tracking | No | `http://127.0.0.1:5000` |
| `SPECTRAL_SERVICE_URL` | Spectral Service backend URL | No | `http://localhost:8001` |

**How to get a Gemini API key:**
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

## Usage

The application consists of three main services that need to be started in separate terminal windows:

### 1. Start MLflow Server (Required for Tracing)
Start the local MLflow server to track agent execution traces and model training experiments.

```bash
uv run mlflow server --port 5000 --backend-store-uri sqlite:///mlruns.db
```
*View Trace UI at: http://127.0.0.1:5000*

### 2. Start Spectral Service (Backend API)
The Spectral Service provides the machine learning backend for spectral analysis. It runs a FastAPI server that hosts the trained UV and IR spectral models.

**Navigate to the Spectral Service directory and start the server:**
```bash
cd "Spectral Service"
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```
*API Documentation available at: http://localhost:8001/docs*

**Note**: The service will automatically load trained models from `Spectral Service/models/`:
- `uv_mlp.pt` - UV spectral model (28 species)
- `ir_mlp.pt` - IR spectral model (22 species)

If models are not found, you will need to train them first (see Training Models section below).

### 3. Run the Web UI
Launch the Streamlit interface for interactive analysis.

**In a new terminal, navigate back to the project root:**
```bash
cd ..
uv run streamlit run app.py
```
*Web UI will open at: http://localhost:8501*

### 4. How to Use the UI

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

## Training Models

The spectral analysis system uses two separate machine learning models trained on real planetary spectroscopic data with physics-based augmentation.

### Prerequisites for Training
- Real spectral data files must be present in `Spectral Service/data/real/`
  - UV spectra: `*_uv.pkl` files
  - IR spectra: `*_ir.pkl` files

### Train UV Model
```bash
cd "Spectral Service/training"
uv run python train_uv.py
```

**Model specifications:**
- **Input**: UV spectral data (3-channel preprocessed: normalized, 1st derivative, 2nd derivative)
- **Output**: 28 molecular species detection probabilities
- **Architecture**: Multi-Layer Perceptron with Optuna hyperparameter optimization
- **Training**: Physics-based augmentation (75 variations per planet) with planet-level validation split

### Train IR Model
```bash
cd "Spectral Service/training"
uv run python train_ir.py
```

**Model specifications:**
- **Input**: IR spectral data (3-channel preprocessed)
- **Output**: 22 molecular species detection probabilities
- **Architecture**: Multi-Layer Perceptron with Optuna hyperparameter optimization
- **Training**: Physics-based augmentation (75 variations per planet) with planet-level validation split

**Note**: Training uses MLflow for experiment tracking. Ensure MLflow server is running to view training metrics, hyperparameters, and model artifacts.

**Expected Training Time**:
- UV Model: ~30-60 minutes (depends on number of Optuna trials)
- IR Model: ~30-60 minutes

**Trained models will be saved to:**
- `Spectral Service/models/uv_mlp.pt` + `uv_config.json`
- `Spectral Service/models/ir_mlp.pt` + `ir_config.json`

## Project Structure

```
astraAI/
├── agent/                          # Multi-agent system
│   ├── agents/                     # Source code for all 6 agents
│   │   ├── orchestrator.py         # Routing agent
│   │   ├── spectral_agent.py       # Spectral model interface
│   │   ├── image_agent.py          # Image model interface
│   │   ├── inference_agent.py      # Prediction consolidation
│   │   ├── validator_agent.py      # Quality control
│   │   └── reporter_agent.py       # Report generation
│   ├── graph.py                    # LangGraph definitions and routing logic
│   └── state.py                    # Shared state schema
│
├── Spectral Service/               # Backend ML service
│   ├── app/                        # FastAPI application
│   │   ├── main.py                 # API entry point
│   │   └── routers/
│   │       └── analyze.py          # Spectral analysis endpoints
│   ├── training/                   # Model training pipeline
│   │   ├── train_uv.py             # UV model training
│   │   ├── train_ir.py             # IR model training
│   │   ├── augmentation.py         # Physics-based augmentation
│   │   ├── expanded_species.py     # Species definitions & planet labels
│   │   └── mlflow_utils.py         # MLflow integration
│   ├── data/
│   │   └── real/                   # Real planetary spectra (.pkl files)
│   └── models/                     # Trained models (generated after training)
│       ├── uv_mlp.pt               # UV model weights
│       ├── uv_config.json          # UV model configuration
│       ├── ir_mlp.pt               # IR model weights
│       └── ir_config.json          # IR model configuration
│
├── app.py                          # Streamlit frontend application
├── experiments/                    # Jupyter notebooks for prototyping
├── pyproject.toml                  # Project dependencies (uv package manager)
├── .env                            # Environment configuration (API keys)
└── README.md                       # This file
```

## Observability

This project uses **MLflow Tracing** for deep observability.
- **Spans**: Track every agent's execution time and inputs/outputs.
- **Metrics**: Monitor token usage, latency, and tool calls.
- **Artifacts**: Store generated reports and data snapshots.

## Troubleshooting

### Spectral Service Connection Error
**Issue**: Web UI shows "Failed to connect to Spectral Service"

**Solution**:
1. Ensure Spectral Service is running on port 8001
2. Check `.env` file has `SPECTRAL_SERVICE_URL=http://localhost:8001`
3. Verify models are trained and present in `Spectral Service/models/`

### Models Not Found
**Issue**: Spectral Service returns "Model not found" error

**Solution**:
1. Train the models using `train_uv.py` and `train_ir.py`
2. Verify `.pt` and `.json` files exist in `Spectral Service/models/`
3. Restart the Spectral Service after training

### Validation Loss = 0.0000
**Issue**: During training, validation loss shows exactly 0.0000

**Solution**: This indicates data leakage or overfitting:
- For **planet-level split**: Need 4+ planets minimum
- For **≤3 planets**: System uses sample-level split (expected behavior)
- Add more real planetary spectra to `Spectral Service/data/real/`

### High Validation Loss (>1.0)
**Issue**: Validation loss is very high during training

**Solution**:
- Check if you have enough training data (recommended: 4+ planets)
- Verify spectral data quality in `.pkl` files
- Increase `N_AUGMENT_PER_PLANET` parameter in training scripts

### LLM Routing Errors
**Issue**: Agent fails to route correctly or returns JSON parsing errors

**Solution**:
1. Verify `GOOGLE_API_KEY` is valid and active
2. Check Gemini API quota limits
3. Review MLflow traces to see exact LLM responses

### Port Already in Use
**Issue**: "Address already in use" error when starting services

**Solution**:
- **MLflow (5000)**: Change port in command: `mlflow server --port 5001`
- **Spectral Service (8001)**: Change port in command and update `.env`
- **Streamlit (8501)**: Streamlit will auto-increment to 8502

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns and structure
- All tests pass before submitting
- Documentation is updated for new features
- Commit messages are clear and descriptive

## License

This project is part of an academic research initiative.
