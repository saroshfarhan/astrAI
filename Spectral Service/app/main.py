import sys
from pathlib import Path
from fastapi import FastAPI

# Add the app directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent))

from routers.analyze import router as analyze_router

app = FastAPI(title="Spectral Service", version="0.1.0")

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Spectral Service", "version": "0.1.0"}

app.include_router(analyze_router, prefix="", tags=["spectral"])
