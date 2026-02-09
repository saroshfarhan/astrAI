from fastapi import FastAPI
from .routers.analyze import router as analyze_router

app = FastAPI(title="Spectral Service", version="0.1.0")

app.include_router(analyze_router, prefix="", tags=["spectral"])
