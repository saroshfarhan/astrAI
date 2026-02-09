"""Agent package exports and env bootstrap."""

import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file if present.
load_dotenv(find_dotenv(usecwd=True))

from .runner import run_pipeline

__all__ = ["run_pipeline"]
