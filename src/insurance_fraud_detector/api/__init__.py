"""
API Package

Contains REST API endpoints and web application interfaces.
"""

from .endpoints import create_api_app
from .schemas import PredictionRequest, PredictionResponse

__all__ = ["create_api_app", "PredictionRequest", "PredictionResponse"]