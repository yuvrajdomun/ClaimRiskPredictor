"""
Insurance Fraud Detector Package

A comprehensive AI-powered system for detecting fraudulent insurance claims
using machine learning, causal analysis, and bias detection.

Author: Insurance Analytics Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Insurance Analytics Team"
__email__ = "contact@insurancefraud.ai"

from .models.predictor import FraudPredictor
from .data.loader import InsuranceDataLoader
from .training.trainer import ModelTrainer

__all__ = [
    "FraudPredictor",
    "InsuranceDataLoader", 
    "ModelTrainer"
]