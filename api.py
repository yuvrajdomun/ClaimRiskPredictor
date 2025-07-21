"""
FastAPI web service for insurance claim fraud prediction.
Provides REST API endpoints for risk assessment and analysis.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import asyncio
import time
from pathlib import Path
import logging

# Import our modules
from data_loader import InsuranceDataLoader
from model import FraudPredictor
from causal_analysis import CausalAnalyzer, BiasDetector
from generative_augmentation import VariationalAutoencoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Claim Risk Predictor",
    description="AI-powered insurance claim fraud prediction with causal analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ClaimRequest(BaseModel):
    """Insurance claim data for prediction."""
    age: int = Field(..., ge=18, le=100, description="Age of policyholder")
    gender: str = Field(..., regex="^(M|F)$", description="Gender (M/F)")
    vehicle_age: int = Field(..., ge=0, le=30, description="Age of vehicle in years")
    vehicle_type: str = Field(..., regex="^(sedan|suv|truck|sports|luxury)$", description="Type of vehicle")
    annual_mileage: int = Field(..., ge=0, le=100000, description="Annual mileage")
    driving_violations: int = Field(..., ge=0, le=20, description="Number of driving violations")
    claim_amount: int = Field(..., ge=0, le=1000000, description="Claim amount in dollars")
    previous_claims: int = Field(..., ge=0, le=50, description="Number of previous claims")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    region: str = Field(..., regex="^(urban|suburban|rural)$", description="Region type")

class PredictionResponse(BaseModel):
    """Fraud prediction response."""
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    risk_level: str = Field(..., description="Risk level (low/medium/high)")
    confidence: float = Field(..., description="Model confidence")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class WhatIfRequest(BaseModel):
    """What-if analysis request."""
    base_claim: ClaimRequest
    interventions: Dict[str, Any] = Field(..., description="Variables to change")

class WhatIfResponse(BaseModel):
    """What-if analysis response."""
    results: Dict[str, Any] = Field(..., description="Analysis results")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class BiasAnalysisResponse(BaseModel):
    """Bias analysis response."""
    overall_fairness_score: float = Field(..., description="Overall fairness score")
    bias_detected: bool = Field(..., description="Whether bias was detected")
    disparate_impact: Dict[str, Any] = Field(..., description="Disparate impact analysis")
    recommendations: List[str] = Field(..., description="Recommendations")

# Global variables for model state
predictor: Optional[FraudPredictor] = None
causal_analyzer: Optional[CausalAnalyzer] = None
bias_detector: Optional[BiasDetector] = None
training_data: Optional[pd.DataFrame] = None
model_loaded = False

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global predictor, causal_analyzer, bias_detector, training_data, model_loaded
    
    try:
        logger.info("Starting model initialization...")
        
        # Load data
        loader = InsuranceDataLoader()
        training_data = loader.load_data()
        logger.info(f"Loaded {len(training_data)} training samples")
        
        # Train predictor
        predictor = FraudPredictor()
        results = predictor.train(training_data)
        logger.info(f"Model trained with AUC: {results['test_auc']:.4f}")
        
        # Initialize analyzers
        causal_analyzer = CausalAnalyzer()
        bias_detector = BiasDetector()
        
        model_loaded = True
        logger.info("Model initialization completed")
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        model_loaded = False

def check_model_loaded():
    """Check if models are loaded."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Insurance Claim Risk Predictor API",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "endpoints": [
            "/predict",
            "/what-if",
            "/bias-analysis",
            "/health",
            "/model-info"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": time.time()
    }

@app.get("/model-info")
async def model_info():
    """Get model information."""
    check_model_loaded()
    
    return {
        "model_type": "ensemble",
        "features": len(predictor.feature_names) if predictor.feature_names else 0,
        "training_samples": len(training_data) if training_data is not None else 0,
        "fraud_rate": float(training_data['is_fraud'].mean()) if training_data is not None else 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(claim: ClaimRequest):
    """Predict fraud probability for a single claim."""
    check_model_loaded()
    
    start_time = time.time()
    
    try:
        # Convert to dict for prediction
        claim_dict = claim.dict()
        
        # Make prediction
        fraud_prob = predictor.predict_single(**claim_dict)
        
        # Determine risk level
        if fraud_prob < 0.3:
            risk_level = "low"
        elif fraud_prob < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        # Calculate confidence (simplified)
        confidence = 1.0 - abs(fraud_prob - 0.5) * 2
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            fraud_probability=float(fraud_prob),
            risk_level=risk_level,
            confidence=float(confidence),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_fraud_batch(claims: List[ClaimRequest]):
    """Predict fraud probability for multiple claims."""
    check_model_loaded()
    
    if len(claims) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 claims per batch")
    
    start_time = time.time()
    
    try:
        # Convert to DataFrame
        claims_data = pd.DataFrame([claim.dict() for claim in claims])
        claims_data['is_fraud'] = 0  # Placeholder
        
        # Make predictions
        predictions = predictor.predict(claims_data)
        
        # Format results
        results = []
        for i, prob in enumerate(predictions):
            risk_level = "low" if prob < 0.3 else "medium" if prob < 0.7 else "high"
            confidence = 1.0 - abs(prob - 0.5) * 2
            
            results.append({
                "claim_index": i,
                "fraud_probability": float(prob),
                "risk_level": risk_level,
                "confidence": float(confidence)
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "results": results,
            "processing_time_ms": processing_time,
            "batch_size": len(claims)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/what-if", response_model=WhatIfResponse)
async def what_if_analysis(request: WhatIfRequest):
    """Perform what-if analysis on claim variables."""
    check_model_loaded()
    
    start_time = time.time()
    
    try:
        # Convert base claim to dict
        base_claim = request.base_claim.dict()
        
        # Perform what-if analysis
        results = causal_analyzer.what_if_analysis(
            training_data, 
            base_claim, 
            request.interventions
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return WhatIfResponse(
            results=results,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"What-if analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/bias-analysis", response_model=BiasAnalysisResponse)
async def bias_analysis():
    """Perform bias analysis on the model."""
    check_model_loaded()
    
    try:
        # Get predictions for training data
        predictions = predictor.predict(training_data)
        
        # Perform bias analysis
        fairness_report = bias_detector.fairness_report(training_data, predictions)
        
        return BiasAnalysisResponse(
            overall_fairness_score=fairness_report['overall_fairness_score'],
            bias_detected=fairness_report['bias_detected'],
            disparate_impact=fairness_report['disparate_impact'],
            recommendations=fairness_report['recommendations']
        )
        
    except Exception as e:
        logger.error(f"Bias analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feature-importance")
async def feature_importance():
    """Get feature importance from the model."""
    check_model_loaded()
    
    try:
        importance = predictor.get_feature_importance()
        
        # Sort by importance
        sorted_importance = sorted(
            importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            "feature_importance": dict(sorted_importance),
            "top_features": sorted_importance[:10]
        }
        
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Trigger model retraining (background task)."""
    check_model_loaded()
    
    def retrain():
        global predictor, model_loaded
        try:
            logger.info("Starting model retraining...")
            model_loaded = False
            
            # Retrain with fresh data (could include synthetic data)
            loader = InsuranceDataLoader()
            new_data = loader.load_data()
            
            predictor = FraudPredictor()
            results = predictor.train(new_data)
            
            model_loaded = True
            logger.info(f"Model retrained with AUC: {results['test_auc']:.4f}")
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            model_loaded = False
    
    background_tasks.add_task(retrain)
    
    return {"message": "Model retraining started in background"}

@app.get("/synthetic-data/{n_samples}")
async def generate_synthetic_data(n_samples: int):
    """Generate synthetic data for testing."""
    if n_samples > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 samples")
    
    try:
        # Use simple generator
        vae = VariationalAutoencoder()
        vae.train_simple_generator(training_data)
        synthetic_data = vae.generate_simple_data(n_samples)
        
        # Convert to list of dicts for JSON response
        return {
            "synthetic_claims": synthetic_data.to_dict('records'),
            "count": len(synthetic_data),
            "fraud_rate": float(synthetic_data['is_fraud'].mean())
        }
        
    except Exception as e:
        logger.error(f"Synthetic data generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False
    )