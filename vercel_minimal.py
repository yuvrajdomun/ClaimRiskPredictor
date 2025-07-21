"""
Minimal Vercel deployment version of the fraud detector.
This version uses only lightweight dependencies to stay under size limits.
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from pathlib import Path

# Simplified model class for Vercel
class MinimalFraudPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = [
            'age', 'vehicle_age', 'annual_mileage', 'driving_violations',
            'claim_amount', 'previous_claims', 'credit_score'
        ]
        self.categorical_features = ['gender', 'vehicle_type', 'region']
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize with pre-trained simple models."""
        # Simple Random Forest
        self.models['rf'] = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42
        )
        
        # Logistic Regression
        self.models['lr'] = LogisticRegression(random_state=42)
        
        # Initialize scalers and encoders
        self.scaler = StandardScaler()
        self.gender_encoder = LabelEncoder()
        self.vehicle_encoder = LabelEncoder()
        self.region_encoder = LabelEncoder()
        
        # Fit with dummy data (would be replaced with actual training)
        self._fit_dummy_data()
    
    def _fit_dummy_data(self):
        """Fit with dummy data for demonstration."""
        # Create dummy training data
        np.random.seed(42)
        n_samples = 1000
        
        X_numeric = np.random.rand(n_samples, len(self.feature_names))
        X_numeric[:, 0] = X_numeric[:, 0] * 60 + 20  # age 20-80
        X_numeric[:, 1] = X_numeric[:, 1] * 20  # vehicle_age 0-20
        X_numeric[:, 2] = X_numeric[:, 2] * 50000 + 5000  # mileage
        X_numeric[:, 3] = (X_numeric[:, 3] * 5).astype(int)  # violations
        X_numeric[:, 4] = X_numeric[:, 4] * 100000 + 1000  # claim_amount
        X_numeric[:, 5] = (X_numeric[:, 5] * 10).astype(int)  # previous_claims
        X_numeric[:, 6] = X_numeric[:, 6] * 550 + 300  # credit_score
        
        # Categorical data
        gender = np.random.choice(['M', 'F'], n_samples)
        vehicle_type = np.random.choice(['sedan', 'suv', 'truck'], n_samples)
        region = np.random.choice(['urban', 'suburban', 'rural'], n_samples)
        
        # Create target (simplified fraud logic)
        y = ((X_numeric[:, 3] > 2) | (X_numeric[:, 4] > 50000) | (X_numeric[:, 6] < 600)).astype(int)
        
        # Fit scalers and encoders
        self.scaler.fit(X_numeric)
        self.gender_encoder.fit(gender)
        self.vehicle_encoder.fit(vehicle_type)
        self.region_encoder.fit(region)
        
        # Prepare full feature matrix
        X_scaled = self.scaler.transform(X_numeric)
        gender_encoded = self.gender_encoder.transform(gender).reshape(-1, 1)
        vehicle_encoded = self.vehicle_encoder.transform(vehicle_type).reshape(-1, 1)
        region_encoded = self.region_encoder.transform(region).reshape(-1, 1)
        
        X_full = np.hstack([X_scaled, gender_encoded, vehicle_encoded, region_encoded])
        
        # Train models
        self.models['rf'].fit(X_full, y)
        self.models['lr'].fit(X_full, y)
    
    def predict(self, claim_data):
        """Make prediction on single claim."""
        try:
            # Extract and scale numerical features
            numeric_data = np.array([[
                claim_data.get('age', 35),
                claim_data.get('vehicle_age', 5),
                claim_data.get('annual_mileage', 15000),
                claim_data.get('driving_violations', 1),
                claim_data.get('claim_amount', 25000),
                claim_data.get('previous_claims', 1),
                claim_data.get('credit_score', 650)
            ]])
            
            numeric_scaled = self.scaler.transform(numeric_data)
            
            # Encode categorical features
            gender = claim_data.get('gender', 'M')
            vehicle_type = claim_data.get('vehicle_type', 'sedan')
            region = claim_data.get('region', 'suburban')
            
            try:
                gender_encoded = self.gender_encoder.transform([gender])[0]
            except ValueError:
                gender_encoded = 0  # Default for unknown values
            
            try:
                vehicle_encoded = self.vehicle_encoder.transform([vehicle_type])[0]
            except ValueError:
                vehicle_encoded = 0
            
            try:
                region_encoded = self.region_encoder.transform([region])[0]
            except ValueError:
                region_encoded = 0
            
            # Combine features
            X_full = np.hstack([
                numeric_scaled,
                [[gender_encoded, vehicle_encoded, region_encoded]]
            ])
            
            # Get predictions from both models
            rf_pred = self.models['rf'].predict_proba(X_full)[0, 1]
            lr_pred = self.models['lr'].predict_proba(X_full)[0, 1]
            
            # Ensemble prediction (simple average)
            fraud_probability = (rf_pred + lr_pred) / 2
            
            # Determine risk level
            if fraud_probability < 0.3:
                risk_level = "Low"
            elif fraud_probability < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            return {
                "fraud_probability": float(fraud_probability),
                "risk_level": risk_level,
                "model_predictions": {
                    "random_forest": float(rf_pred),
                    "logistic_regression": float(lr_pred)
                },
                "processing_time_ms": 10.0  # Placeholder
            }
            
        except Exception as e:
            return {
                "fraud_probability": 0.5,
                "risk_level": "Unknown",
                "error": str(e)
            }

# Initialize global predictor
predictor = MinimalFraudPredictor()

# Simplified FastAPI app
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(title="Insurance Fraud Detector (Minimal)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClaimRequest(BaseModel):
    age: int = 35
    gender: str = "M"
    vehicle_age: int = 5
    vehicle_type: str = "sedan"
    annual_mileage: int = 15000
    driving_violations: int = 1
    claim_amount: float = 25000
    previous_claims: int = 1
    credit_score: int = 650
    region: str = "suburban"

@app.get("/")
async def root():
    return {
        "name": "Insurance Fraud Detector API (Minimal)",
        "version": "1.0.0",
        "status": "active",
        "description": "Lightweight version optimized for Vercel deployment"
    }

@app.post("/predict")
async def predict_fraud(request: ClaimRequest):
    """Predict fraud probability for a claim."""
    claim_dict = request.dict()
    result = predictor.predict(claim_dict)
    return result

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# For Vercel
def handler(request):
    return app(request)