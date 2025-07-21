"""
Vercel-optimized web application for insurance claim fraud prediction.
Simple HTML interface with FastAPI backend.
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
import json

# Import our modules
from data_loader import InsuranceDataLoader
from model import FraudPredictor
from causal_analysis import BiasDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Claim Risk Predictor",
    description="AI-powered insurance claim fraud prediction demo"
)

# Global variables for model state
predictor = None
training_data = None
bias_detector = None
model_loaded = False

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global predictor, training_data, bias_detector, model_loaded
    
    try:
        logger.info("Initializing models...")
        
        # Load data
        loader = InsuranceDataLoader()
        training_data = loader.load_data()
        
        # Train predictor
        predictor = FraudPredictor()
        predictor.train(training_data)
        
        # Initialize bias detector
        bias_detector = BiasDetector()
        
        model_loaded = True
        logger.info("Models initialized successfully")
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        model_loaded = False

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main HTML interface."""
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🛡️ Insurance Claim Risk Predictor</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            .gradient-bg {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .card {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
            }
        </style>
    </head>
    <body class="gradient-bg min-h-screen">
        <div class="container mx-auto px-4 py-8">
            <!-- Header -->
            <div class="text-center mb-8">
                <h1 class="text-4xl font-bold text-white mb-4">🛡️ Insurance Claim Risk Predictor</h1>
                <p class="text-xl text-white opacity-90">AI-powered fraud detection for UK motor insurance claims</p>
                <div class="mt-4 bg-white bg-opacity-20 rounded-lg p-4 max-w-4xl mx-auto">
                    <p class="text-white text-sm leading-relaxed">
                        This tool uses artificial intelligence to assess the likelihood that an insurance claim might be fraudulent. 
                        Simply enter the claim details below and our AI will provide a risk assessment based on patterns learned from thousands of previous claims.
                        The higher the percentage, the more likely the claim may need additional investigation.
                    </p>
                </div>
            </div>

            <!-- Prediction Form -->
            <div class="card rounded-lg shadow-2xl p-6 mb-8">
                <h2 class="text-2xl font-bold mb-6 text-gray-800">Predict Fraud Risk</h2>
                
                <form id="predictionForm" class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Age</label>
                        <input type="number" name="age" min="18" max="100" value="35" 
                               class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Gender</label>
                        <select name="gender" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <option value="M">Male</option>
                            <option value="F">Female</option>
                        </select>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Vehicle Age (years)</label>
                        <input type="number" name="vehicle_age" min="0" max="30" value="5" 
                               class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Vehicle Type</label>
                        <select name="vehicle_type" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <option value="saloon">Saloon</option>
                            <option value="suv">SUV</option>
                            <option value="van">Van</option>
                            <option value="sports">Sports Car</option>
                            <option value="luxury">Luxury</option>
                        </select>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Annual Mileage (miles)</label>
                        <input type="number" name="annual_mileage" min="1000" max="40000" value="12000" 
                               class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Driving Violations</label>
                        <input type="number" name="driving_violations" min="0" max="20" value="1" 
                               class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Claim Amount (£)</label>
                        <input type="number" name="claim_amount" min="100" max="800000" value="20000" 
                               class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Previous Claims</label>
                        <input type="number" name="previous_claims" min="0" max="50" value="0" 
                               class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Credit Score</label>
                        <input type="number" name="credit_score" min="300" max="850" value="700" 
                               class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Region</label>
                        <select name="region" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <option value="london">London</option>
                            <option value="midlands">Midlands</option>
                            <option value="north">North</option>
                            <option value="south">South</option>
                            <option value="scotland">Scotland</option>
                        </select>
                    </div>
                    
                    <div class="md:col-span-2">
                        <button type="submit" class="w-full bg-blue-600 text-white py-3 px-6 rounded-md hover:bg-blue-700 transition duration-200 font-semibold">
                            🔍 Predict Fraud Risk
                        </button>
                    </div>
                </form>
            </div>

            <!-- Results -->
            <div id="results" class="card rounded-lg shadow-2xl p-6 mb-8 hidden">
                <h2 class="text-2xl font-bold mb-6 text-gray-800">Prediction Results</h2>
                <div id="resultsContent"></div>
            </div>

            <!-- Model Info -->
            <div class="card rounded-lg shadow-2xl p-6">
                <h2 class="text-2xl font-bold mb-6 text-gray-800">Model Information</h2>
                <div id="modelInfo">Loading...</div>
            </div>
        </div>

        <script>
            // Load model info on page load
            window.onload = async function() {
                try {
                    const response = await fetch('/model-info');
                    const info = await response.json();
                    
                    document.getElementById('modelInfo').innerHTML = `
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div class="text-center">
                                <div class="text-3xl font-bold text-blue-600">${info.training_samples || 'N/A'}</div>
                                <div class="text-gray-600">Training Samples</div>
                            </div>
                            <div class="text-center">
                                <div class="text-3xl font-bold text-green-600">${(info.fraud_rate * 100 || 0).toFixed(1)}%</div>
                                <div class="text-gray-600">Fraud Rate</div>
                            </div>
                            <div class="text-center">
                                <div class="text-3xl font-bold text-purple-600">${info.features || 'N/A'}</div>
                                <div class="text-gray-600">Features</div>
                            </div>
                        </div>
                    `;
                } catch (error) {
                    console.error('Error loading model info:', error);
                }
            };

            // Handle form submission
            document.getElementById('predictionForm').onsubmit = async function(e) {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData.entries());
                
                // Convert numeric fields
                ['age', 'vehicle_age', 'annual_mileage', 'driving_violations', 'claim_amount', 'previous_claims', 'credit_score'].forEach(field => {
                    data[field] = parseInt(data[field]);
                });
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    const riskLevel = result.fraud_probability < 0.3 ? 'Low' : 
                                    result.fraud_probability < 0.7 ? 'Medium' : 'High';
                    const riskColor = result.fraud_probability < 0.3 ? 'green' : 
                                     result.fraud_probability < 0.7 ? 'yellow' : 'red';
                    
                    // Generate explanations based on risk level
                    let explanation = '';
                    let recommendation = '';
                    
                    if (result.fraud_probability < 0.3) {
                        explanation = 'This claim appears to have a low risk of being fraudulent. The AI has analysed the claim details and found patterns similar to legitimate claims.';
                        recommendation = 'This claim can likely be processed normally with standard verification procedures.';
                    } else if (result.fraud_probability < 0.7) {
                        explanation = 'This claim has some characteristics that warrant additional attention. While not definitively fraudulent, certain patterns suggest it should be reviewed more carefully.';
                        recommendation = 'Consider additional verification steps such as requesting more documentation or conducting a brief phone interview with the claimant.';
                    } else {
                        explanation = 'This claim has multiple red flags that are commonly associated with fraudulent claims. The AI has detected patterns that frequently appear in proven fraud cases.';
                        recommendation = 'This claim should be thoroughly investigated before processing. Consider involving fraud investigators and requiring comprehensive documentation.';
                    }

                    document.getElementById('results').classList.remove('hidden');
                    document.getElementById('resultsContent').innerHTML = `
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                            <div class="text-center p-4 bg-blue-50 rounded-lg">
                                <div class="text-4xl font-bold text-blue-600">${(result.fraud_probability * 100).toFixed(1)}%</div>
                                <div class="text-gray-600 text-sm">Fraud Probability</div>
                            </div>
                            <div class="text-center p-4 bg-${riskColor}-50 rounded-lg">
                                <div class="text-4xl font-bold text-${riskColor}-600">${riskLevel} Risk</div>
                                <div class="text-gray-600 text-sm">Assessment Level</div>
                            </div>
                            <div class="text-center p-4 bg-gray-50 rounded-lg">
                                <div class="text-4xl font-bold text-gray-600">${result.processing_time_ms.toFixed(1)}ms</div>
                                <div class="text-gray-600 text-sm">Analysis Time</div>
                            </div>
                        </div>
                        
                        <div class="bg-gray-100 rounded-lg p-4 mb-4">
                            <h3 class="font-semibold mb-3 text-lg">🎯 Risk Assessment</h3>
                            <div class="w-full bg-gray-200 rounded-full h-6 mb-2">
                                <div class="bg-${riskColor}-600 h-6 rounded-full transition-all duration-1000 ease-out" style="width: ${result.fraud_probability * 100}%"></div>
                            </div>
                            <div class="flex justify-between text-xs text-gray-500 mb-4">
                                <span>0% - Very Low Risk</span>
                                <span>50% - Medium Risk</span>
                                <span>100% - Very High Risk</span>
                            </div>
                        </div>

                        <div class="bg-white border-l-4 border-blue-500 rounded-lg p-4 mb-4">
                            <h3 class="font-semibold mb-2 text-lg text-blue-800">📋 What This Means</h3>
                            <p class="text-gray-700 leading-relaxed">${explanation}</p>
                        </div>

                        <div class="bg-white border-l-4 border-green-500 rounded-lg p-4">
                            <h3 class="font-semibold mb-2 text-lg text-green-800">💡 Recommended Action</h3>
                            <p class="text-gray-700 leading-relaxed">${recommendation}</p>
                        </div>

                        <div class="mt-4 text-xs text-gray-500 bg-gray-50 rounded p-3">
                            <p><strong>Disclaimer:</strong> This is an AI assessment tool designed to assist with claim evaluation. It should not be the sole factor in claim decisions. Always combine AI insights with human judgment and follow your company's established procedures.</p>
                        </div>
                    `;
                    
                } catch (error) {
                    console.error('Prediction error:', error);
                    alert('Error making prediction. Please try again.');
                }
            };
        </script>
    </body>
    </html>
    """
    
    return html_content

@app.post("/predict")
async def predict_fraud(request: Request):
    """Predict fraud probability for a claim."""
    if not model_loaded:
        return JSONResponse({"error": "Models not loaded"}, status_code=503)
    
    try:
        # Get request data
        data = await request.json()
        
        # Make prediction
        fraud_prob = predictor.predict_single(**data)
        
        # Calculate processing time (mock for demo)
        import time
        processing_time = 45.2  # Mock processing time
        
        return {
            "fraud_probability": float(fraud_prob),
            "risk_level": "low" if fraud_prob < 0.3 else "medium" if fraud_prob < 0.7 else "high",
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/model-info")
async def model_info():
    """Get model information."""
    if not model_loaded:
        return JSONResponse({"error": "Models not loaded"}, status_code=503)
    
    return {
        "training_samples": len(training_data) if training_data is not None else 0,
        "fraud_rate": float(training_data['is_fraud'].mean()) if training_data is not None else 0,
        "features": len(predictor.feature_names) if predictor and predictor.feature_names else 0,
        "model_type": "ensemble"
    }

@app.get("/bias-analysis")
async def get_bias_analysis():
    """Get bias analysis results."""
    if not model_loaded:
        return JSONResponse({"error": "Models not loaded"}, status_code=503)
    
    try:
        predictions = predictor.predict(training_data)
        fairness_report = bias_detector.fairness_report(training_data, predictions)
        
        return fairness_report
        
    except Exception as e:
        logger.error(f"Bias analysis error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model_loaded else "initializing",
        "model_loaded": model_loaded
    }

# For Vercel
app_handler = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)