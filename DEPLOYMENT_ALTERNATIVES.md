# 🚀 Deployment Alternatives for Insurance Fraud Detector

This guide provides multiple deployment options to overcome Vercel's 250MB serverless function limit.

## 🎯 Quick Comparison

| Platform | Pros | Cons | Best For |
|----------|------|------|----------|
| **Railway** | No size limits, easy setup, free tier | Newer platform | Quick deployment |
| **Render** | ML-optimized, free tier | Build times | ML applications |
| **Google Cloud Run** | Auto-scaling, pay-per-use | Requires GCP account | Production apps |
| **Hugging Face Spaces** | Free, ML-focused | Public only (free tier) | ML demos |
| **Heroku** | Simple, established | Expensive, no free tier | Traditional apps |

## 1. 🚂 Railway (Recommended - Easiest)

### Setup Steps:
```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login to Railway
railway login

# 3. Initialize project
railway init

# 4. Deploy
railway up
```

### Configuration:
Create `railway.toml`:
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn vercel_app:app --host 0.0.0.0 --port $PORT"

[variables]
PORT = "8000"
PYTHON_VERSION = "3.9"
```

**Live in ~2 minutes!** ⚡

---

## 2. 🎨 Render (Great for ML)

### Setup Steps:
1. Connect your GitHub repo at [render.com](https://render.com)
2. Choose "Web Service"
3. Use these settings:

```yaml
# render.yaml (optional)
services:
  - type: web
    name: fraud-detector
    env: python
    region: oregon
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn vercel_app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.16
    scaling:
      minInstances: 1
      maxInstances: 10
```

**Manual Setup:**
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn vercel_app:app --host 0.0.0.0 --port $PORT`
- **Python Version:** 3.9.16

---

## 3. ☁️ Google Cloud Run (Production-Ready)

### Prerequisites:
```bash
# Install Google Cloud CLI
# Visit: https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Deployment:
```bash
# 1. Build and deploy directly from source
gcloud run deploy fraud-detector \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --port 8000

# 2. Set environment variables (if needed)
gcloud run services update fraud-detector \
    --set-env-vars PYTHON_ENV=production \
    --region us-central1
```

### Dockerfile (optional):
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8080

# Start application
CMD ["uvicorn", "vercel_app:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## 4. 🤗 Hugging Face Spaces (Perfect for ML Demos)

### Setup Steps:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space
3. Choose "Streamlit" or "Gradio"
4. Upload your files

### For Streamlit:
Create `app.py`:
```python
import streamlit as st
import sys
import os
sys.path.append('.')

# Import your streamlit app
from streamlit_app import main

if __name__ == "__main__":
    main()
```

### For Gradio (API interface):
```python
import gradio as gr
from vercel_app import predict_fraud

def gradio_predict(age, gender, vehicle_age, vehicle_type, annual_mileage, 
                   driving_violations, claim_amount, previous_claims, 
                   credit_score, region):
    
    claim_data = {
        "age": age,
        "gender": gender,
        "vehicle_age": vehicle_age,
        "vehicle_type": vehicle_type,
        "annual_mileage": annual_mileage,
        "driving_violations": driving_violations,
        "claim_amount": claim_amount,
        "previous_claims": previous_claims,
        "credit_score": credit_score,
        "region": region
    }
    
    result = predict_fraud(claim_data)
    return f"Fraud Probability: {result['fraud_probability']:.1%}\\nRisk Level: {result['risk_level']}"

# Create Gradio interface
interface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Slider(18, 100, value=35, label="Age"),
        gr.Dropdown(["M", "F"], value="M", label="Gender"),
        gr.Slider(0, 30, value=5, label="Vehicle Age"),
        gr.Dropdown(["sedan", "suv", "truck", "sports", "luxury"], value="sedan", label="Vehicle Type"),
        gr.Slider(1000, 100000, value=15000, label="Annual Mileage"),
        gr.Slider(0, 20, value=1, label="Driving Violations"),
        gr.Slider(100, 1000000, value=25000, label="Claim Amount (£)"),
        gr.Slider(0, 50, value=1, label="Previous Claims"),
        gr.Slider(300, 850, value=650, label="Credit Score"),
        gr.Dropdown(["urban", "suburban", "rural"], value="suburban", label="Region")
    ],
    outputs="text",
    title="🛡️ Insurance Fraud Detector",
    description="AI-powered fraud detection for UK insurance claims"
)

interface.launch()
```

---

## 5. 🟣 Heroku Alternative

### Setup:
```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create your-fraud-detector

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main
```

### Procfile:
```
web: uvicorn vercel_app:app --host 0.0.0.0 --port $PORT
```

---

## 🔧 Vercel Size Optimization (If You Must Use Vercel)

### Option 1: Minimal Requirements
Use `requirements-vercel.txt` and `vercel_minimal.py` I created above:

```bash
# Deploy minimal version
cp requirements-vercel.txt requirements.txt
cp vercel-minimal.json vercel.json
git add . && git commit -m "Minimal Vercel version"
git push
```

### Option 2: Separate API and Frontend
1. Deploy API on Railway/Render/Cloud Run
2. Deploy Streamlit frontend on Vercel/Streamlit Cloud
3. Configure frontend to call external API

---

## 📊 Performance Comparison

| Platform | Cold Start | Concurrent Users | Monthly Cost (Free Tier) |
|----------|------------|------------------|---------------------------|
| Railway | ~2s | 500+ | $0 (500 hours) |
| Render | ~10s | 100+ | $0 (750 hours) |
| Cloud Run | ~1s | 1000+ | $0 (2M requests) |
| HF Spaces | ~5s | 50+ | $0 (public apps) |
| Vercel | ~0.5s | Limited by size | $0 (100GB-hrs) |

## 🎯 My Recommendation

**For Your Use Case:** Use **Railway** 🚂

**Why Railway?**
1. **No size limits** - Deploy full ML stack
2. **GitHub integration** - Auto-deploy on push  
3. **Free tier** - 500 hours/month
4. **Fast setup** - 2 minutes from code to live URL
5. **Scales automatically** - Handles traffic spikes

### Quick Railway Deployment:
```bash
npm install -g @railway/cli
railway login
cd ClaimRiskPredictor
railway init
railway up
```

**Your app will be live at:** `https://yourapp.railway.app`

Would you like me to help you deploy to Railway or any other platform?