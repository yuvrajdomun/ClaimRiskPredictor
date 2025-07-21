# 🚀 Streamlit Deployment Guide

## Option 1: Streamlit Community Cloud (Recommended - FREE!)

### Prerequisites
- GitHub account
- Repository pushed to GitHub

### Steps:
1. **Push code to GitHub** (if not already done)
   ```bash
   git add .
   git commit -m "Prepare for Streamlit deployment"
   git push origin main
   ```

2. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io/
   - Sign in with GitHub

3. **Deploy your app**
   - Click "New app"
   - Select your repository: `ClaimRiskPredictor`
   - Main file path: `streamlit_app.py`
   - Click "Deploy!"

4. **App will be available at:**
   - URL: `https://[your-username]-claimriskpredictor-streamlit-app-[hash].streamlit.app`

### Configuration Files Created:
- ✅ `requirements-streamlit.txt` - Streamlined dependencies
- ✅ `.streamlit/config.toml` - UI/server configuration

---

## Option 2: Railway (Alternative)

1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and deploy:**
   ```bash
   railway login
   railway init
   railway up
   ```

3. **Set environment:**
   - Add `requirements-streamlit.txt` as requirements file
   - Set start command: `streamlit run streamlit_app.py --server.port $PORT`

---

## Option 3: Render (Alternative)

1. **Go to:** https://render.com/
2. **Connect GitHub repository**
3. **Settings:**
   - Build Command: `pip install -r requirements-streamlit.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

---

## 🔧 Local Testing Before Deployment

Test locally to ensure everything works:
```bash
# Use the streamlit requirements
pip install -r requirements-streamlit.txt

# Run the app
streamlit run streamlit_app.py
```

---

## 📝 Notes

- **Main app file:** `streamlit_app.py`
- **Dependencies:** Listed in `requirements-streamlit.txt`
- **Data file:** `data/insurance_claims.csv` (included in repository)
- **No environment variables needed**
- **No external APIs required**

Your app should deploy successfully with the configuration files provided!