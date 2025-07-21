# 🚀 Deployment Guide

This guide covers various deployment options for the UK Insurance Claim Risk Predictor.

## 📋 Quick Start

The application is designed to be deployment-ready on multiple platforms with minimal configuration.

### 🌐 Vercel Deployment (Recommended)

Vercel provides the easiest deployment option with automatic scaling and global CDN.

#### Option 1: One-Click Deploy

1. Click the deploy button in the README
2. Fork the repository to your GitHub account
3. Vercel will automatically build and deploy your application
4. Your app will be live in minutes!

#### Option 2: Manual Deploy

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy from your project directory**
   ```bash
   cd ClaimRiskPredictor
   vercel
   ```

4. **Follow the prompts:**
   - Link to existing project? **N**
   - What's your project's name? **claim-risk-predictor**
   - In which directory is your code located? **.**
   - Want to override settings? **N**

### 🐳 Docker Deployment

Run the application in a containerized environment.

1. **Build the Docker image**
   ```bash
   docker build -t claim-risk-predictor .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 claim-risk-predictor
   ```

### ☁️ Cloud Platform Deployment

#### Heroku

1. **Create a Heroku app**
   ```bash
   heroku create your-app-name
   ```

2. **Deploy**
   ```bash
   git push heroku main
   ```

#### Railway

1. **Connect your GitHub repository** at [railway.app](https://railway.app)
2. **Select your repository** and branch
3. **Railway will automatically deploy** your application

#### Google Cloud Run

1. **Build and push to Container Registry**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/claim-risk-predictor
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy --image gcr.io/PROJECT-ID/claim-risk-predictor --platform managed
   ```

## 🔧 Configuration

### Environment Variables

The application supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 8000 |
| `HOST` | Server host | 0.0.0.0 |
| `INSURANCE_DATA_PATH` | Path to data file | data/insurance_claims.csv |
| `MODEL_CACHE_DIR` | Model cache directory | ./models |

### Vercel Configuration

The `vercel.json` file is already configured with optimal settings:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "vercel_app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "vercel_app.py"
    }
  ],
  "env": {
    "PYTHONPATH": "./"
  }
}
```

## 📊 Performance Optimization

### Cold Start Optimization

1. **Model Caching**: Models are cached after first load
2. **Lazy Loading**: Components load only when needed
3. **Optimized Dependencies**: Minimal dependency footprint

### Memory Management

- **Model Size**: ~50MB in memory
- **Data Processing**: Streaming for large datasets
- **Garbage Collection**: Automatic cleanup of unused objects

## 🔒 Security Considerations

### Production Security Checklist

- [ ] **HTTPS Only**: Ensure all traffic is encrypted
- [ ] **Input Validation**: All user inputs are validated
- [ ] **Rate Limiting**: Implement API rate limiting
- [ ] **Error Handling**: Don't expose internal errors
- [ ] **Logging**: Log requests but not sensitive data
- [ ] **CORS**: Configure appropriate CORS settings

### Data Privacy

- **No Data Storage**: The app doesn't store claim data permanently
- **In-Memory Processing**: All processing happens in memory
- **GDPR Compliant**: No personal data is retained

## 🚨 Troubleshooting

### Common Issues

#### "Models not loaded" Error

**Solution**: Wait for model initialization (30-60 seconds on first deployment)

```bash
# Check model loading status
curl https://your-app.vercel.app/health
```

#### Timeout Errors

**Solution**: Increase timeout settings or optimize model loading

```python
# In vercel_app.py
@app.on_event("startup")
async def startup_event():
    # Add timeout handling
    asyncio.create_task(load_models_async())
```

#### Memory Issues

**Solution**: Use smaller model or upgrade plan

- Vercel Pro: 1GB memory limit
- Vercel Hobby: 512MB memory limit

### Monitoring

#### Health Checks

```bash
# Basic health check
curl https://your-app.vercel.app/health

# Model information
curl https://your-app.vercel.app/model-info
```

#### Performance Monitoring

```bash
# Test prediction latency
time curl -X POST https://your-app.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "gender": "M", ...}'
```

## 📈 Scaling

### Horizontal Scaling

**Vercel**: Automatic scaling based on demand
**Container Platforms**: Use orchestration tools like Kubernetes

### Vertical Scaling

- **Memory**: Increase for larger models
- **CPU**: More cores for concurrent requests
- **Storage**: SSD for faster model loading

### Database Integration (Optional)

For production use, consider adding a database:

```python
# Example: PostgreSQL integration
import psycopg2
from sqlalchemy import create_engine

DATABASE_URL = os.environ.get('DATABASE_URL')
engine = create_engine(DATABASE_URL)
```

## 🔄 CI/CD Pipeline

### GitHub Actions (Recommended)

```yaml
# .github/workflows/deploy.yml
name: Deploy to Vercel

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Vercel
      uses: amondnet/vercel-action@v20
      with:
        vercel-token: ${{ secrets.VERCEL_TOKEN }}
        vercel-org-id: ${{ secrets.ORG_ID }}
        vercel-project-id: ${{ secrets.PROJECT_ID }}
```

### Testing Pipeline

```bash
# Run tests before deployment
python -m pytest tests/
python -m flake8 .
python -m black --check .
```

## 📱 Mobile Optimization

The web interface is responsive and works well on mobile devices:

- **Touch-friendly**: Large buttons and form fields
- **Responsive Design**: Adapts to all screen sizes
- **Fast Loading**: Optimized for mobile networks

## 🌍 Multi-region Deployment

### Vercel Edge Functions

Vercel automatically deploys to multiple regions for optimal performance:

- **London**: Primary region for UK users
- **Frankfurt**: European backup
- **Global CDN**: Static assets cached worldwide

### Custom Regions

For specific deployment regions:

```json
{
  "functions": {
    "vercel_app.py": {
      "regions": ["lhr1", "fra1", "iad1"]
    }
  }
}
```

## 📋 Pre-deployment Checklist

- [ ] **Test locally**: Application runs without errors
- [ ] **Check dependencies**: All packages in requirements.txt
- [ ] **Validate data**: Sample data loads correctly
- [ ] **Test endpoints**: All API endpoints respond
- [ ] **Mobile test**: Interface works on mobile
- [ ] **Security review**: No hardcoded credentials
- [ ] **Performance test**: Response times acceptable

## 🎯 Production Recommendations

1. **Use a CDN** for static assets
2. **Enable monitoring** for uptime tracking
3. **Set up alerts** for errors and performance issues
4. **Regular backups** of any persistent data
5. **Security updates** for all dependencies
6. **Load testing** for expected traffic

---

**Need help?** [Open an issue](https://github.com/yourusername/ClaimRiskPredictor/issues) or check the [troubleshooting guide](#troubleshooting).