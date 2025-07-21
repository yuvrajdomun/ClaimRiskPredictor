"""
Main application entry point for Vercel deployment.
This file serves as the WSGI/ASGI application for the FastAPI server.
"""

from api import app

# For Vercel deployment, we need to export the FastAPI app
# Vercel will automatically detect this as the main application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)