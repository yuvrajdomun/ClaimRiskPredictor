"""
Main entry point for Vercel Streamlit deployment.
This file imports and runs the Streamlit app directly.
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the Streamlit app components
from streamlit_app import main

# Set Streamlit configuration for deployment
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_CORS", "false")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

def application():
    """Entry point for WSGI/ASGI servers."""
    # Import and run the Streamlit app
    try:
        import streamlit as st
        from streamlit.web import cli as stcli
        
        # Configure Streamlit
        sys.argv = [
            "streamlit", 
            "run", 
            str(current_dir / "streamlit_app.py"),
            "--server.headless=true",
            "--server.enableCORS=false",
            "--server.fileWatcherType=none",
            "--browser.gatherUsageStats=false"
        ]
        
        # Start Streamlit CLI
        stcli.main()
        
    except Exception as e:
        print(f"Error starting Streamlit: {e}")
        # Fallback to direct import
        main()

# For Vercel
app = application

if __name__ == "__main__":
    application()