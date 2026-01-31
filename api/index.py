from fastapi import FastAPI
from mangum import Mangum
import sys
import os

# Add parent directory to path to import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the main app
from main import app

# Wrap FastAPI app for serverless
handler = Mangum(app)
