from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import List, Literal
import google.generativeai as genai
from google.api_core import retry
import requests
from pathlib import Path
from datetime import datetime
from io import StringIO, BytesIO
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from gtts import gTTS
from collections import defaultdict
import time
import os
from dotenv import load_dotenv
import json
import re
import csv
import random
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable")

# Copy all your model classes from main.py here
class Contact(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    job_title: str = Field(..., min_length=1, max_length=100)
    group: Literal["A", "B"] = "A"

# ... copy other model classes ...

# Initialize FastAPI
app = FastAPI(
    title="Email Campaign Generator",
    description="Generate personalized email campaigns using Gemini AI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Configure static files and templates
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Copy your utility functions from main.py
def get_default_interests(job_title: str) -> List[str]:
    # ... copy function implementation ...
    pass

# Copy your route handlers
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ... copy other route handlers ...

# Initialize Gemini AI
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content("Test")
    if not response:
        raise ValueError("Unable to generate test content")
    logger.info("Gemini API initialized successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}")
    raise HTTPException(
        status_code=500,
        detail="Failed to initialize AI model. Please check your API key."
    )

