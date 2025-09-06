# api/index.py
# Vercel Python Function entrypoint that exposes the FastAPI app defined in main.py

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so "from main import app" works even when deployed
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent  # repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the FastAPI app instance from main.py
# main.py must define: app = FastAPI(...)
from main import app  # noqa: E402
