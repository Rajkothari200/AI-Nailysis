"""
AI Nailysis V2 - Modular FastAPI Application Core
===================================================
Main web application initializer mounting static assets and including API router endpoints.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from .api.routes import router as api_router
from utils.logger import get_logger

logger = get_logger("AppCore")

app = FastAPI(title="AI Nailysis V2 - Clinical Diagnostic System", version="2.0.0")

# Ensure static and templates folders exist
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API Routes
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Serves the main application dashboard HTML page."""
    template_path = "templates/index.html"
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Template index.html not found")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
