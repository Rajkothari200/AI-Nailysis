"""
AI Nailysis V2 - Modular FastAPI Route Handlers
================================================
Defines API endpoints for image analysis, batch processing, database lookup, log reception, and chat.
"""

from typing import List, Dict, Any
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import inference.legacy_adapter as adapter
from utils.logger import get_logger

logger = get_logger("APIRoutes")
router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class LogRequest(BaseModel):
    level: str
    message: str


@router.post("/analyze")
async def analyze_single(file: UploadFile = File(...)):
    """
    Analyzes a single uploaded nail image using the V2 pipeline.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return JSONResponse(status_code=400, content={"detail": "Invalid image format"})
            
        results = adapter.analyze_image_bgr(img_bgr)
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error processing /analyze image: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Internal pipeline error: {str(e)}"})


@router.post("/analyze_batch")
async def analyze_batch(files: List[UploadFile] = File(...)):
    """
    Analyzes a batch of uploaded finger images (Thumb, Index, Middle, Ring, Pinky).
    """
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    img_list = []
    
    try:
        for idx, file in enumerate(files):
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                img_list.append(img_bgr)
            else:
                img_list.append(np.zeros((224, 224, 3), dtype=np.uint8))
                
        results = adapter.analyze_batch_bgr(img_list, finger_names)
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error processing /analyze_batch images: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Internal batch pipeline error: {str(e)}"})


@router.get("/db")
async def get_database():
    """
    Returns the clinical disease information database.
    """
    if hasattr(adapter, 'v2_pipeline') and adapter.v2_pipeline:
        return JSONResponse(content=adapter.v2_pipeline.disease_info_db)
    else:
        import ai_nailysis_pipeline as legacy_pipe
        return JSONResponse(content=legacy_pipe.disease_info)


@router.post("/log")
async def client_log(req: LogRequest):
    """
    Receives real-time mobile browser client console logs for debugging.
    """
    logger.info(f"[MobileClient {req.level.upper()}] {req.message}")
    return {"status": "ok"}


@router.post("/chat")
async def chat_assistant(req: ChatRequest):
    """
    Keyword-matching assistant for clinical nail hygiene and pathological information.
    """
    user_msg = req.message.lower().strip()
    
    if any(k in user_msg for k in ["hello", "hi", "hey", "greetings"]):
        reply = "Hello! I am your AI Nail Care Assistant. Ask me about a disease, hygiene tips, or nail polish effects!"
    elif any(k in user_msg for k in ["fungus", "fungal", "onychomycosis", "yellow"]):
        reply = "Onychomycosis causes thickening, crumbling, and yellowing. Precautions: Keep feet/hands dry and clean. Treatment: Antifungal lacquers (e.g. Amorolfine) or oral prescription medications."
    elif any(k in user_msg for k in ["melanoma", "dark streak", "black stripe", "cancer"]):
        reply = "Subungual melanoma appears as a dark vertical band under the nail. If you see a new, darkening, or expanding vertical streak without prior injury, consult a dermatologist immediately."
    elif any(k in user_msg for k in ["blue", "cyanosis", "oxygen"]):
        reply = "Cyanosis is a bluish discoloration of the nails caused by low oxygen levels in the bloodstream or poor circulation. Seek medical evaluation immediately."
    elif any(k in user_msg for k in ["clubbing", "curved"]):
        reply = "Nail clubbing is characterized by nails that curve around enlarged fingertips. It is often linked to chronic lung or heart disease."
    elif any(k in user_msg for k in ["clean", "hygiene", "trim"]):
        reply = "Golden rules of nail hygiene: 1. Cut nails straight across. 2. Sanitize clippers. 3. Keep hands dry. 4. Apply cuticle cream."
    else:
        reply = "I can assist you with nail diagnostics! The AI checks for 6 major pathologies: Clubbing, Cyanosis, Melanoma, Onychogryphosis, Onychomycosis, and Psoriasis."
        
    return JSONResponse(content={"reply": reply})
