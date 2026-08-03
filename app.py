from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import os
import socket
import ai_nailysis_pipeline as pipeline

app = FastAPI(title="AI Nailysis Backend")

# Ensure static and templates folders exist
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request Model for Chatbot
class ChatRequest(BaseModel):
    message: str

def get_local_ip():
    """Retrieves the local LAN IP address of the machine to help user access it on mobile."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def generate_self_signed_cert():
    """Generates self-signed SSL certificate files key.pem and cert.pem to enable secure HTTPS."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    import datetime

    # Generate key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Generate cert
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"CA"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"San Francisco"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"AI Nailysis"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow() - datetime.timedelta(days=1)
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=365 * 10)
    ).add_extension(
        x509.SubjectAlternativeName([x509.DNSName(u"localhost"), x509.DNSName(u"127.0.0.1")]),
        critical=False,
    ).sign(key, hashes.SHA256())

    # Write key
    with open("key.pem", "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    # Write cert
    with open("cert.pem", "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    print("[OK] Generated self-signed SSL certificates (key.pem, cert.pem).")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    template_path = "templates/index.html"
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Template index.html not found")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return JSONResponse(status_code=400, content={"detail": "Invalid image format"})
        
        results = pipeline.analyze_image_bgr(img_bgr)
        return JSONResponse(content=results)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Internal pipeline error: {str(e)}"})

@app.post("/analyze_batch")
async def analyze_batch(files: list[UploadFile] = File(...)):
    results = []
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    
    try:
        for idx, file in enumerate(files):
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                results.append({
                    "finger": finger_names[idx] if idx < len(finger_names) else f"Nail {idx+1}",
                    "error": "Invalid image format"
                })
                continue
            
            res = pipeline.analyze_image_bgr(img_bgr)
            res["finger"] = finger_names[idx] if idx < len(finger_names) else f"Nail {idx+1}"
            results.append(res)
            
        return JSONResponse(content=results)
        
    except Exception as e:
        print(f"Error processing batch: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Internal pipeline batch error: {str(e)}"})

@app.get("/db")
async def get_database():
    return JSONResponse(content=pipeline.disease_info)

class LogRequest(BaseModel):
    level: str
    message: str

@app.post("/log")
async def client_log(req: LogRequest):
    print(f"[MobileClient {req.level.upper()}] {req.message}")
    return {"status": "ok"}

@app.post("/chat")
async def chat_assistant(req: ChatRequest):
    user_msg = req.message.lower().strip()
    reply = ""
    
    # Keyword-based matching
    if any(k in user_msg for k in ["hello", "hi", "hey", "greetings"]):
        reply = "Hello! I am your AI Nail Care Assistant. I can answer questions about nail hygiene, nail polish effects, and describe specific pathologies like melanoma or fungal infections. Ask me about a disease or how to clean your nails!"
        
    elif any(k in user_msg for k in ["fungus", "fungal", "onychomycosis", "yellow"]):
        reply = (
            "Onychomycosis is a fungal nail infection causing thickening, crumbling, and yellowing. "
            "Best precautions: Keep feet and hands dry and clean, use breathable footwear, and never share manicuring tools. "
            "Treatment options: Topical antifungal lacquers (e.g., Amorolfine) or oral prescription medications. Consult a podiatrist or dermatologist."
        )
        
    elif any(k in user_msg for k in ["melanoma", "dark streak", "black stripe", "cancer", "streak"]):
        reply = (
            "Melanoma of the nail unit (subungual melanoma) often appears as a dark vertical band or streak. "
            "IMPORTANT: While dark streaks can be benign, nail melanoma is a serious condition. "
            "If you see a new, darkening, or expanding vertical streak without prior injury, "
            "please consult a dermatologist immediately for a biopsy. Early detection is critical."
        )
        
    elif any(k in user_msg for k in ["blue", "cyanosis", "oxygen", "purple nails"]):
        reply = (
            "Cyanosis is a bluish discoloration of the nails. It is usually caused by low oxygen levels "
            "in the bloodstream or poor circulation. This can be associated with cardiorespiratory conditions, "
            "exposure to extreme cold, or vascular issues. You should seek a physician's evaluation to check oxygen saturation levels."
        )
        
    elif any(k in user_msg for k in ["clubbing", "curved fingers", "drumstick", "rounded finger"]):
        reply = (
            "Nail clubbing is characterized by nails that curve around enlarged fingertips, making the nail bed feel soft. "
            "It is often associated with chronic lung diseases (COPD, lung cancer), heart conditions, or gastrointestinal issues. "
            "Since it usually develops over years, a full clinical examination is recommended to find the underlying systemic cause."
        )
        
    elif any(k in user_msg for k in ["claw", "onychogryphosis", "ram's horn", "thick curved"]):
        reply = (
            "Onychogryphosis (Ram's Horn Nails) causes severe thickening and curvature, usually on the big toe. "
            "It is often due to repetitive trauma, neglected trimming, or poor circulation. "
            "Care includes regular professional podiatric debridement (thinning/trimming) and wearing wide-fitting shoes. Surgery is rare."
        )
        
    elif any(k in user_msg for k in ["psoriasis", "pit", "pitting", "autoimmune"]):
        reply = (
            "Nail psoriasis affects up to 80% of people with psoriasis, causing pitting, crumbling, and discolored spots ('oil drops'). "
            "Since it is an autoimmune condition, treatment involves prescription topical corticosteroids, vitamin D analogs, "
            "or systemic/biological therapies under a dermatologist's guidance. Keep nails trimmed short to avoid friction."
        )
        
    elif any(k in user_msg for k in ["polish", "paint", "art", "manicure"]):
        reply = (
            "Nail polish and nail art apply a solid, high-saturation layer over the nail. "
            "This blocks the camera/AI from seeing the color, texture, and anomalies of the nail bed, masking signs of fungus, "
            "cyanosis, or melanoma. We highly recommend scanning clean, bare nails for accurate results."
        )
        
    elif any(k in user_msg for k in ["clean", "hygiene", "trim", "prevent", "care"]):
        reply = (
            "Here are the golden rules of nail hygiene:\n"
            "1. Always cut nails straight across to avoid ingrown nails.\n"
            "2. Thoroughly sanitize clippers and nail files after each use.\n"
            "3. Keep hands and feet completely dry, especially after bathing.\n"
            "4. Apply a moisturizing cuticle cream nightly.\n"
            "5. Do not bite or pick at nails/hangnails."
        )
        
    else:
        reply = (
            "I can assist you with nail diagnostics! The AI checks for 6 major pathologies: "
            "Clubbing, Cyanosis, Melanoma, Onychogryphosis, Onychomycosis, and Psoriasis. "
            "Please ask me about any of these conditions, or tell me about your symptoms for care advice."
        )
        
    return JSONResponse(content={"reply": reply})

if __name__ == "__main__":
    import uvicorn
    # Generate self-signed SSL certs if not present
    if not os.path.exists("key.pem") or not os.path.exists("cert.pem"):
        try:
            generate_signed = True
            generate_self_signed_cert()
        except Exception as e:
            print(f"[Error] Failed to generate self-signed cert: {e}")
            
    local_ip = get_local_ip()
    print(f"\n=======================================================")
    print(f"AI NAILYSIS WEB SERVER RUNNING OVER HTTPS")
    print(f"Local access: https://127.0.0.1:8000")
    print(f"Mobile access (on same Wi-Fi): https://{local_ip}:8000")
    print(f"Note: Since this is a self-signed certificate, your browser")
    print(f"will show a warning. Click 'Advanced' -> 'Proceed' to open it.")
    print(f"=======================================================\n")
    
    # Run uvicorn on plain HTTP (tunnels will wrap this in public HTTPS)
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
