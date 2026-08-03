"""
AI Nailysis V2 - Root Application Server Entrypoint
===================================================
Bridges execution commands (python app.py / uvicorn app.main:app) to app/main.py.
"""

import sys
import os

# Add root directory to sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.main import app

if __name__ == "__main__":
    import uvicorn
    print("\n=======================================================")
    print("AI NAILYSIS V2 RESEARCH WEB SERVER RUNNING")
    print("Local access: http://127.0.0.1:8000")
    print("=======================================================\n")
    
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
