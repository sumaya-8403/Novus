import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from orchestrator import run_novus_pipeline

load_dotenv()

app = FastAPI(title="Novus Research Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProposalRequest(BaseModel):
    proposal_text: str

@app.get("/")
def root():
    return {"status": "Novus API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "agents": 6}

@app.post("/analyze")
async def analyze_proposal(request: ProposalRequest):
    """
    Main endpoint - runs the full 6 agent pipeline
    and returns structured results.
    """
    try:
        result = run_novus_pipeline(request.proposal_text)
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)