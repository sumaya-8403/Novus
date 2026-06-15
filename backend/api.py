import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from orchestrator import run_novus_pipeline
from agents.duplication_scout import search_semantic_scholar, search_openalex, search_arxiv

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

@app.get("/more-papers")
async def get_more_papers(keywords: str):
    """
    Fetch additional papers from academic APIs.
    Called when user clicks Show More in the UI.
    """
    try:
        keyword_list = [k.strip() for k in keywords.split(",")]

        ss_papers = search_semantic_scholar(keyword_list, limit=7)
        oa_papers = search_openalex(keyword_list, limit=7)
        arxiv_papers = search_arxiv(keyword_list, limit=6)

        all_papers = ss_papers + oa_papers + arxiv_papers
        all_papers.sort(key=lambda x: x.get("year") or 0, reverse=True)

        return {
            "success": True,
            "papers": all_papers,
            "total": len(all_papers)
        }
    except Exception as e:
        return {
            "success": False,
            "papers": [],
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)