import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://api.anthropic.com/v1"
)

def run_relevance_agent(proposal_info: dict, duplication_result: dict) -> dict:
    """
    Relevance Agent - scores the proposal against:
    - Current funding trends
    - Research impact potential
    - Strategic alignment
    - Novelty score based on duplication results
    """
    print("[RELEVANCE AGENT] Scoring proposal relevance...")

    duplicate_risk = duplication_result.get("duplicate_risk", "UNKNOWN")
    recommendation = duplication_result.get("recommendation", "unknown")

    response = client.chat.completions.create(
        model="claude-opus-4-6",
        messages=[
            {
                "role": "system",
                "content": """You are a research relevance and funding trends specialist.
                Your job is to score research proposals based on current trends,
                impact potential, and strategic value.
                Respond ONLY in this JSON format, no preamble, no markdown:
                {
                    "relevance_score": 0-100,
                    "novelty_score": 0-100,
                    "impact_score": 0-100,
                    "overall_score": 0-100,
                    "trending_alignment": ["trend1", "trend2"],
                    "strengths": ["strength1", "strength2"],
                    "weaknesses": ["weakness1", "weakness2"],
                    "funding_outlook": "HIGH|MEDIUM|LOW",
                    "recommendation": "strongly recommend|recommend|neutral|do not recommend",
                    "justification": "detailed justification paragraph"
                }"""
            },
            {
                "role": "user",
                "content": f"""Evaluate this research proposal for relevance and funding potential:

Title: {proposal_info.get('title')}
Domain: {proposal_info.get('domain')}
Keywords: {proposal_info.get('keywords')}
Summary: {proposal_info.get('summary')}
Research Questions: {proposal_info.get('research_questions')}

Duplication Risk: {duplicate_risk}
Duplication Recommendation: {recommendation}

Score this proposal on relevance to current funding trends, 
novelty, and potential research impact."""
            }
        ],
        max_tokens=1500
    )

    result = response.choices[0].message.content
    result = result.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(result)

    print(f"[RELEVANCE AGENT] Complete - overall score: {parsed['overall_score']}/100, funding outlook: {parsed['funding_outlook']}")
    return parsed


if __name__ == "__main__":
    test_proposal_info = {
        "title": "Deep Learning for Alzheimer Detection using MRI",
        "keywords": ["deep learning", "Alzheimer", "MRI", "detection"],
        "domain": "Biomedical AI",
        "summary": "A deep learning framework for early Alzheimer detection using MRI scans.",
        "research_questions": ["Can CNNs detect early Alzheimer from MRI?"]
    }
    test_duplication = {
        "duplicate_risk": "MEDIUM",
        "recommendation": "revise"
    }
    result = run_relevance_agent(test_proposal_info, test_duplication)
    print(json.dumps(result, indent=2))