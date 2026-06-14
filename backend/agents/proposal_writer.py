import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AIML_API_KEY"),
    base_url="https://api.aimlapi.com/v1"
)

def run_proposal_writer(
    proposal_info: dict,
    duplication_result: dict,
    relevance_result: dict,
    eligibility_result: dict
) -> dict:
    print("[PROPOSAL WRITER] Drafting grant proposal...")

    best_grant = eligibility_result.get("best_match", "General Research Grant")
    strengths = relevance_result.get("strengths", [])[:2]
    trending = relevance_result.get("trending_alignment", [])[:2]

    response = client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[
            {
                "role": "system",
                "content": """You are a grant proposal writer.
                Respond with ONLY a valid JSON object. No markdown. No backticks. No extra text before or after.
                Every string value must be under 200 characters.
                Every array must have at most 3 items.
                Use only double quotes. Escape apostrophes as \\u0027 if needed.
                Return exactly this structure:
                {
                "executive_summary": "string",
                "background_and_significance": "string",
                "research_objectives": ["obj1", "obj2", "obj3"],
                "methodology": "string",
                "innovation_statement": "string",
                "expected_outcomes": ["outcome1", "outcome2", "outcome3"],
                "broader_impacts": "string",
                "timeline": [
                {"phase": "Phase 1", "duration": "months 1-6", "activities": "string"},
                {"phase": "Phase 2", "duration": "months 7-12", "activities": "string"}
                ],
                "budget_justification": "string",
                "team_qualifications": "string"
                }"""
            },
            {
                "role": "user",
                "content": f"""Write a grant proposal:
Title: {proposal_info.get('title')}
Domain: {proposal_info.get('domain')}
Keywords: {str(proposal_info.get('keywords', []))[:200]}
Target Grant: {str(best_grant)[:100]}
Strengths: {str(strengths)[:100]}
Score: {relevance_result.get('overall_score')}/100

Return valid JSON only. Keep every string under 200 characters."""
            }
        ],
        max_tokens=1500
    )

    result = response.choices[0].message.content.strip()
    result = result.replace("```json", "").replace("```", "").strip()
    start = result.find("{")
    end = result.rfind("}") + 1
    if start != -1 and end > start:
        result = result[start:end]

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        # Fallback: clean common JSON breaking characters
        result = result.replace("\u2019", "'").replace("\u2018", "'")
        result = result.replace("\u201c", '"').replace("\u201d", '"')
        result = result.replace("\n", " ").replace("\t", " ")
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            # Last resort fallback
            parsed = {
                "executive_summary": "Proposal generation encountered a formatting issue. Please retry.",
                "background_and_significance": "",
                "research_objectives": [],
                "methodology": "",
                "innovation_statement": "",
                "expected_outcomes": [],
                "broader_impacts": "",
                "timeline": [],
                "budget_justification": "",
                "team_qualifications": ""
            }
    print("[PROPOSAL WRITER] Complete - proposal drafted successfully")
    return parsed


if __name__ == "__main__":
    test_proposal_info = {
        "title": "Deep Learning for Alzheimer Detection using MRI",
        "keywords": ["deep learning", "Alzheimer", "MRI"],
        "domain": "Biomedical AI",
        "summary": "A deep learning framework for early Alzheimer detection."
    }
    test_duplication = {"duplicate_risk": "MEDIUM"}
    test_relevance = {
        "overall_score": 82,
        "funding_outlook": "HIGH",
        "strengths": ["Novel approach", "High impact"],
        "trending_alignment": ["AI in healthcare"]
    }
    test_eligibility = {"best_match": "NIH R01 Biomedical AI Grant"}

    result = run_proposal_writer(
        test_proposal_info,
        test_duplication,
        test_relevance,
        test_eligibility
    )
    print(json.dumps(result, indent=2))