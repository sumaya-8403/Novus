import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AIML_API_KEY"),
    base_url="https://api.aimlapi.com/v1"
)

def run_compliance_agent(
    proposal_info: dict,
    eligibility_result: dict,
    proposal_draft: dict
) -> dict:
    """
    Compliance Agent - reviews the final proposal draft against
    grant requirements and flags any compliance issues.
    """
    print("[COMPLIANCE AGENT] Reviewing proposal for compliance...")

    best_grant = eligibility_result.get("best_match", "General Research Grant")
    requirements_missing = []
    for grant in eligibility_result.get("eligible_grants", []):
        if grant.get("recommended"):
            requirements_missing = grant.get("requirements_missing", [])
            break

    # Trim proposal to avoid long context JSON issues
    trimmed_draft = {
        "executive_summary": str(proposal_draft.get("executive_summary", ""))[:300],
        "methodology": str(proposal_draft.get("methodology", ""))[:300],
        "innovation_statement": str(proposal_draft.get("innovation_statement", ""))[:200],
        "expected_outcomes": proposal_draft.get("expected_outcomes", [])[:3],
        "broader_impacts": str(proposal_draft.get("broader_impacts", ""))[:200],
        "budget_justification": str(proposal_draft.get("budget_justification", ""))[:200],
        "timeline": proposal_draft.get("timeline", [])[:2]
    }

    response = client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[
            {
                "role": "system",
                "content": """You are a research grant compliance specialist.
                You MUST respond with valid JSON only. No markdown, no backticks, no extra text.
                Use only double quotes. Keep all string values concise under 100 characters.
                Respond in exactly this format:
                {
                    "compliance_score": 80,
                    "status": "APPROVED",
                    "passed_checks": ["check1", "check2"],
                    "failed_checks": ["check1"],
                    "ethical_concerns": [],
                    "missing_sections": [],
                    "improvement_suggestions": ["suggestion1"],
                    "ready_to_submit": true,
                    "compliance_summary": "brief summary under 100 chars",
                    "next_steps": ["step1", "step2"]
                }"""
            },
            {
                "role": "user",
                "content": f"""Review this proposal for compliance:

Title: {proposal_info.get('title')}
Target Grant: {best_grant}
Missing Requirements: {requirements_missing}

Proposal Summary:
{json.dumps(trimmed_draft)}

Return valid JSON only."""
            }
        ],
        max_tokens=1000
    )

    result = response.choices[0].message.content
    result = result.replace("```json", "").replace("```", "").strip()
    start = result.find("{")
    end = result.rfind("}") + 1
    if start != -1 and end != 0:
        result = result[start:end]

    parsed = json.loads(result)
    print(f"[COMPLIANCE AGENT] Complete - status: {parsed['status']}, score: {parsed['compliance_score']}/100, ready: {parsed['ready_to_submit']}")
    return parsed


if __name__ == "__main__":
    test_proposal_info = {
        "title": "Deep Learning for Alzheimer Detection using MRI",
        "domain": "Biomedical AI"
    }
    test_eligibility = {
        "best_match": "NIH R01 Biomedical AI Grant",
        "eligible_grants": [
            {
                "grant_title": "NIH R01 Biomedical AI Grant",
                "recommended": True,
                "requirements_missing": ["IRB approval documentation"]
            }
        ]
    }
    test_draft = {
        "executive_summary": "This research develops deep learning for Alzheimer detection...",
        "methodology": "We will use CNNs trained on ADNI dataset...",
        "innovation_statement": "First framework combining MRI and genetic markers...",
        "expected_outcomes": ["90% detection accuracy", "Open source toolkit"],
        "broader_impacts": "Improved early intervention for millions of patients",
        "timeline": [{"phase": "Phase 1", "duration": "months 1-6", "activities": "Data collection"}],
        "budget_justification": "Computing resources and research staff"
    }

    result = run_compliance_agent(
        test_proposal_info,
        test_eligibility,
        test_draft
    )
    print(json.dumps(result, indent=2))