import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AIML_API_KEY"),
    base_url="https://api.aimlapi.com/v1"
)

NIH_REPORTER_BASE = "https://api.reporter.nih.gov/v2/projects/search"


def search_grants_gov(keywords: list, domain: str) -> list:
    """Search Grants.gov for relevant funding opportunities."""
    query = " ".join(keywords[:3])

    payload = {
        "keyword": query,
        "oppStatuses": "posted|forecasted",
        "rows": 5,
        "startRecordNum": 0
    }

    try:
        response = requests.post(
            "https://apply07.grants.gov/grantsws/rest/opportunities/search",
            json=payload,
            timeout=10
        )

        if response.status_code != 200:
            print(f"[ELIGIBILITY AGENT] Grants.gov error: {response.status_code}")
            return []

        data = response.json()
        grants = []

        for opp in data.get("oppHits", []):
            title = (
                opp.get("oppTitle")
                or opp.get("title")
                or opp.get("opportunityTitle")
                or "Research Funding Opportunity"
            )
            agency = (
                opp.get("agencyName")
                or opp.get("agency")
                or opp.get("agencyCode")
                or "Federal Agency"
            )
            grants.append({
                "title": title,
                "agency": agency,
                "close_date": opp.get("closeDate") or opp.get("closingDate") or "See listing",
                "funding_amount": opp.get("awardCeiling") or opp.get("estimatedFunding") or "See listing",
                "opportunity_number": opp.get("oppNumber") or opp.get("opportunityNumber") or "",
                "source": "Grants.gov",
                "url": f"https://www.grants.gov/search-results-detail/{opp.get('id', '')}"
            })
        return grants

    except Exception as e:
        print(f"[ELIGIBILITY AGENT] Grants.gov exception: {e}")
        return []


def search_nih_reporter(keywords: list) -> list:
    """Search NIH Reporter for funded research projects similar to proposal."""
    query = " ".join(keywords[:3])

    payload = {
        "criteria": {
            "advanced_text_search": {
                "operator": "and",
                "search_field": "all",
                "search_text": query
            }
        },
        "limit": 5,
        "offset": 0,
        "fields": [
            "project_title",
            "agency_ic_admin",
            "fiscal_year",
            "award_amount",
            "principal_investigators",
            "project_start_date",
            "project_end_date",
            "organization",
            "appl_id",
            "abstract_text"
        ]
    }

    try:
        response = requests.post(NIH_REPORTER_BASE, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"[ELIGIBILITY AGENT] NIH Reporter error: {response.status_code}")
            return []

        data = response.json()
        projects = []
        for project in data.get("results", []):
            pi_list = project.get("principal_investigators", [{}])
            pi_name = pi_list[0].get("full_name", "Unknown") if pi_list else "Unknown"
            org = project.get("organization", {})
            org_name = org.get("org_name", "") if org else ""
            award = project.get("award_amount")
            appl_id = project.get("appl_id", "")

            projects.append({
                "title": project.get("project_title", "Unknown"),
                "agency": "NIH - " + project.get("agency_ic_admin", {}).get("abbreviation", ""),
                "fiscal_year": project.get("fiscal_year", "Unknown"),
                "funding_amount": award,
                "pi_name": pi_name,
                "organization": org_name,
                "abstract": str(project.get("abstract_text", ""))[:300] if project.get("abstract_text") else "",
                "start_date": project.get("project_start_date", ""),
                "end_date": project.get("project_end_date", ""),
                "source": "NIH Reporter",
                "url": f"https://reporter.nih.gov/project-details/{appl_id}"
            })
        return projects

    except Exception as e:
        print(f"[ELIGIBILITY AGENT] NIH Reporter exception: {e}")
        return []


def analyze_eligibility(proposal_info: dict, relevance_result: dict, grants: list, nih_projects: list) -> dict:
    """Use Claude to match proposal against available grants and assess eligibility."""

    trimmed_grants = [
        {
            "title": g.get("title", ""),
            "agency": g.get("agency", ""),
            "url": g.get("url", "")
        }
        for g in grants[:3]
    ]

    trimmed_nih = [
        {
            "title": p.get("title", ""),
            "agency": p.get("agency", ""),
            "fiscal_year": p.get("fiscal_year", "")
        }
        for p in nih_projects[:3]
    ]

    grants_text = json.dumps(trimmed_grants)
    nih_text = json.dumps(trimmed_nih)

    response = client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[
            {
                "role": "system",
                "content": """You are a grant eligibility specialist with expertise in global research funding.
                You MUST respond with valid JSON only. No markdown, no backticks, no extra text.
                Use only double quotes. Keep all string values concise under 100 characters.
                Respond in exactly this format:
                {
                    "eligible_grants": [
                        {
                            "grant_title": "title",
                            "agency": "agency name",
                            "eligibility_score": 50,
                            "match_reasons": ["reason1"],
                            "requirements_met": ["req1"],
                            "requirements_missing": ["req1"],
                            "url": "url",
                            "recommended": true
                        }
                    ],
                    "best_match": "title of best matching grant",
                    "overall_eligibility": "HIGH",
                    "action_items": ["action1"],
                    "summary": "brief summary",
                    "alternative_funders": [
                        {
                            "name": "funder name",
                            "program": "specific program name",
                            "url": "direct url to funding page",
                            "reason": "why this funder fits",
                            "region": "Global|US|EU|Asia|UK"
                        }
                    ]
                }"""
            },
            {
                "role": "user",
                "content": f"""Match this proposal to grants and suggest 3-4 alternative funders:
Title: {proposal_info.get('title')}
Domain: {proposal_info.get('domain')}
Keywords: {proposal_info.get('keywords')}

Grants.gov: {grants_text}
NIH: {nih_text}

Suggest funders from: EU Horizon Europe, Wellcome Trust, Gates Foundation, NSF, UKRI, ASEAN Foundation, WHO grants. Return valid JSON only."""
            }
        ],
        max_tokens=1500
    )

    result = response.choices[0].message.content
    result = result.replace("```json", "").replace("```", "").strip()
    start = result.find("{")
    end = result.rfind("}") + 1
    if start != -1 and end != 0:
        result = result[start:end]
    return json.loads(result)


def run_eligibility_agent(proposal_info: dict, relevance_result: dict) -> dict:
    """
    Eligibility Agent - searches Grants.gov and NIH Reporter
    and matches the proposal to available funding opportunities.
    """
    print("[ELIGIBILITY AGENT] Searching funding opportunities...")

    keywords = proposal_info.get("keywords", [])
    domain = proposal_info.get("domain", "")

    grants = search_grants_gov(keywords, domain)
    print(f"[ELIGIBILITY AGENT] Grants.gov: {len(grants)} opportunities found")

    nih_projects = search_nih_reporter(keywords)
    print(f"[ELIGIBILITY AGENT] NIH Reporter: {len(nih_projects)} projects found")

    analysis = analyze_eligibility(proposal_info, relevance_result, grants, nih_projects)

    analysis["funded_similar_projects"] = [
        {
            "title": p.get("title", ""),
            "pi_name": p.get("pi_name", ""),
            "organization": p.get("organization", ""),
            "agency": p.get("agency", ""),
            "fiscal_year": p.get("fiscal_year", ""),
            "funding_amount": p.get("funding_amount"),
            "abstract": p.get("abstract", ""),
            "url": p.get("url", "")
        }
        for p in nih_projects[:3]
    ]

    print(f"[ELIGIBILITY AGENT] Complete - eligibility: {analysis.get('overall_eligibility')}, best match: {analysis.get('best_match')}")
    return analysis


if __name__ == "__main__":
    test_proposal_info = {
        "title": "Deep Learning for Alzheimer Detection using MRI",
        "keywords": ["deep learning", "Alzheimer", "MRI", "detection"],
        "domain": "Biomedical AI",
        "summary": "A deep learning framework for early Alzheimer detection using MRI scans."
    }
    test_relevance = {
        "overall_score": 82,
        "funding_outlook": "HIGH"
    }
    result = run_eligibility_agent(test_proposal_info, test_relevance)
    print(json.dumps(result, indent=2))