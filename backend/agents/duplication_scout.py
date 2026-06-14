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

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
OPENALEX_BASE = "https://api.openalex.org"
ARXIV_BASE = "http://export.arxiv.org/api/query"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"

def search_semantic_scholar(keywords: list, limit: int = 5) -> list:
    """Search Semantic Scholar for papers matching keywords."""
    query = " ".join(keywords)
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,abstract,externalIds,openAccessPdf,url,publicationDate,venue"
    }

    response = requests.get(
        f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
        params=params,
        headers=headers
    )

    if response.status_code != 200:
        print(f"[DUPLICATION SCOUT] Semantic Scholar error: {response.status_code}")
        return []

    data = response.json()
    papers = []
    for paper in data.get("data", []):
        authors = paper.get("authors", [])
        author_names = [a.get("name") for a in authors[:3]]
        if len(authors) > 3:
            author_names.append("et al.")

        papers.append({
            "title": paper.get("title", "Unknown"),
            "authors": author_names,
            "year": paper.get("year"),
            "publication_date": paper.get("publicationDate", ""),
            "venue": paper.get("venue", ""),
            "abstract": paper.get("abstract", ""),
            "url": paper.get("url", ""),
            "source": "Semantic Scholar",
            "open_access": paper.get("openAccessPdf") is not None
        })

    # Sort by year descending — most recent first
    papers.sort(key=lambda x: x.get("year") or 0, reverse=True)
    return papers


def search_openalex(keywords: list, limit: int = 5) -> list:
    """Search OpenAlex for papers matching keywords."""
    query = " ".join(keywords)
    email = os.getenv("OPENALEX_EMAIL", "research@novus.ai")
    headers = {"User-Agent": f"mailto:{email}"}

    params = {
        "search": query,
        "per-page": limit,
        "select": "title,authorships,publication_year,abstract_inverted_index,doi,open_access"
    }

    response = requests.get(
        f"{OPENALEX_BASE}/works",
        params=params,
        headers=headers
    )

    if response.status_code != 200:
        print(f"[DUPLICATION SCOUT] OpenAlex error: {response.status_code}")
        return []

    data = response.json()
    papers = []
    for work in data.get("results", []):
        doi = work.get("doi", "")
        is_open = work.get("open_access", {}).get("is_oa", False)
        papers.append({
            "title": work.get("title", "Unknown"),
            "authors": [
                a.get("author", {}).get("display_name")
                for a in work.get("authorships", [])
            ],
            "year": work.get("publication_year"),
            "abstract": "",
            "url": doi if doi else "",
            "source": "OpenAlex",
            "open_access": is_open
        })
    return papers


def search_arxiv(keywords: list, limit: int = 5) -> list:
    """Search arXiv for papers matching keywords."""
    query = "+AND+".join(keywords[:3])
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    response = requests.get(ARXIV_BASE, params=params)
    if response.status_code != 200:
        print(f"[DUPLICATION SCOUT] arXiv error: {response.status_code}")
        return []

    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns)
        summary = entry.find("atom:summary", ns)
        link = entry.find("atom:id", ns)
        published = entry.find("atom:published", ns)
        authors = entry.findall("atom:author", ns)

        author_names = [
            a.find("atom:name", ns).text
            for a in authors[:3]
            if a.find("atom:name", ns) is not None
        ]
        if len(authors) > 3:
            author_names.append("et al.")

        pub_date = published.text[:10] if published is not None else ""
        year = int(pub_date[:4]) if pub_date else None

        papers.append({
            "title": title.text.strip() if title is not None else "Unknown",
            "authors": author_names,
            "year": year,
            "publication_date": pub_date,
            "venue": "arXiv",
            "abstract": summary.text.strip() if summary is not None else "",
            "url": link.text.strip() if link is not None else "",
            "source": "arXiv",
            "open_access": True
        })
    return papers


def check_unpaywall(doi: str) -> str:
    """Check if a paywalled paper has a free version via Unpaywall."""
    if not doi:
        return None
    email = os.getenv("OPENALEX_EMAIL", "research@novus.ai")
    response = requests.get(f"{UNPAYWALL_BASE}/{doi}?email={email}")
    if response.status_code == 200:
        data = response.json()
        if data.get("is_oa"):
            return data.get("best_oa_location", {}).get("url_for_pdf")
    return None


def analyze_duplicates(proposal_info: dict, papers: list) -> dict:
    """Use Claude to analyze similarity between proposal and found papers."""
    
    # Only send titles and abstracts to Claude, keep it short
    papers_summary = [
        {
            "title": p["title"],
            "abstract": p["abstract"][:200] if p["abstract"] else "No abstract",
            "year": p.get("year"),
            "source": p["source"]
        }
        for p in papers[:10]
    ]

    response = client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[
            {
                "role": "system",
                "content": """You are a research duplication detection specialist.
                Analyze similarity between a research proposal and existing papers.
                Respond ONLY in this JSON format, no preamble, no markdown:
                {
                    "duplicate_risk": "HIGH|MEDIUM|LOW",
                    "recommendation": "proceed|revise|reconsider",
                    "summary": "brief summary of findings under 300 chars",
                    "similar_paper_titles": ["title1", "title2", "title3"]
                }"""
            },
            {
                "role": "user",
                "content": f"""Research Proposal:
Title: {proposal_info.get('title')}
Keywords: {proposal_info.get('keywords')}
Summary: {proposal_info.get('summary')}

Papers found:
{json.dumps(papers_summary)}

Identify which papers are most similar and assess duplication risk."""
            }
        ],
        max_tokens=800
    )

    result = response.choices[0].message.content
    result = result.replace("```json", "").replace("```", "").strip()
    start = result.find("{")
    end = result.rfind("}") + 1
    if start != -1 and end > start:
        result = result[start:end]
    return json.loads(result)


def run_duplication_scout(proposal_info: dict) -> dict:
    """
    Duplication Scout Agent - searches multiple sources for similar papers
    and flags potential duplicates including restricted access papers.
    """
    print("[DUPLICATION SCOUT] Starting literature search...")

    keywords = proposal_info.get("keywords", [])

    ss_papers = search_semantic_scholar(keywords)
    print(f"[DUPLICATION SCOUT] Semantic Scholar: {len(ss_papers)} papers found")

    oa_papers = search_openalex(keywords)
    print(f"[DUPLICATION SCOUT] OpenAlex: {len(oa_papers)} papers found")

    arxiv_papers = search_arxiv(keywords)
    print(f"[DUPLICATION SCOUT] arXiv: {len(arxiv_papers)} papers found")

    all_papers = ss_papers + oa_papers + arxiv_papers

    # Sort all papers by year descending - most recent first
    all_papers.sort(key=lambda x: x.get("year") or 0, reverse=True)

    restricted_papers = [p for p in all_papers if not p["open_access"]]
    open_papers = [p for p in all_papers if p["open_access"]]

    print(f"[DUPLICATION SCOUT] Total: {len(all_papers)} papers ({len(open_papers)} open, {len(restricted_papers)} restricted)")

    analysis = analyze_duplicates(proposal_info, all_papers)

    # Use REAL papers from APIs with real authors/years
    # Match Claude's identified similar titles back to real papers
    similar_titles = analysis.get("similar_paper_titles", [])
    
    similar_papers = []
    used_titles = set()
    
    # First add papers Claude flagged as similar
    for title in similar_titles:
        for paper in all_papers:
            if paper["title"] not in used_titles:
                if title.lower()[:40] in paper["title"].lower() or paper["title"].lower()[:40] in title.lower():
                    similar_papers.append({
                        "title": paper["title"],
                        "authors": paper.get("authors", []),
                        "year": paper.get("year"),
                        "venue": paper.get("venue", paper.get("source", "")),
                        "abstract": paper.get("abstract", ""),
                        "url": paper.get("url", ""),
                        "open_access": paper.get("open_access", True),
                        "source": paper.get("source", ""),
                        "similarity_reason": "Identified as similar by semantic analysis"
                    })
                    used_titles.add(paper["title"])
                    break

    # Fill remaining slots with top papers by recency
    for paper in all_papers:
        if len(similar_papers) >= 6:
            break
        if paper["title"] not in used_titles:
            similar_papers.append({
                "title": paper["title"],
                "authors": paper.get("authors", []),
                "year": paper.get("year"),
                "venue": paper.get("venue", paper.get("source", "")),
                "abstract": paper.get("abstract", ""),
                "url": paper.get("url", ""),
                "open_access": paper.get("open_access", True),
                "source": paper.get("source", ""),
                "similarity_reason": "Related paper found in literature search"
            })
            used_titles.add(paper["title"])

    result = {
        "duplicate_risk": analysis.get("duplicate_risk"),
        "recommendation": analysis.get("recommendation"),
        "summary": analysis.get("summary"),
        "similar_papers": similar_papers,
        "restricted_access_papers": [
            {
                "title": p["title"],
                "authors": p.get("authors", []),
                "year": p.get("year"),
                "source": p["source"],
                "url": p["url"],
                "note": "Full text unavailable - institutional access required. Title and overview suggests potential overlap. Recommend manual review."
            }
            for p in restricted_papers[:3]
        ],
        "total_papers_searched": len(all_papers)
    }

    print(f"[DUPLICATION SCOUT] Complete - risk: {result['duplicate_risk']}, recommendation: {result['recommendation']}")
    return result


if __name__ == "__main__":
    test_proposal_info = {
        "title": "Deep Learning for Alzheimer Detection using MRI",
        "keywords": ["deep learning", "Alzheimer", "MRI", "detection"],
        "domain": "Biomedical AI",
        "summary": "A deep learning framework for early Alzheimer detection using MRI scans."
    }
    result = run_duplication_scout(test_proposal_info)
    print(json.dumps(result, indent=2))