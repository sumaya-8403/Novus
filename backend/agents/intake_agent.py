import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AIML_API_KEY"),
    base_url="https://api.aimlapi.com/v1"
)

def run_intake_agent(proposal_text: str) -> dict:
    """
    Intake Agent - parses the research proposal and extracts:
    - Title
    - Keywords
    - Research domain
    - Summary
    - Research questions
    """
    print("[INTAKE AGENT] Analyzing proposal...")

    response = client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[
            {
                "role": "system",
                "content": """You are a research intake specialist. 
                Analyze the given research proposal and extract structured information.
                Respond ONLY in this JSON format, no preamble, no markdown:
                {
                    "title": "extracted or inferred title",
                    "keywords": ["keyword1", "keyword2", "keyword3"],
                    "domain": "research domain (e.g. AI, Biomedical, Environmental)",
                    "summary": "2-3 sentence summary of the proposal",
                    "research_questions": ["question1", "question2"]
                }"""
            },
            {
                "role": "user",
                "content": f"Analyze this research proposal:\n\n{proposal_text}"
            }
        ],
        max_tokens=1000
    )

    result = response.choices[0].message.content
    result = result.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(result)

    print(f"[INTAKE AGENT] Complete - domain: {parsed['domain']}, keywords: {parsed['keywords']}")
    return parsed

if __name__ == "__main__":
    test_proposal = """
    This research proposes to develop a deep learning framework for early 
    detection of Alzheimer's disease using MRI brain scans. We will use 
    convolutional neural networks trained on the ADNI dataset to classify 
    patients into early, mild, and severe categories with the goal of 
    improving early intervention outcomes.
    """
    result = run_intake_agent(test_proposal)
    print(result)