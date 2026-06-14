import os
import json
import agentops
from dotenv import load_dotenv

from agents.intake_agent import run_intake_agent
from agents.duplication_scout import run_duplication_scout
from agents.relevance_agent import run_relevance_agent
from agents.eligibility_agent import run_eligibility_agent
from agents.proposal_writer import run_proposal_writer
from agents.compliance_agent import run_compliance_agent

load_dotenv()

agentops.init(api_key=os.getenv("AGENTOPS_API_KEY"))

def run_novus_pipeline(proposal_text: str) -> dict:
    """
    Main Novus orchestrator - runs all 6 agents in sequence,
    passing context from each agent to the next.
    """
    print("[NOVUS] Starting research intelligence pipeline...")
    print("=" * 60)

    results = {
        "proposal_text": proposal_text,
        "pipeline_status": "running",
        "stages": {}
    }

    # Stage 1 - Intake Agent
    print("[NOVUS] Stage 1/6 - Intake Agent")
    try:
        intake_result = run_intake_agent(proposal_text)
        results["stages"]["intake"] = {
            "status": "complete",
            "result": intake_result
        }
    except Exception as e:
        print(f"[NOVUS] Intake Agent failed: {e}")
        results["stages"]["intake"] = {"status": "failed", "error": str(e)}
        results["pipeline_status"] = "failed"
        return results

    print("=" * 60)

    # Stage 2 - Duplication Scout
    print("[NOVUS] Stage 2/6 - Duplication Scout")
    try:
        duplication_result = run_duplication_scout(intake_result)
        results["stages"]["duplication"] = {
            "status": "complete",
            "result": duplication_result
        }
    except Exception as e:
        print(f"[NOVUS] Duplication Scout failed: {e}")
        results["stages"]["duplication"] = {"status": "failed", "error": str(e)}
        results["pipeline_status"] = "failed"
        return results

    print("=" * 60)

    # Stage 3 - Relevance Agent
    print("[NOVUS] Stage 3/6 - Relevance Agent")
    try:
        relevance_result = run_relevance_agent(intake_result, duplication_result)
        results["stages"]["relevance"] = {
            "status": "complete",
            "result": relevance_result
        }
    except Exception as e:
        print(f"[NOVUS] Relevance Agent failed: {e}")
        results["stages"]["relevance"] = {"status": "failed", "error": str(e)}
        results["pipeline_status"] = "failed"
        return results

    print("=" * 60)

    # Stage 4 - Eligibility Agent
    print("[NOVUS] Stage 4/6 - Eligibility Agent")
    try:
        eligibility_result = run_eligibility_agent(intake_result, relevance_result)
        results["stages"]["eligibility"] = {
            "status": "complete",
            "result": eligibility_result
        }
    except Exception as e:
        print(f"[NOVUS] Eligibility Agent failed: {e}")
        results["stages"]["eligibility"] = {"status": "failed", "error": str(e)}
        results["pipeline_status"] = "failed"
        return results

    print("=" * 60)

    # Stage 5 - Proposal Writer
    print("[NOVUS] Stage 5/6 - Proposal Writer")
    try:
        proposal_draft = run_proposal_writer(
            intake_result,
            duplication_result,
            relevance_result,
            eligibility_result
        )
        results["stages"]["proposal"] = {
            "status": "complete",
            "result": proposal_draft
        }
    except Exception as e:
        print(f"[NOVUS] Proposal Writer failed: {e}")
        results["stages"]["proposal"] = {"status": "failed", "error": str(e)}
        results["pipeline_status"] = "failed"
        return results

    print("=" * 60)

    # Stage 6 - Compliance Agent
    print("[NOVUS] Stage 6/6 - Compliance Agent")
    try:
        compliance_result = run_compliance_agent(
            intake_result,
            eligibility_result,
            proposal_draft
        )
        results["stages"]["compliance"] = {
            "status": "complete",
            "result": compliance_result
        }
    except Exception as e:
        print(f"[NOVUS] Compliance Agent failed: {e}")
        results["stages"]["compliance"] = {"status": "failed", "error": str(e)}
        results["pipeline_status"] = "failed"
        return results

    print("=" * 60)
    results["pipeline_status"] = "complete"
    print("[NOVUS] Pipeline complete - all 6 agents finished successfully")

    return results


if __name__ == "__main__":
    test_proposal = """
    This research proposes to develop a deep learning framework for early
    detection of Alzheimer's disease using MRI brain scans. We will use
    convolutional neural networks trained on the ADNI dataset to classify
    patients into early, mild, and severe categories with the goal of
    improving early intervention outcomes. The framework will be open
    sourced and made available to medical researchers worldwide.
    """

    result = run_novus_pipeline(test_proposal)

    with open("pipeline_output.json", "w") as f:
        json.dump(result, f, indent=2)

    print("[NOVUS] Results saved to pipeline_output.json")