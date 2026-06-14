import os
import yaml
import requests
from dotenv import load_dotenv

load_dotenv()

BAND_BASE_URL = "https://app.band.ai"

# Agent pipeline order for mentions
AGENT_ORDER = ["intake", "duplication", "relevance", "eligibility", "proposal", "compliance"]

def load_agent_config():
    config_path = os.path.join(os.path.dirname(__file__), "agent_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_next_agent(agent_name: str, config: dict) -> tuple:
    """Get the next agent in the pipeline."""
    try:
        idx = AGENT_ORDER.index(agent_name)
        if idx < len(AGENT_ORDER) - 1:
            next_name = AGENT_ORDER[idx + 1]
            next_config = config.get(next_name, {})
            return next_name, next_config.get("agent_id")
    except ValueError:
        pass
    return None, None

def notify_band_agent(agent_name: str, stage: str, input_summary: str, output_summary: str):
    """Notify Band when an agent completes its task and passes to next agent."""
    try:
        config = load_agent_config()
        agent_config = config.get(agent_name, {})
        api_key = agent_config.get("api_key")
        agent_id = agent_config.get("agent_id")

        if not api_key or not agent_id:
            print(f"[BAND] Missing config for {agent_name}")
            return False

        session = requests.Session()
        session.headers.update({
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "Novus-Research-System/1.0"
        })

        # Step 1: Create chat room
        room_response = session.post(
            f"{BAND_BASE_URL}/api/v1/agent/chats",
            json={"chat": {}},
            timeout=15,
            verify=True
        )

        if room_response.status_code not in [200, 201]:
            print(f"[BAND] Room error {room_response.status_code}: {room_response.text[:200]}")
            return False

        room_id = room_response.json().get("data", {}).get("id")
        print(f"[BAND] Room created: {room_id}")

        # Step 2: Get next agent and add as participant
        next_name, next_agent_id = get_next_agent(agent_name, config)

        if next_agent_id:
            participant_response = session.post(
                f"{BAND_BASE_URL}/api/v1/agent/chats/{room_id}/participants",
                json={"participant": {"agent_id": next_agent_id}},
                timeout=15,
                verify=True
            )
            print(f"[BAND] Participant added: {participant_response.status_code}")

            # Step 3: Send message mentioning next agent
            message_text = f"[NOVUS - {stage.upper()}] Stage complete. Passing context to {next_name}. Output: {output_summary}"
            msg_response = session.post(
                f"{BAND_BASE_URL}/api/v1/agent/chats/{room_id}/messages",
                json={
                    "message": {
                        "content": message_text,
                        "mentions": [{"id": next_agent_id}]
                    }
                },
                timeout=15,
                verify=True
            )
            print(f"[BAND] Message status: {msg_response.status_code}")
            if msg_response.status_code not in [200, 201]:
                print(f"[BAND] Message error: {msg_response.text[:200]}")
        else:
            # Last agent — just log completion
            print(f"[BAND] Final stage complete: {agent_name}")

        print(f"[BAND] Notified {agent_name} successfully")
        return True

    except Exception as e:
        print(f"[BAND] Error: {e}")
        return False
