import requests, os, json

BASE = os.getenv("DITTOFEED_URL", "http://localhost:9000")
TOK  = os.getenv("DITTOFEED_TOKEN")

def run(journey_name, steps, audience="all-users"):
    payload = {
        "name": journey_name,
        "audience": audience,
        "steps": steps
    }
    r = requests.post(f"{BASE}/api/v1/journeys", headers={"Authorization": f"Bearer {TOK}"}, json=payload)
    r.raise_for_status()
    return r.json()["id"]   # Dittofeed journey docs :contentReference[oaicite:2]{index=2}
