"""
Dittofeed journey creator.  Dittofeed channels & docs: :contentReference[oaicite:4]{index=4}
"""
import uuid, json, pathlib
def run(journey_name, steps):
    jid = str(uuid.uuid4())
    fp = pathlib.Path(f"journeys/{jid}.json")
    fp.parent.mkdir(exist_ok=True)
    fp.write_text(json.dumps({"name": journey_name, "steps": steps}, indent=2))
    return f"Journey saved → {fp}"
