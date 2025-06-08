import json, pathlib
db = pathlib.Path("ideas/ledger.json")
approved = json.loads(db.read_text())  # from Milestone 1 recorder
for idea in approved:
    yes = input(f"🌟 Build {idea['name']}? [y/N] ")
    if yes.lower() == "y":
        print("→ queued for build-loop")
