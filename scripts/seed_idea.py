import json
import uuid
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: seed_idea.py 'Idea description'")
    sys.exit(1)

idea_desc = sys.argv[1]
idea = {
    "id": str(uuid.uuid4()),
    "description": idea_desc
}

seed_file = Path('db/seeded_ideas.json')
if seed_file.exists():
    data = json.loads(seed_file.read_text())
    if not isinstance(data, list):
        data = []
else:
    data = []

data.append(idea)
seed_file.write_text(json.dumps(data, indent=2))
print(f"Seeded idea {idea['id']}: {idea_desc}")
