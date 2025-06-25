import json
import re
from pathlib import Path
from typing import Dict


def parse_plan(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    # find first unchecked task
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("- [ ]"):
            task = line[5:].strip()
            if not task.lower().startswith("**no"):
                return task
    raise ValueError("No unchecked tasks found")


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")


EPIC = parse_plan(Path("DEVELOPMENT_PLAN.md"))
PLAIN_EPIC = re.sub(r"\*", "", EPIC)

sub_tasks = [
    f"Set up pytest and coverage infrastructure for {PLAIN_EPIC}",
    "Implement unit tests for Idea model and CLI.",
    "Write tests for ArchitectureAnalyzer module.",
    "Integrate coverage reports and ensure >=80% coverage.",
]

# Sprint board markdown
lines = [
    f"# Sprint Board: {EPIC}",
    "",
    "## Backlog",
    "| Task | Owner | Priority | Status |",
    "| --- | --- | --- | --- |",
]
for task in sub_tasks:
    lines.append(f"| {task} | @agent | P1 | Ready |")
Path("SPRINT_BOARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Acceptance criteria json
criteria: Dict[str, Dict] = {}
criteria[slugify(sub_tasks[0])] = {
    "test_file": f"tests/test_{slugify(sub_tasks[0])}.py",
    "cases": {
        "success": "pytest runs successfully and coverage data is produced",
        "edge": "Handles absence of .coveragerc gracefully",
    },
}
criteria[slugify(sub_tasks[1])] = {
    "test_file": f"tests/test_{slugify(sub_tasks[1])}.py",
    "cases": {
        "success": "Valid idea inputs pass validation and CLI returns success",
        "invalid": "Invalid inputs raise validation errors",
    },
}
criteria[slugify(sub_tasks[2])] = {
    "test_file": f"tests/test_{slugify(sub_tasks[2])}.py",
    "cases": {
        "success": "ArchitectureAnalyzer generates a markdown report",
        "edge": "Handles unparsable files without crashing",
    },
}
criteria[slugify(sub_tasks[3])] = {
    "test_file": f"tests/test_{slugify(sub_tasks[3])}.py",
    "cases": {
        "success": "Coverage percentage is at least 80%",
        "failing": "Fails the build if coverage drops below threshold",
    },
}

Path("tests/sprint_acceptance_criteria.json").write_text(
    json.dumps(criteria, indent=2), encoding="utf-8"
)
print("Sprint plan generated for:", EPIC)
