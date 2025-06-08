from agents.loader import load_all_agents
engineer = load_all_agents()["engineer"]

def run_build(idea_id, spec):
    print(engineer.run(spec))

if __name__ == "__main__":
    run_build("demo-123", "# Spec\nBuild a Flask API that echoes.")
