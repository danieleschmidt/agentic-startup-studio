"""
Utility to load YAML-defined founders / investors into lightweight Agent
objects that expose a uniform `.run(prompt|doc)` API.

During Milestone-1 we keep it minimal: each agent delegates to one of
two toy LLM wrappers (`OpenAIWrapper` or `GeminiWrapper`).  Swap these
for full CrewAI / LangChain agents later.
"""
from pathlib import Path
import yaml, os, textwrap, random
from types import SimpleNamespace
from dotenv import load_dotenv

load_dotenv()

# --- naïve LLM shims ---------------------------------------------------- #
class BaseWrapper:
    def __init__(self, model, name): self.model, self.name = model, name
    def run(self, x):
        # placeholder deterministic answer so graphs compile offline
        seed = hash((self.name, str(x))) & 0xFFFF
        random.seed(seed)
        if isinstance(x, str):
            return f"[{self.name}] draft-response → {x[:60]}..."
        if isinstance(x, dict):
            return "\n".join(f"- {k}: {v}" for k, v in x.items())
        return str(x)

class OpenAIWrapper(BaseWrapper): pass
class GeminiWrapper(BaseWrapper): pass

MODEL_MAP = {
    "openai": OpenAIWrapper,
    "google": GeminiWrapper,
}

# --- public helpers ----------------------------------------------------- #
def _load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)

def _to_wrapper(dct):
    provider, model = dct["llm"].split(":")
    wrapper_cls = MODEL_MAP[provider]
    return wrapper_cls(model, dct["name"])

def load_all_agents(root="agents"):
    """Return {'ceo': Agent, 'vc': Agent, ...}"""
    agent_dir = Path(root)
    out = {}
    for yml in agent_dir.rglob("*.yaml"):
        cfg = _load_yaml(yml)
        agent = _to_wrapper(cfg)
        out[cfg["name"].lower()] = agent
    return out

def load_agent(name: str, root: str = "agents"):
    """Load a single agent configuration by name.

    Parameters
    ----------
    name:
        Name of the agent (case-insensitive, matches the ``name`` field in YAML).
    root:
        Root directory containing agent YAML files. Defaults to ``"agents"``.

    Returns
    -------
    object | None
        The loaded agent wrapper instance or ``None`` if not found.
    """
    name = name.lower()
    agents = load_all_agents(root)
    return agents.get(name)
