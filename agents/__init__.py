from importlib.metadata import version, PackageNotFoundError
from .loader import load_all_agents, load_agent

__all__ = ["load_all_agents", "load_agent"]
try:  # Optional: CrewAI may not be installed in all dev environments
    __version__ = version("crewai")
except PackageNotFoundError:  # pragma: no cover - version lookup is trivial
    __version__ = "unknown"
