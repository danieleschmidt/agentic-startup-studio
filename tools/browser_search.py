from duckduckgo_search import DDGS
def run(query, k=5):
    """Return top-k DuckDuckGo snippet tuples (title, url, snippet)."""
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=k))
