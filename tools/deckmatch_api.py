import random
def run(deck_markdown):
    """Return deterministic mock score 0-1."""
    random.seed(hash(deck_markdown) & 0xFFFF)
    return round(0.6 + random.random() * 0.4, 2)
