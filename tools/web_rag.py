def run(url):
    """Totally fake: fetches first 200 chars of a page (placeholder)."""
    import requests, re, textwrap
    html = requests.get(url, timeout=10).text
    txt = re.sub(r"<[^>]+>", " ", html)
    return textwrap.shorten(txt, width=200, placeholder="…")
