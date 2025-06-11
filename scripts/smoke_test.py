"""
Render landing page → serve with Python’s http.server → open browser.
Later you’ll swap serve_static() with Fly.io deploy.  Fly.io docs: :contentReference[oaicite:1]{index=1}
"""
import http.server, socketserver, webbrowser, pathlib, argparse, uuid
from jinja2 import Template

def render(idea_name, tagline):
    tid = str(uuid.uuid4())
    tpl = Template(pathlib.Path("templates/landing.html.j2").read_text())
    html = tpl.render(
        idea_name=idea_name,
        idea_tagline=tagline,
        idea_id=tid,
        posthog_key="phc_demo_123",           # env var later
        posthog_host="http://localhost:8000", # self-host
    )
    out = pathlib.Path(f"static/{tid}.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    return out

def serve_static(path):
    PORT = 8888
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🚀 serving on http://localhost:{PORT}/{path.name}")
        webbrowser.open(f"http://localhost:{PORT}/{path.name}")
        httpd.serve_forever()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("idea_name"), ap.add_argument("tagline")
    args = ap.parse_args()
    fp = render(args.idea_name, args.tagline)
    serve_static(fp)
