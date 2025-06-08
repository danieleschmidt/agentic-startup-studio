"""
Simple per-run token accounting.
Later, export metrics to Prometheus → Grafana (dashboard 21345). :contentReference[oaicite:5]{index=5}
"""
import contextlib, time, json, pathlib, os
LOG = pathlib.Path("artifacts/cost.jsonl")

@contextlib.contextmanager
def track(run_name):
    start = time.time(); tokens = 0
    yield lambda t: locals().update(tokens=t)   # setter fn
    elapsed = round(time.time()-start,2)
    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text(LOG.read_text() if LOG.exists() else "")
    LOG.write_text(f"{json.dumps({'run':run_name,'sec':elapsed,'tokens':tokens})}\n", append=True)
