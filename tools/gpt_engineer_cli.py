import subprocess, tempfile, shutil, pathlib, json, os, uuid, time
from tools.cost_tracker import counter, TOKENS_SPENT

def run(spec_file: str, idea_id: str):
    work = pathlib.Path(tempfile.mkdtemp(prefix="gpteng_"))
    shutil.copy(spec_file, work/"prompt.md")

    tic = time.time()
    cp = subprocess.run(["gpt-engineer", "generate", "prompt.md"], cwd=work, text=True)
    toc = time.time()

    if cp.returncode:
        raise RuntimeError(cp.stderr)

    # count tokens (via gpt-engineer log JSON)
    usage_path = work/"generated"/"gpt-engineer-token-usage.json"
    if usage_path.exists():
        TOKENS_SPENT.inc(json.loads(usage_path.read_text())["total_tokens"])

    repo = pathlib.Path(f"mvp/{idea_id}")
    repo.mkdir(parents=True, exist_ok=True)
    shutil.move(str(work/"generated"), repo)

    with counter("gpt_engineer_seconds_total"):
        pass  # duration emitted

    return f"✅ MVP ready in {repo}  ({toc-tic:.1f}s)"
