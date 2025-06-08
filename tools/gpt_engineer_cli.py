"""
Thin wrapper around GPT-Engineer’s `gpt-engineer generate`.
Repo: :contentReference[oaicite:2]{index=2}
"""
import subprocess, tempfile, shutil, pathlib, os, uuid

def run(spec_text):
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="gpteng_"))
    (workdir/"spec.md").write_text(spec_text)
    cmd = ["gpt-engineer", "generate", str(workdir/"spec.md")]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode:
        return f"❌ GPT-Engineer failed\n{cp.stderr}"
    repo_path = workdir / "generated"
    # push stub – replace with real Git commands
    dest = pathlib.Path(f"mvp/{uuid.uuid4()}")
    shutil.move(repo_path, dest)
    return f"✅ Repo scaffolded at {dest}"
