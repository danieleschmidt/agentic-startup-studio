"""
Stub for Social-GPT project: :contentReference[oaicite:3]{index=3}
"""
def run(prompt_sched):
    return [f"(tweet-{i}) {prompt_sched[:40]}…" for i in range(1,8)]
