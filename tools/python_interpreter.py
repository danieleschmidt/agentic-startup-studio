def run(code: str):
    """Eval *safe* one-liners (demo only)."""
    allowed = {"sum", "len", "range"}
    if any(tok not in allowed for tok in code.split()):
        return "⚠️ Unsafe code blocked"
    return str(eval(code))
