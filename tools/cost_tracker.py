from prometheus_client import Counter, Summary, start_http_server
import time, contextlib

start_http_server(9102)  # default exporter port

TOKENS_SPENT = Counter("llm_tokens_total", "Total LLM tokens spent")
counter = Summary
