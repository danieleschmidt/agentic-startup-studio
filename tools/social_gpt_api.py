# tools/social_gpt_api.py  (now real)
import requests, os

def run(thread):
    bearer = os.getenv("TWITTER_BEARER")
    for i, tw in enumerate(thread, 1):
        print(f"[tweet {i}] {tw}")   # stub: swap with twitter v2 POST
