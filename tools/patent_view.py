def run(keyword):
    return [{"patent": f"US-{hash(keyword)%999999}", "title": f\"Patent on {keyword}\"}]
