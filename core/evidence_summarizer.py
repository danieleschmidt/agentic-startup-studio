# core/evidence_summarizer.py
from typing import List, Dict, Any


def summarize_evidence(
    evidence_items: List[Dict[str, Any]], summary_length: str = "medium"
) -> str:
    """
    Simulates summarizing a list of evidence items.
    In a real scenario, this would involve NLP summarization techniques.

    Args:
        evidence_items: A list of dictionaries, where each dictionary represents
                        an evidence item (e.g., from EvidenceCollector,
                        containing keys like 'source_url', 'content_snippet', etc.).
        summary_length: Desired length of the summary ("short", "medium", "long").
                        This is a conceptual parameter for the mock.

    Returns:
        A mock summary string.
    """
    if not isinstance(evidence_items, list):
        return "Error: evidence_items must be a list."

    if not evidence_items:
        return "No evidence items provided to summarize."

    num_items = len(evidence_items)

    # Prioritize 'source_url', then 'url', then 'unknown_source'
    source_urls: List[str] = []
    for item in evidence_items:
        if isinstance(item, dict):
            url = item.get("source_url", item.get("url", "unknown_source"))
            source_urls.append(url)
        else:
            source_urls.append("invalid_item_format")  # Handle non-dict items

    # Basic content check (conceptual)
    content_snippets_present = any(
        isinstance(item, dict) and item.get("content_snippet")
        for item in evidence_items
    )

    summary_intro = f"Mock {summary_length} summary of {num_items} evidence item(s)."

    sources_to_list = source_urls[:3]  # Get up to the first 3 sources
    sources_string = "Sources mentioned"
    if not sources_to_list:
        sources_string = "No specific sources identifiable"
    elif len(sources_to_list) == 1:
        sources_string += f" (1 source): {sources_to_list[0]}."
    else:
        sources_string += (
            f" (first {len(sources_to_list)}): {', '.join(sources_to_list)}."
        )

    content_status = (
        "Content snippets were present in at least one item."
        if content_snippets_present
        else "Content snippets were not consistently present."
    )

    summary = f"{summary_intro} {sources_string} {content_status}"

    print(f"Evidence summarizer called. Generated summary for {num_items} items.")
    return summary


if __name__ == "__main__":
    sample_evidence_list = [
        {
            "source_url": "http://example.com/source1",
            "content_snippet": "Lorem ipsum...",
        },
        {"url": "http://example.com/source2", "title": "Another Finding"},
        {
            "source_url": "http://example.com/source3",
            "content_snippet": "Dolor sit amet...",
        },
        {"source_url": "http://example.com/source4"},  # Not listed by URL in summary
        {"some_other_key": "value"},  # No URL
    ]
    empty_evidence_list: List[Dict[str, Any]] = []
    invalid_evidence_input: Any = "not a list of dicts"

    print("\n--- Testing with sample evidence (medium) ---")
    print(summarize_evidence(sample_evidence_list))

    print("\n--- Testing with sample evidence (short) ---")
    print(summarize_evidence(sample_evidence_list, summary_length="short"))

    print("\n--- Testing with empty evidence ---")
    print(summarize_evidence(empty_evidence_list))

    print("\n--- Testing with invalid evidence type ---")
    print(summarize_evidence(invalid_evidence_input))  # type: ignore

    print("\n--- Testing with items missing url/source_url ---")
    evidence_no_urls = [{"title": "Title only"}, {"content": "Content only"}]
    print(summarize_evidence(evidence_no_urls))

    print("\n--- Testing with non-dict items in list ---")
    evidence_mixed_types = [{"source_url": "url1"}, "not a dict", {"url": "url2"}]
    print(summarize_evidence(evidence_mixed_types))  # type: ignore
