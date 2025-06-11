# tests/core/test_evidence_summarizer.py
import pytest
from core.evidence_summarizer import summarize_evidence
from typing import List, Dict, Any


def test_summarize_evidence_valid_input():
    evidence = [
        {"source_url": "url1", "content_snippet": "text1"},
        {"url": "url2"},
    ]
    summary = summarize_evidence(evidence, summary_length="medium")
    assert isinstance(summary, str)
    assert "Mock medium summary of 2 evidence item(s)" in summary
    assert "url1" in summary
    assert "url2" in summary
    # Based on current logic, if any item has 'content_snippet', it's "present"
    assert "Content snippets were present in at least one item." in summary


def test_summarize_evidence_different_lengths():
    evidence = [{"source_url": "url1"}]  # No content_snippet here
    short_summary = summarize_evidence(evidence, summary_length="short")
    assert "Mock short summary" in short_summary
    assert "Content snippets were not consistently present." in short_summary

    long_summary = summarize_evidence(evidence, summary_length="long")
    assert "Mock long summary" in long_summary
    assert "Content snippets were not consistently present." in long_summary


def test_summarize_evidence_empty_list():
    empty_evidence: List[Dict[str, Any]] = []
    summary = summarize_evidence(empty_evidence)
    assert summary == "No evidence items provided to summarize."


def test_summarize_evidence_invalid_input_type():
    invalid_evidence: Any = "this is not a list"
    summary = summarize_evidence(invalid_evidence)  # type: ignore
    assert summary == "Error: evidence_items must be a list."


def test_summarize_evidence_no_content_snippets():
    evidence = [
        {"source_url": "url1", "title": "Finding A"},
        {"url": "url2", "notes": "Some notes"},
    ]
    summary = summarize_evidence(evidence, summary_length="medium")
    assert "Content snippets were not consistently present." in summary


def test_summarize_evidence_source_url_priority():
    evidence = [
        {"url": "url_fallback", "source_url": "url_priority"},
    ]
    summary = summarize_evidence(evidence)
    assert "url_priority" in summary
    assert "url_fallback" not in summary  # Check that source_url is preferred


def test_summarize_evidence_only_url_key():
    evidence = [
        {"url": "only_url_key"},
    ]
    summary = summarize_evidence(evidence)
    assert "only_url_key" in summary


def test_summarize_evidence_max_three_sources_in_summary():
    evidence = [
        {"source_url": "url1"},
        {"source_url": "url2"},
        {"source_url": "url3"},
        {"source_url": "url4"},
    ]
    summary = summarize_evidence(evidence)
    assert "url1, url2, url3" in summary
    assert "url4" not in summary  # Checks that only first 3 are listed


def test_summarize_evidence_item_not_dict():
    evidence = [{"source_url": "url1"}, "not_a_dict_item", {"url": "url2"}]
    summary = summarize_evidence(evidence)  # type: ignore
    assert "invalid_item_format" in summary
    assert "url1" in summary
    assert "url2" in summary
    assert "3 evidence item(s)" in summary  # Still counts the invalid item


def test_summarize_evidence_item_no_url_keys():
    evidence = [{"title": "Title only"}, {"notes": "Notes only"}]
    summary = summarize_evidence(evidence)
    assert "unknown_source, unknown_source" in summary  # Both items default
    assert "2 evidence item(s)" in summary
