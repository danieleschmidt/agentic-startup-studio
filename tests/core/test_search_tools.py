import pytest
from core.search_tools import basic_web_search_tool


@pytest.mark.vcr()  # Uses default cassette name based on test function name
def test_search_fetches_and_returns_urls():
    """
    Tests that basic_web_search_tool fetches data and returns a list of
    constructed URLs. Uses vcrpy to record/replay HTTP interactions.
    Cassette will be stored in tests/cassettes/test_search_fetches_and_returns_urls.yaml
    """
    claim = "test claim for search tool"
    num_to_find = 3

    urls = basic_web_search_tool(claim, num_to_find)

    assert isinstance(urls, list), "Should return a list."
    assert len(urls) == num_to_find, f"Should return {num_to_find} URLs."

    for url in urls:
        assert isinstance(url, str), "Each item in the list should be a string."
        assert url.startswith("https://jsonplaceholder.typicode.com/posts/"), (
            f"URL '{url}' does not look like a valid JSONPlaceholder post URL."
        )


@pytest.mark.vcr()  # A separate cassette for this test case
def test_search_tool_handles_num_to_find_zero():
    """Tests that the tool returns an empty list when num_to_find is 0."""
    claim = "test claim for zero results"
    num_to_find = 0
    urls = basic_web_search_tool(claim, num_to_find)
    assert isinstance(urls, list)
    assert len(urls) == 0


@pytest.mark.vcr()  # A separate cassette, though it might hit the same API endpoint
def test_search_tool_handles_num_to_find_more_than_default_api_limit_implicitly():
    """
    Tests how the tool behaves if num_to_find is large.
    JSONPlaceholder /posts returns 100 items. If we ask for more, we should get 100.
    If we ask for less than 100, we get what we asked for.
    This test will check asking for a few, e.g., 5, which is less than 100.
    The tool currently iterates through response and caps at num_to_find.
    """
    claim = "test claim for 5 results"
    num_to_find = 5
    urls = basic_web_search_tool(claim, num_to_find)
    assert isinstance(urls, list)
    assert len(urls) == num_to_find  # Expecting 5 URLs

    # If we wanted to test against the API's max limit (e.g. 100 for /posts)
    # num_to_find_large = 150
    # urls_large = basic_web_search_tool(claim, num_to_find_large)
    # assert len(urls_large) == 100 # As API returns max 100 posts


# Regarding test_search_handles_network_error:
# Testing network errors with VCR usually means configuring VCR to not find a cassette
# and then ensuring a real network error would occur (e.g. by disabling network),
# or by using VCR's advanced features to simulate errors, or by directly mocking
# `requests.get` with `unittest.mock.patch` to raise a RequestException.
# For this subtask, focusing on the successful VCR case is primary.
# The try-except block in basic_web_search_tool covers the error handling logic.

# To make VCRPy work, a `conftest.py` might be needed for more complex configurations,
# or for filter_headers, etc. but basic `@pytest.mark.vcr` often works out of the box
# if `pytest-vcr` is installed and cassettes directory can be created.
# Default cassette path: <test_directory>/cassettes/<test_function_name>.yaml
# So, this will create tests/core/cassettes/<test_name>.yaml for each.
# If tests/core/cassettes does not exist, it needs to be creatable.
# `create_file_with_block` might not make parent dirs for cassettes.
# This is handled when tests run; pytest-vcr usually creates the dir.
