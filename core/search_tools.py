import requests
import asyncio
from typing import List, Dict, Any


def basic_web_search_tool(claim: str, num_to_find: int) -> List[str]:
    """
    A basic web search tool that fetches data from JSONPlaceholder
    and returns a list of URLs constructed from the fetched posts.

    Args:
        claim: The claim or query to search for (currently unused by this mock tool).
        num_to_find: The desired number of source URLs to find.

    Returns:
        A list of constructed URLs, or an empty list if an error occurs.
    """
    # The 'claim' argument is unused in this basic version, as JSONPlaceholder
    # doesn't support arbitrary queries. In a real search tool, it would be used.
    print(
        f"Basic web search tool activated for claim: '{claim}', "
        f"finding up to {num_to_find} items."
    )

    api_url = "https://jsonplaceholder.typicode.com/posts"
    found_urls: List[str] = []

    try:
        response = requests.get(api_url, timeout=10)  # Added timeout
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        posts = response.json()

        for i, post in enumerate(posts):
            if i >= num_to_find:
                break
            if "id" in post:
                # Construct a URL for each post
                found_urls.append(
                    f"https://jsonplaceholder.typicode.com/posts/{post['id']}"
                )
            else:
                # Fallback if 'id' is not in post, though unlikely for this API
                found_urls.append(
                    f"https://jsonplaceholder.typicode.com/mock_post_url_{i + 1}"
                )

    except requests.exceptions.RequestException as e:
        print(f"Error during web search: {e}")
        return []  # Return empty list on error
    except ValueError as e:  # Handles JSON decoding errors
        print(f"Error decoding JSON response: {e}")
        return []

    return found_urls


async def search_for_evidence(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Async evidence search function that returns structured evidence data.
    
    Args:
        query: The search query/claim to find evidence for
        num_results: Number of evidence items to return
        
    Returns:
        List of evidence dictionaries with url, title, and snippet
    """
    # Simulate async operation
    await asyncio.sleep(0.1)
    
    # Use existing sync search function and transform results
    urls = basic_web_search_tool(query, num_results)
    
    evidence = []
    for i, url in enumerate(urls):
        evidence.append({
            'url': url,
            'title': f'Evidence {i+1} for: {query[:50]}...',
            'snippet': f'This is evidence content related to {query}. Supporting data and analysis.',
            'relevance_score': max(0.7, 1.0 - (i * 0.1))  # Decreasing relevance
        })
    
    return evidence


if __name__ == "__main__":
    print("--- Testing basic_web_search_tool ---")

    claim1 = "Test claim for 3 URLs"
    urls1 = basic_web_search_tool(claim1, 3)
    print(f"\nFor claim: '{claim1}', found {len(urls1)} URLs:")
    for url in urls1:
        print(url)

    claim2 = "Test claim for 0 URLs"
    urls2 = basic_web_search_tool(claim2, 0)
    print(f"\nFor claim: '{claim2}', found {len(urls2)} URLs:")
    for url in urls2:
        print(url)

    claim3 = "Test claim for 10 URLs (API might return fewer initially)"
    urls3 = basic_web_search_tool(
        claim3, 10
    )  # JSONPlaceholder /posts returns 100 items
    print(f"\nFor claim: '{claim3}', found {len(urls3)} URLs:")
    for url in urls3:
        print(url)

    # Simulate a non-existent domain for error handling (won't work with VCR here).
    # Proper testing for this needs mocking 'requests.get' or specific VCR setup.
    # This __main__ block shows try-except if run directly without VCR & bad URL.
    # For this demo, one would temporarily change api_url to something invalid.
    # e.g. api_url = "https://nonexistentdomain123abc.com/api"
    print("\n--- Demo: (Manually simulated) bad URL for error path ---")
    # This part is for direct execution demo of the error path.
    # Actual network error tests are better controlled in pytest.
    # Modifying the function's api_url here isn't ideal for this demo.
    print("Error path testing in unit tests is more robust (e.g. with mocking).")
