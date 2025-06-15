from typing import List, Optional, Dict, Any, Callable


class EvidenceCollector:
    """
    Collects and verifies evidence for a given claim, allowing for an injectable
    search tool function for sourcing evidence.
    """

    @staticmethod
    def default_mock_search_tool(claim: str, num_to_find: int) -> List[str]:
        """Default mock search tool if no external one is provided."""
        # This mock will return exactly num_to_find sources.
        # A more advanced mock could have fixed results or simulate failure.
        print(f"Default mock search: Finding up to {num_to_find} sources for '{claim}'")
        claim_slug = claim.replace(" ", "_").replace("'", "")
        return [
            f"http://mock-default.com/claim/{claim_slug}/source_{i + 1}"
            for i in range(num_to_find)
        ]

    def __init__(
        self,
        min_citations_per_claim: int = 3,
        search_tool: Optional[Callable[[str, int], List[str]]] = None,
    ):
        """
        Initializes the EvidenceCollector.

        Args:
            min_citations_per_claim: Minimum citations for a claim.
            search_tool: Callable taking (claim: str, num_to_find: int)
                         and returning List[str] (source URLs). Defaults to
                         default_mock_search_tool.
        """
        if min_citations_per_claim < 1:
            raise ValueError("min_citations_per_claim must be at least 1.")
        self.min_citations_per_claim = min_citations_per_claim
        # search_tool manages how many results are returned.
        self.search_tool = (
            search_tool if search_tool is not None else self.default_mock_search_tool
        )

    def collect_and_verify_evidence(
        self, claim: str, existing_sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Collects and verifies evidence for a given claim.

        Args:
            claim: The statement/claim that needs supporting evidence.
            existing_sources: List of URLs or source identifiers already provided
                              for the claim.

        Returns:
            A dictionary with claim, citation counts, source lists, and status.
        """
        if existing_sources is None:
            existing_sources = []

        provided_citations_count = len(existing_sources)
        found_sources: List[str] = []
        status: str = ""

        if provided_citations_count < self.min_citations_per_claim:
            sources_needed = self.min_citations_per_claim - provided_citations_count

            print(
                f"Claim '{claim}': Needs {sources_needed} more source(s). "
                f"Using search tool..."
            )
            # Call the configured search tool.
            # The search tool is responsible for how many sources it actually returns.
            # It's given `sources_needed` as a hint of the maximum useful number.
            found_sources = self.search_tool(claim, sources_needed)

            print(f"Search tool returned {len(found_sources)} new source(s).")

            current_total_sources = provided_citations_count + len(found_sources)
            if current_total_sources >= self.min_citations_per_claim:
                status = "Sufficient (new sources found)"  # More generic status
            else:
                status = "Insufficient (even after search)"
        else:
            print(f"Claim '{claim}': Sufficient sources provided initially.")
            status = "Sufficient (initial sources)"

        all_sources = existing_sources + found_sources

        notes = "Evidence collection process completed."
        if self.search_tool == self.default_mock_search_tool:  # Check for default
            notes += " Used default mock search tool."
        # Check for other mock tools by naming convention for detailed notes
        elif (
            hasattr(self.search_tool, "__name__")
            and "mock" in self.search_tool.__name__.lower()
        ):
            notes += f" Used mock search tool: {self.search_tool.__name__}."
        else:
            notes += " Used custom search tool."

        return {
            "claim": claim,
            "required_citations": self.min_citations_per_claim,
            "provided_citations_count": provided_citations_count,
            "search_tool_provided_count": len(found_sources),  # Renamed field
            "all_sources": all_sources,
            "status": status,
            "notes": notes,
        }

    # Placeholders for future helper methods can remain if desired,
    # but the core search logic is now through self.search_tool.
    # def _search_for_sources(self, claim: str, num_needed: int) -> List[str]:
    #     """Simulates searching for sources using a tool like AutoGPT."""
    #     # This would trigger an external process or API call.
    #     print(f"Searching for {num_needed} sources for claim: {claim}")
    #     # Mock response
    #     return [f"http://searched-source.com/source{i+1}" for i in range(num_needed)]

    # def _verify_source_relevance(self, claim: str, source_url: str) -> bool:
    #     """Simulates verifying source relevance using RAG or other validation."""
    #     # This might fetch content and use an LLM to assess relevance.
    #     print(f"Verifying relevance of {source_url} for claim: {claim}")
    #     return True # Mock response

    # def _save_source_to_docs(
    #    self, source_content: str, claim_id: str, source_id: str
    # ):
    #     """Saves verified source content to the docs/ directory."""
    #     # This would involve file I/O.
    #     pass


if __name__ == "__main__":
    collector_min_2 = EvidenceCollector(min_citations_per_claim=2)
    collector_min_3 = EvidenceCollector(min_citations_per_claim=3)

    print("--- Scenario 1: Not enough sources, mock search helps ---")
    result1 = collector_min_3.collect_and_verify_evidence(
        claim="LLMs are good at math.", existing_sources=["http://example.com/source1"]
    )
    print(f"Result 1: {result1}\n")

    print("--- Scenario 2: Enough sources initially ---")
    result2 = collector_min_2.collect_and_verify_evidence(
        claim="Python is a popular language.",
        existing_sources=[
            "http://python.org",
            "http://stackoverflow.com/questions/tagged/python",
        ],
    )
    print(f"Result 2: {result2}\n")

    # Scenario 3: Using an injected mock search tool that finds limited results
    def limited_mock_search(claim: str, num_to_find: int) -> List[str]:
        print(f"Limited mock search: Max 1 source for '{claim}' (req: {num_to_find})")
        claim_slug = claim.replace(" ", "_").replace("'", "")
        base_url = "http://limitedmock.com/claim"
        return [f"{base_url}/{claim_slug}/source_1"] if num_to_find > 0 else []

    collector_limited_mock = EvidenceCollector(
        min_citations_per_claim=3, search_tool=limited_mock_search
    )
    print(
        "--- Scenario 3: Insufficient sources, injected limited mock search "
        "also not enough ---"
    )
    # Needs 3, existing 0, limited mock provides 1
    result3 = collector_limited_mock.collect_and_verify_evidence(
        claim="AI will achieve AGI soon."
    )
    print(f"Result 3: {result3}\n")

    # Scenario 4: Default mock search behavior (will find sources_needed)
    collector_min_5_default_mock = EvidenceCollector(min_citations_per_claim=5)
    print(
        "--- Scenario 4: Insufficient sources, default mock search "
        "helps (fills exactly) ---"
    )
    result4 = collector_min_5_default_mock.collect_and_verify_evidence(
        claim="Quantum computing is commercially viable for all businesses.",
        # Needs 4, default mock provides 4
        existing_sources=["http://example.com/quantum_intro"],
    )
    print(f"Result 4: {result4}\n")

    # Scenario 5: Zero existing sources, relies on default mock search
    print("--- Scenario 5: Zero existing sources, relies on default mock search ---")
    result5 = collector_min_2.collect_and_verify_evidence(
        claim="Cold fusion is real."
    )  # Needs 2, default mock provides 2
    print(f"Result 5: {result5}\n")

    try:
        EvidenceCollector(min_citations_per_claim=0)
    except ValueError as e:
        print("--- Scenario 6: Invalid init ---")
        print(f"Successfully caught error for invalid min_citations: {e}")
