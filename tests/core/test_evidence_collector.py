import pytest
from core.evidence_collector import EvidenceCollector


class TestEvidenceCollectorInit:
    def test_default_min_citations(self):
        """Test default min_citations_per_claim."""
        collector = EvidenceCollector()
        assert collector.min_citations_per_claim == 3

    def test_custom_min_citations(self):
        """Test custom valid min_citations_per_claim."""
        collector = EvidenceCollector(min_citations_per_claim=5)
        assert collector.min_citations_per_claim == 5

    def test_min_citations_is_1(self):
        """Test initialization with min_citations_per_claim = 1."""
        collector = EvidenceCollector(min_citations_per_claim=1)
        assert collector.min_citations_per_claim == 1

    def test_invalid_min_citations_zero(self):
        """Test initialization with min_citations_per_claim = 0."""
        with pytest.raises(
            ValueError, match="min_citations_per_claim must be at least 1."
        ):
            EvidenceCollector(min_citations_per_claim=0)

    def test_invalid_min_citations_negative(self):
        """Test initialization with negative min_citations_per_claim."""
        with pytest.raises(
            ValueError, match="min_citations_per_claim must be at least 1."
        ):
            EvidenceCollector(min_citations_per_claim=-1)


class TestCollectAndVerifyEvidence:
    TEST_CLAIM = "This is a test claim."

    def test_sufficient_sources_initially(self):
        collector = EvidenceCollector(min_citations_per_claim=3)
        existing_sources = ["s1", "s2", "s3"]
        result = collector.collect_and_verify_evidence(
            self.TEST_CLAIM, existing_sources
        )

        assert result["status"] == "Sufficient (initial sources)"
        assert result["provided_citations_count"] == 3
        assert result["search_tool_provided_count"] == 0  # Key changed
        assert result["all_sources"] == existing_sources
        assert result["required_citations"] == 3
        assert result["claim"] == self.TEST_CLAIM
        assert result["claim"] == self.TEST_CLAIM
        assert "notes" in result

    def test_insufficient_default_mock_provides_enough(self):
        # Default mock search will provide exactly the number of sources needed.
        collector = EvidenceCollector(min_citations_per_claim=3)
        existing_sources = ["s1"]  # Needs 2 sources
        result = collector.collect_and_verify_evidence(
            self.TEST_CLAIM, existing_sources
        )

        # Status text changed from "Sufficient (mock sources added)"
        assert result["status"] == "Sufficient (new sources found)"
        assert result["provided_citations_count"] == 1
        # Needs 2, default mock is asked for 2, and provides 2.
        assert result["search_tool_provided_count"] == 2  # Key changed
        assert len(result["all_sources"]) == 3
        assert existing_sources[0] in result["all_sources"]
        assert "Used default mock search tool." in result["notes"]

    # Note: Tests for "insufficient even after search" or "mock finds fewer"
    # now require an *injected* custom mock search tool, as the default
    # mock tool always returns the number of sources requested (sources_needed).
    # These scenarios are tested below with custom mocks.

    def test_zero_initial_sources_default_mock_sufficient(self):
        # Default mock provides exactly the number needed.
        collector = EvidenceCollector(min_citations_per_claim=2)
        result = collector.collect_and_verify_evidence(self.TEST_CLAIM, [])  # Needs 2

        assert result["status"] == "Sufficient (new sources found)"
        assert result["provided_citations_count"] == 0
        # Needs 2, default mock is asked for 2, provides 2.
        assert result["search_tool_provided_count"] == 2
        assert len(result["all_sources"]) == 2

    def test_min_citations_1_sufficient_initial(self):
        collector = EvidenceCollector(min_citations_per_claim=1)
        result = collector.collect_and_verify_evidence(self.TEST_CLAIM, ["s1"])
        assert result["status"] == "Sufficient (initial sources)"
        assert result["provided_citations_count"] == 1
        assert result["search_tool_provided_count"] == 0

    def test_min_citations_1_needs_one_default_mock_finds_it(self):
        collector = EvidenceCollector(min_citations_per_claim=1)
        result = collector.collect_and_verify_evidence(self.TEST_CLAIM, [])  # Needs 1
        assert result["status"] == "Sufficient (new sources found)"
        assert result["provided_citations_count"] == 0
        # Needs 1, default mock asked for 1, provides 1.
        assert result["search_tool_provided_count"] == 1

    def test_all_dictionary_keys_present_with_default_mock(self):
        collector = EvidenceCollector()
        result = collector.collect_and_verify_evidence("Any claim")
        expected_keys = [
            "claim",
            "required_citations",
            "provided_citations_count",
            "search_tool_provided_count",
            "all_sources",
            "status",
            "notes",  # Key changed
        ]
        for key in expected_keys:
            assert key in result, f"Key '{key}' missing from result"
        assert "Used default mock search tool." in result["notes"]

    # --- New tests for injected custom mock search functions ---

    def custom_mock_search_success_enough(self, claim: str, num_to_find: int):
        # This mock successfully finds the number of items requested.
        claim_slug = claim.replace(" ", "_").replace("'", "")
        # Break long f-string line
        base_url = f"http://custom.com/{claim_slug}"
        return [f"{base_url}/src{i + 1}" for i in range(num_to_find)]

    def custom_mock_search_success_too_few(self, claim: str, num_to_find: int):
        # This mock finds fewer items than requested (e.g., only 1).
        if num_to_find == 0:
            return []  # Fixed E701 by moving to new line
        # Simulates finding only 1, regardless of num_to_find (if > 0)
        claim_slug = claim.replace(" ", "_").replace("'", "")
        return [f"http://custom-few.com/{claim_slug}/src1"]

    def custom_mock_search_none(self, claim: str, num_to_find: int):
        # This mock finds no items.
        return []

    def test_injected_search_provides_enough(self):
        collector = EvidenceCollector(
            min_citations_per_claim=3,
            search_tool=self.custom_mock_search_success_enough,
        )
        existing_sources = ["s1"]  # Needs 2
        result = collector.collect_and_verify_evidence(
            self.TEST_CLAIM, existing_sources
        )

        assert result["status"] == "Sufficient (new sources found)"
        assert result["provided_citations_count"] == 1
        # Custom tool asked for 2, found 2:
        assert result["search_tool_provided_count"] == 2
        assert len(result["all_sources"]) == 3
        assert "http://custom.com/" in result["all_sources"][1]
        expected_note = "Used mock search tool: custom_mock_search_success_enough."
        assert expected_note in result["notes"]

    def test_injected_search_provides_too_few(self):
        collector = EvidenceCollector(
            min_citations_per_claim=3,
            search_tool=self.custom_mock_search_success_too_few,
        )
        existing_sources = ["s1"]  # Needs 2
        result = collector.collect_and_verify_evidence(
            self.TEST_CLAIM, existing_sources
        )

        assert result["status"] == "Insufficient (even after search)"
        assert result["provided_citations_count"] == 1
        # Custom tool asked for 2, found 1:
        assert result["search_tool_provided_count"] == 1
        assert len(result["all_sources"]) == 2
        expected_note = "Used mock search tool: custom_mock_search_success_too_few."
        assert expected_note in result["notes"]

    def test_injected_search_provides_none(self):
        collector = EvidenceCollector(
            min_citations_per_claim=3, search_tool=self.custom_mock_search_none
        )
        existing_sources = ["s1"]  # Needs 2
        result = collector.collect_and_verify_evidence(
            self.TEST_CLAIM, existing_sources
        )

        assert result["status"] == "Insufficient (even after search)"
        assert result["provided_citations_count"] == 1
        # Custom tool asked for 2, found 0:
        assert result["search_tool_provided_count"] == 0
        assert len(result["all_sources"]) == 1
        assert "Used mock search tool: custom_mock_search_none." in result["notes"]

    def test_injected_search_zero_initial_sources(self):
        collector = EvidenceCollector(
            min_citations_per_claim=2,
            search_tool=self.custom_mock_search_success_enough,
        )
        result = collector.collect_and_verify_evidence(self.TEST_CLAIM, [])  # Needs 2

        assert result["status"] == "Sufficient (new sources found)"
        assert result["provided_citations_count"] == 0
        # Custom tool asked for 2, found 2:
        assert result["search_tool_provided_count"] == 2
        assert len(result["all_sources"]) == 2
        expected_note = "Used mock search tool: custom_mock_search_success_enough."
        assert expected_note in result["notes"]
