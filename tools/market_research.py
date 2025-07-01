from tools.browser_search import BrowserSearch

class MarketResearch:
    """
    A tool for performing market research using web search.
    """

    def __init__(self):
        self.browser_search = BrowserSearch()

    def search_market_trends(self, query: str) -> str:
        """
        Searches for market trends related to the given query.
        """
        search_query = f"latest market trends in {query}"
        return self.browser_search.search(search_query)

    def search_competitors(self, query: str) -> str:
        """
        Searches for competitors related to the given query.
        """
        search_query = f"competitors for {query}"
        return self.browser_search.search(search_query)

    def search_customer_demographics(self, query: str) -> str:
        """
        Searches for customer demographics related to the given query.
        """
        search_query = f"customer demographics for {query}"
        return self.browser_search.search(search_query)
