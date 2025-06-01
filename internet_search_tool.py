import os
import json
from langchain_tavily import TavilySearch

# Ensure the Tavily API key is set in environment variables
if not os.environ.get("TAVILY_API_KEY"):
    raise EnvironmentError("TAVILY_API_KEY environment variable not set")

# Initialize the Tavily Search tool
tavily_tool_instance = TavilySearch(
    max_results=5,
    topic="general"
)

async def Web_Search(query: str, num_results: int = 5) -> list:
    """
    Performs an internet search using Tavily and retrieves detailed results in JSON format asynchronously.

    Parameters:
        query (str): The search query to perform.
        num_results (int): The number of results to fetch (default is 5).

    Returns:
        list: A list of dictionaries containing detailed search results including title, content snippet, and URL.
    """
    try:
        # Invoke Tavily asynchronously with query
        search_results = await tavily_tool_instance.ainvoke({"query": query})

        # If the results are returned as JSON string, parse it
        if isinstance(search_results, str):
            search_results = json.loads(search_results)

        # Extract the 'results' field which contains the actual search entries
        detailed_results = search_results.get("results", [])[:num_results]
        return detailed_results

    except Exception as e:
        raise RuntimeError(f"Error performing Tavily search: {repr(e)}")

# For backward compatibility
internet_search_tool = Web_Search
