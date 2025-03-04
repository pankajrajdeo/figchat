import json
from langchain_community.tools import DuckDuckGoSearchResults

async def Web_Search(query: str, num_results: int = 5) -> list:
    """
    Performs an internet search using DuckDuckGo and retrieves detailed results in JSON format asynchronously.

    Parameters:
        query (str): The search query to perform.
        num_results (int): The number of results to fetch (default is 5).

    Returns:
        list: A list of dictionaries containing detailed search results including title, snippet, and link.
    """
    try:
        # Initialize the DuckDuckGo Search Tool
        search_tool_instance = DuckDuckGoSearchResults(output_format="json")
        # Perform the search query asynchronously
        search_results = await search_tool_instance.ainvoke(query)
        # Parse the search results into a Python object if needed
        if isinstance(search_results, str):
            search_results = json.loads(search_results)
        # Limit the results to the specified number
        detailed_results = search_results[:num_results]
        return detailed_results
    except Exception as e:
        raise RuntimeError(f"Error performing search: {repr(e)}")

# For backward compatibility
internet_search_tool = Web_Search