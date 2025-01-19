# internet_search_tool.py
import json
from langchain_community.tools import DuckDuckGoSearchResults

def internet_search_tool(query: str, num_results: int = 5) -> list:
    """
    Performs an internet search using DuckDuckGo and retrieves detailed results in JSON format.

    Parameters:
        query (str): The search query to perform.
        num_results (int): The number of results to fetch (default is 5).

    Returns:
        list: A list of dictionaries containing detailed search results including title, snippet, and link.
    """
    try:
        # Initialize the DuckDuckGo Search Tool
        search_tool_instance = DuckDuckGoSearchResults(output_format="json")

        # Perform the search query
        search_results = search_tool_instance.invoke(query)

        # Parse the search results into a Python object
        if isinstance(search_results, str):
            search_results = json.loads(search_results)

        # Limit the results to the specified number
        detailed_results = search_results[:num_results]

        return detailed_results
    except Exception as e:
        raise RuntimeError(f"Error performing search: {repr(e)}")
