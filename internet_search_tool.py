import os
import json
import ast
import asyncio
from langchain_community.utilities import GoogleSerperAPIWrapper

# 1. Ensure the Serper API key is set
if not os.getenv("SERPER_API_KEY"):
    raise EnvironmentError("SERPER_API_KEY environment variable not set")

# 2. Initialize Serper wrapper, fetching up to 5 results
serper = GoogleSerperAPIWrapper(k=5, type="search")

async def web_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Return up to *num_results* Google results for *query* via Serper.dev.
    Each result dict contains at least: title, snippet, link.
    """
    loop = asyncio.get_running_loop()
    raw = await loop.run_in_executor(None, serper.results, query)

    # If Serper returned a string, parse it into a dict
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = ast.literal_eval(raw)

    if not isinstance(raw, dict):
        raise TypeError(f"Serper returned unexpected type: {type(raw)}")

    return (raw.get("organic") or raw.get("results") or [])[:num_results]

# Alias for backward compatibility
internet_search_tool = web_search

# Demo: runs only if executed directly
if __name__ == "__main__":
    async def main():
        query = "Latest trends in AI 2025"
        results = await web_search(query)
        print(f"\nTop {len(results)} results for: \"{query}\"\n")
        for i, item in enumerate(results, 1):
            print(f"{i}. {item.get('title')}")
            print(f"   {item.get('snippet')}")
            print(f"   {item.get('link')}\n")

    asyncio.run(main())
