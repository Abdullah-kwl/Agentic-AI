import os
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv(r"D:\Projects\Agentic-AI\.env")

mcp = FastMCP(name="WebSearchMCP")

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@mcp.tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs and snippets."""
    results = tavily.search(query=query, max_results=5)

    out = []
    for r in results["results"]:
        out.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
        )

    return "\n----\n".join(out)


@mcp.tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)


# fastmcp run web-search-mcp/web_search_mcp.py --transport http --port 9000 --reload
# OR
# python web-search-mcp/web_search_mcp.py
# fastmcp dev inspector web-search-mcp/web_search_mcp.py
