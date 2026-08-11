import os
import json
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
API_KEY = os.getenv("SERP_API_KEY")


@tool
def web_search(query: str) -> str:
    """Search Google for current information. Returns a direct answer if available,
    otherwise returns top result snippets for synthesis."""
    response = requests.get(
        "https://serpapi.com/search",
        params={
            "engine": "google",
            "q": query,
            "api_key": API_KEY,
            "hl": "en"
        }
    )
    data = response.json()

    if "answer_box" in data:
        return f"[ANSWER_BOX]\n{json.dumps(data['answer_box'], indent=2)}"

    if "knowledge_graph" in data:
        kg = data["knowledge_graph"]
        parts = []
        for k, v in kg.items():
            key_lower = k.lower()
            if any(bad in key_lower for bad in ["link", "image", "kgmid", "search_for"]):
                continue
            if isinstance(v, dict) and k == "source" and v.get("name"):
                parts.append(f"source: {v['name']} ({v.get('link', '')})")
                continue
            if isinstance(v, (str, int, float)):
                parts.append(f"{k}: {v}")
        return f"[KNOWLEDGE_GRAPH]\n" + "\n".join(parts)

    results = data.get("organic_results", [])
    if not results:
        return "[NO RESULTS]"

    formatted = "\n".join(
        f"- {r.get('title')}: {r.get('snippet')} (Source: {r.get('link')})"
        for r in results
    )
    return f"[SNIPPETS FALLBACK]\n{formatted}"