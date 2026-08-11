import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("SERP_API_KEY")

test_queries = [
    # "NFLX stock price",               # answer_box -> finance_results
    "Elon Musk",                      # knowledge_graph
    # "Who is president of Pakistan?",  # fallback -> snippets
]

def search(query):
    response = requests.get(
        "https://serpapi.com/search",
        params={
            "engine": "google",
            "q": query,
            "api_key": API_KEY,
            "hl": "en",
        }
    )
    return response.json()

def parse_result(data):
    if "answer_box" in data:
        return f"[ANSWER_BOX]\n{json.dumps(data['answer_box'], indent=2)}"

    if "knowledge_graph" in data:
        kg = data["knowledge_graph"]
        parts = []

        for k, v in kg.items():
            key_lower = k.lower()
            # skip navigation/tracking fields — links, images, ids, search-for lists
            if any(bad in key_lower for bad in ["link", "image", "kgmid", "search_for"]):
                continue
            # keep the citation source name specifically
            if isinstance(v, dict) and k == "source" and v.get("name"):
                parts.append(f"source: {v['name']}")
                continue
            # keep simple factual fields only — skip nested lists/dicts
            if isinstance(v, (str, int, float)):
                parts.append(f"{k}: {v}")

        return f"[KNOWLEDGE_GRAPH]\n" + "\n".join(parts)

    results = data.get("organic_results", [])[:5]
    if not results:
        return "[NO RESULTS]"

    formatted = "\n".join(f"- {r.get('title')}: {r.get('snippet')}" for r in results)
    return f"[SNIPPETS FALLBACK]\n{formatted}"

for q in test_queries:
    print(f"\n=== Query: {q} ===")
    data = search(q)
    print(parse_result(data))