import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("SERP_API_KEY")

response = requests.get(
    "https://serpapi.com/search",
    params={
        "engine": "google",
        "q": "Who is president of Pakistan?",
        "api_key": API_KEY
    }
)

data = response.json()

if "answer_box" in data:
    print("ANSWER BOX:", json.dumps(data["answer_box"], indent=2))
elif "knowledge_graph" in data:
    print("KNOWLEDGE GRAPH:", json.dumps(data["knowledge_graph"], indent=2))
else:
    print("No direct answer — falling back to snippets:")
    for r in data.get("organic_results", [])[:5]:
        print(f"- {r.get('title')}: {r.get('snippet')}")