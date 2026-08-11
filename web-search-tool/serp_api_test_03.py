import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("SERP_API_KEY")

test_queries = ["what is 25 * 17", "Elon Musk", "define ephemeral"]

for q in test_queries:
    response = requests.get(
        "https://serpapi.com/search",
        params={"engine": "google", "q": q, "api_key": API_KEY}
    )
    data = response.json()
    print(f"\n=== Query: {q} ===")
    if "answer_box" in data:
        print("ANSWER BOX:", json.dumps(data["answer_box"], indent=2))
    if "knowledge_graph" in data:
        print("KNOWLEDGE GRAPH:", json.dumps(data["knowledge_graph"], indent=2))
    if "answer_box" not in data and "knowledge_graph" not in data:
        print("Neither present for this query.")