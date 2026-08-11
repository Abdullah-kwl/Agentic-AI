import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("SERP_API_KEY")

test_queries = [
    "NFLX stock price",
    "population of New York",
    "define transparent",
]

for q in test_queries:
    response = requests.get(
        "https://serpapi.com/search",
        params={"engine": "google", "q": q, "api_key": API_KEY}
    )
    data = response.json()
    print(f"\n=== Query: {q} ===")
    if "answer_box" in data:
        print("ANSWER BOX:", json.dumps(data["answer_box"], indent=2))
    else:
        print("Not present.")