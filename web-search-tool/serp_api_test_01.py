# import os
# import requests
# import json
# from dotenv import load_dotenv
# load_dotenv()

# API_KEY = os.getenv("SERP_API_KEY")

# response = requests.get(
#     "https://serpapi.com/search",
#     params={
#         "engine": "google",
#         "q": "Who is president of Pakistan?",
#         "api_key": API_KEY
#     }
# )

# print(json.dumps(response.json(), indent=2))


import os
import requests
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("SERP_API_KEY")

response = requests.get(
    "https://serpapi.com/search",
    params={
        "engine": "google",
        "q": "Who is president of Pakistan?",
        "api_key": API_KEY,
        "hl": "en",
        "gl": "us"
    }
)

data = response.json()
results = data.get("organic_results", [])
print(f"Total organic_results returned: {len(results)}")
for i, r in enumerate(results, 1):
    print(f"{i}. {r.get('title')}")