import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

# <script async src="https://cse.google.com/cse.js?cx=30500d39a18aa4949">
# </script>
# <div class="gcse-search"></div>

API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
CX = os.getenv("GOOGLE_SEARCH_CX")


response = requests.get(
    "https://www.googleapis.com/customsearch/v1",
    params={
        "key": API_KEY,
        "cx": CX,
        "q": "Who is president of Pakistan?"
    }
)

print(json.dumps(response.json(), indent=2))