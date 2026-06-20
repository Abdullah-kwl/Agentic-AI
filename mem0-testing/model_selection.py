import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv(r"D:\Projects\Agentic-AI\.env")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

for m in genai.list_models():
    if "embed" in m.name.lower():
        print(m.name, m.supported_generation_methods)