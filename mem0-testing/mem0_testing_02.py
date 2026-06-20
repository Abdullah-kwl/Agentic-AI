import os
import warnings
warnings.filterwarnings("ignore")
from mem0 import Memory
from dotenv import load_dotenv
load_dotenv(r"D:\Projects\Agentic-AI\.env")

def cleanup():
    try:
        m.vector_store.client.close()
    except Exception:
        pass

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 2000,
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "memories",
            "path": r"D:\Projects\Agentic-AI\mem0-testing\qdrant_db",
        }
    }
}
# Remove-Item -Recurse -Force "C:\Users\Abdullah\.mem0"


m = Memory.from_config(config)

# messages1 = [
#     {"role": "user", "content": "I'm currently living in Pakistan."},
#     {"role": "assistant", "content": "Got it! I'll update your location to Pakistan."},
# ]
# m.add(messages1, user_id="test_user")


messages2 = [
    {"role": "user", "content": "I moved to Canada"},
    {"role": "assistant", "content": "Got it! I'll update your location to Canada."},
]
# m.add(messages2, user_id="test_user")

# Step 3 - check
results = m.search("where do I live?", filters={"user_id": "test_user"}, limit=5)
print(results)
# for r in results["results"]:
#     print(r["memory"], "->", r["score"])


# result1 = m.add(messages1, user_id="test_user")
# print("Add 1 result:", result1)

# result2 = m.add(messages2, user_id="test_user")
# print("Add 2 result:", result2)

# explicit cleanup LAST line
try:
    m.vector_store.client.close()
except Exception:
    pass


# cleanup - must be absolute last lines
import gc
m.vector_store.client.close()
gc.collect()


# New message arrives: "I moved to Canada"
#         ↓
# Step 1: LLM extracts clean fact
#         → "User moved to Canada"
#         ↓
# Step 2: Vector search with that fact
#         → finds "User lives in Pakistan" (similar enough)
#         ↓
# Step 3: Send BOTH to LLM
#         new: "User moved to Canada"
#         old: "User lives in Pakistan"
#         ↓
# Step 4: LLM rewrites intelligently
#         → "User moved from Pakistan to Canada"
#         ↓
# Step 5: LLM also returns an action
#         → UPDATE / DELETE / ADD / NONE
#         ↓
# Step 6: Execute action
#         UPDATE → overwrite old memory with new rewritten one
#         DELETE → remove old memory entirely
#         ADD    → keep old, add new separately