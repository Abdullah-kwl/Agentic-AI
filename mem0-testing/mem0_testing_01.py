import os
from mem0 import Memory
from dotenv import load_dotenv
load_dotenv(r"D:\Projects\Agentic-AI\.env")

config = {
    "llm": {
        "provider": "groq",
        "config": {
            "model": "mixtral-8x7b-32768",
            "temperature": 0.1,
            "max_tokens": 2000,
        }
    },
    
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/gemini-embedding-001",
        }
    }
}


config = {
    "llm": {
        "provider": "groq",
        "config": {
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.1,
            "max_tokens": 2000,
        }
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model":  "models/gemini-embedding-2",
            # "api_key": os.getenv("GEMINI_API_KEY"),
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "memories",
            "path": r"D:\Projects\Agentic-AI\mem0-testing\chroma_db",  # where it saves files
        }
    }
}

m = Memory.from_config(config)



# messages = [
#     {"role": "user", "content": "Hi, I'm Alex. I love basketball and gaming."},
#     {"role": "assistant", "content": "Hey Alex! I'll remember your interests."}
# ]

# messages = [
#     {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
#     {"role": "assistant", "content": "How about thriller movies? They can be quite engaging."},
#     {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
#     {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
# ]

# m.add(messages, user_id="alex")

# results = m.search("can you give me some movie recommendations?", filters={"user_id": "alex"},  top_k=3)
# results = m.search("do you know my name?", filters={"user_id": "alex"}, top_k=3)
# print(results)

# First add
# messages1 = [{"role": "user", "content": "I live in Pakistan"}]
# m.add(messages1, user_id="alex")

# Later add
# messages2 = [{"role": "user", "content": "I moved to Canada"}]
# m.add(messages2, user_id="alex")

# Search
# results = m.search("In which country do I live?", filters={"user_id": "alex"}, limit=3, top_k=5)
# print(results)

# messages1 = [{"role": "user", "content": "I live in Pakistan"}]
# m.add(messages1, user_id="alex")

# results = m.search("In which country do I live?", filters={"user_id": "alex"}, limit=5)
# print(results)


# messages2 = [{"role": "user", "content": "I moved to Canada"}]
# m.add(messages2, user_id="alex")

# results = m.search("In which country do I live?", filters={"user_id": "alex"}, limit=5)
# print(results)


# Step 1 - add Pakistan only
messages1 = [{"role": "user", "content": "I live in Pakistan"}]
m.add(messages1, user_id="test_user")

# Step 2 - add Canada
messages2 = [{"role": "user", "content": "I moved to Canada"}]
m.add(messages2, user_id="test_user")

# Step 3 - check
results = m.search("where do I live?", filters={"user_id": "test_user"}, limit=5)
for r in results["results"]:
    print(r["memory"], "->", r["score"])