import asyncio

# =============================================================================
# FILE 02 — create_memory_store_manager (STATEFUL / STORE-BACKED API)
# =============================================================================
# PURPOSE:
#   Same extraction logic as file 01, but wired directly to a BaseStore.
#   The manager handles read-before-extract and write-after-extract automatically.
#
# WHAT IT DOES ON EVERY ainvoke() CALL:
#   1. Runs store.search(namespace, query=current_conversation) → fetches top-K
#      semantically relevant existing memories (default K=5, set via query_limit)
#   2. Passes those as "existing" to the extraction LLM (you don't do this manually)
#   3. LLM returns ExtractedMemory + RemoveDoc decisions
#   4. Manager applies them to the store: inserts new, patches updated, deletes stale
#
# CRITICAL — it searches, not loads ALL:
#   Step 1 uses SEMANTIC SEARCH not "load everything".
#   If an existing memory isn't semantically close to the current conversation,
#   it may not appear in "existing" → LLM won't see it → may create a duplicate.
#   This is why dedicated namespaces per memory type matter (see file 04).
#
# WHEN TO USE THIS (not file 01):
#   - LangGraph-based agents with a BaseStore already configured
#   - You want zero manual memory plumbing
#   - Production multi-user apps (add {langgraph_user_id} to namespace)
# =============================================================================

# create_memory_store_manager: STATEFUL memory manager backed by a persistent store.
# Unlike create_memory_manager (file 01), you do NOT manually pass "existing" memories —
# the manager queries the store automatically before each ainvoke call and saves
# results back to the store. This is the production-ready approach.
from langmem import create_memory_store_manager
from langgraph.store.memory import InMemoryStore
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv(r"D:\Projects\Agentic-AI\.env")

MEMORY_INSTRUCTIONS = """You are a precise memory manager that extracts and maintains individual user preferences and facts.

## EXTRACTION RULES
- One memory entry per distinct preference or fact — never merge two separate topics.
- Call the Memory tool once per preference using parallel multi-tool calling.
- Each of the following is a SEPARATE memory: UI preference, communication style, work habit, decision style, etc.

## WHAT TO EXTRACT
- Explicit preferences: "I prefer X", "I like X", "I want X"
- Context that explains a preference: treat it as part of that preference's memory, not a separate one
- Communication style: response length, format, tone
- Workflow habits: when/how the user works

## UPDATE RULES (when existing memories are provided)
- Only modify the single memory entry that the new information directly contradicts or refines.
- Never update an unrelated memory as a side effect.
- If a preference reverses, update only that specific entry.
- If information is entirely new, insert a fresh entry — never append it to an existing memory.

## NEVER DO
- Combine two separate preferences into one memory entry
- Summarize the entire conversation into a single memory
- Consolidate or compress memories that cover different topics
- Speculate beyond what the user explicitly stated"""


model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# InMemoryStore: LangGraph's in-process key-value store.
# The `index` config enables semantic (vector) search over stored memories so
# the manager can retrieve the most relevant ones for each new conversation.
# dims=1536 matches OpenAI's text-embedding-3-small output size.
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

# namespace=("memories",): all memories for this manager live under this path.
# In multi-user apps you'd add a user ID segment e.g. ("memories", "{langgraph_user_id}").
#
# enable_inserts=True  → the manager can create new memory entries when new info arrives.
#                        Required here because the store starts empty and needs to populate.
# enable_deletes=True  → the manager can remove a memory entry that is fully contradicted.
manager = create_memory_store_manager(
    model,
    namespace=("memories", ),
    store=store,
    instructions=MEMORY_INSTRUCTIONS,
    enable_inserts=True,
    enable_deletes=True,
)

# First conversation: multiple distinct preferences.
# The manager inserts a separate entry for each one.
conversation_messages_1 = [
    {"role": "user", "content": "I prefer dark mode in all my apps."},
    {"role": "assistant", "content": "Got it, I'll keep that preference in mind."},
    {"role": "user", "content": "I usually work late at night, so bright interfaces strain my eyes."},
    {"role": "assistant", "content": "That makes sense. Dark mode can be more comfortable in low-light environments."},
    {"role": "user", "content": "I also prefer concise answers instead of long explanations."},
    {"role": "assistant", "content": "Understood. I'll try to keep responses brief and to the point."},
    {"role": "user", "content": "When comparing options, I like seeing pros and cons."},
    {"role": "assistant", "content": "Noted. I'll present trade-offs clearly when helping you make decisions."}
]

# Second conversation: user changes dark-mode preference to gray mode.
# The manager automatically loads existing memories from the store, identifies
# the dark-mode entry, and updates it — no manual "existing" passing needed.
conversation_messages_2 = [
    {"role": "user", "content": "I prefer gray mode in all my apps as it looks better than black."},
    {"role": "assistant", "content": "Got it, I'll keep that preference in mind."},
]

# conversation_messages_2 = [
#     {"role": "user", "content": "I prefer gray mode in all my apps as it looks better than black."},
#     {"role": "assistant", "content": "Got it, I'll keep that preference in mind."},
# ]

async def main():
    memories_1 = await manager.ainvoke({"messages": conversation_messages_1})
    print(memories_1)
    print(len(memories_1))
    print("----"*5)

    # No need to pass memories_1 here — the manager reads from the store automatically.
    memories_2 = await manager.ainvoke({"messages": conversation_messages_2})
    print(memories_2)
    print(len(memories_2))

asyncio.run(main())

print("----"*5)
# manager.search() lets you query the store directly after the session.
# query=None returns all stored memories without semantic filtering.
print(manager.search(query=None))
