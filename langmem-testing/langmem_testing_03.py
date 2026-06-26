import asyncio

# =============================================================================
# FILE 03 — Hot Path Memory (Agent uses tools directly during conversation)
# =============================================================================
# PURPOSE:
#   The agent itself decides when to save and search memories, in real time,
#   as part of the conversation turn. Memory is conscious and deliberate.
#
# HOW IT DIFFERS FROM FILES 01 AND 02:
#   Files 01/02: Memory extraction runs AFTER the conversation (background/passive)
#   File 03:     The agent calls manage_memory/search_memory DURING its response
#                The agent "thinks" about memory as it talks
#
# TWO TOOLS:
#   create_manage_memory_tool → agent can create, update, delete memories by ID
#   create_search_memory_tool → agent can search memories by semantic query
#
# TRADEOFF vs background approach:
#   PRO:  Agent is aware of what it's saving — more intentional, less noise
#   PRO:  User can say "remember this" and agent acts immediately
#   CON:  Adds tool calls to every response turn → higher latency + token cost
#   CON:  Agent may forget to search or save if system prompt is weak
#
# SYSTEM PROMPT IS CRITICAL HERE:
#   Without explicit rules in the system prompt, the agent will save memories
#   inconsistently and may forget to search at conversation start.
#   Always instruct: when to search, when to save, how to handle conflicts.
#
# namespace=("memories", "{user_id}"):
#   {user_id} is filled from config["configurable"]["user_id"] at runtime.
#   Each user gets their own isolated namespace — no cross-user memory leakage.
# =============================================================================

# Agent-based memory approach: instead of a background manager that extracts
# memories after the fact, here the LLM agent ITSELF decides when to call
# manage_memory and search_memory as tools during the conversation turn.
# This gives the agent real-time awareness of its own memory state.
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from dotenv import load_dotenv
load_dotenv(r"D:\Projects\Agentic-AI\.env")


# InMemoryStore with vector index for semantic search.
# dims=1536 matches OpenAI text-embedding-3-small.
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# The system prompt instructs the agent on WHEN to use each memory tool.
# Without explicit rules here, the agent may forget to search at the start
# or save memories inconsistently.
SYSTEM_PROMPT = """You are a helpful assistant with persistent memory.

## Memory Usage Rules
- ALWAYS call search_memory at the start of each response to recall relevant context about the user.
- When the user shares a preference, habit, or personal fact, call manage_memory to save it.
- When an existing memory contradicts new information, update it instead of creating a duplicate.
- Use retrieved memories naturally in your responses — don't announce "I found in memory that..."
"""

# namespace=("memories", "{user_id}"): memories are scoped per user.
# {user_id} is a template filled at runtime from config["configurable"]["user_id"],
# so different users never share or see each other's memories.
agent = create_agent(
    model,
    tools=[
        create_manage_memory_tool(namespace=("memories", "{user_id}")),
        create_search_memory_tool(namespace=("memories", "{user_id}")),
    ],
    store=store,
    system_prompt=SYSTEM_PROMPT,
)

EXIT_WORDS = {"exit", "quit", "q", "bye", "stop"}


def print_memories(user_id: str):
    """Print all memories stored for the given user at end of session."""
    namespace = ("memories", user_id)
    items = store.search(namespace)
    if not items:
        print("  (no memories stored)")
        return
    for item in items:
        print(f"  [{item.key}] {item.value}")


async def chat(user_id: str = "user123"):
    # config passes the user_id into the namespace template at runtime.
    config = {"configurable": {"user_id": user_id}}
    messages = []
    print(f"\nChat started for user '{user_id}'. Type 'exit' to end.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in EXIT_WORDS:
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # ainvoke passes the full message history each turn so the agent has
        # conversation context. The store provides cross-session memory.
        result = await agent.ainvoke({"messages": messages}, config=config)
        messages = result["messages"]

        print(f"\nAssistant: {messages[-1].content}\n")

    print("\n--- Memories stored for this session ---")
    print_memories(user_id)


USER_ID = "user123"

if __name__ == "__main__":
    asyncio.run(chat(user_id=USER_ID))
