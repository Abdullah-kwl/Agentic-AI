import asyncio

# =============================================================================
# langmem_testing_01.py — create_memory_manager (STATELESS / FUNCTIONAL API)
# =============================================================================
# PURPOSE:
#   Lowest-level LangMem primitive. Pure extraction function — no storage at all.
#   Give it a conversation, get back a list of memory decisions. That's it.
#
# HOW IT WORKS INTERNALLY:
#   1. ainvoke({"messages": [...]}) called
#   2. LLM reads conversation + "existing" memories (if provided)
#   3. LLM returns ExtractedMemory objects (new/updated facts)
#      and/or RemoveDoc objects (signals to delete by ID)
#   4. You receive that list — YOU decide what to persist, filter, or ignore
#
# KEY POINT — "existing" is YOUR responsibility:
#   Without passing "existing", every call starts with zero memory context.
#   The manager has no store to read from — it only knows what YOU hand it.
#
# RemoveDoc explained:
#   When enable_deletes=True and info changes, the LLM emits:
#     RemoveDoc(json_doc_id="abc-123")   <- "please delete this ID"
#     ExtractedMemory(content=...)       <- "replace it with this"
#   It does NOT delete anything. You must act on RemoveDoc yourself.
#
# WHEN TO USE THIS (not file 02):
#   - Custom storage backend (not LangGraph BaseStore)
#   - You need to inspect/transform memories before saving
#   - Unit testing extraction logic in isolation
# =============================================================================

# create_memory_manager: STATELESS memory extractor — no store, no persistence.
# It processes a conversation and returns a list of memory objects in-process only.
# You are responsible for passing previous memories back via the "existing" key
# on the next call, otherwise it starts fresh every time.
from langmem import create_memory_manager
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv(r"D:\Projects\Agentic-AI\.env")

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Custom instructions tell the LLM exactly how to extract and store memories.
# Without this, the default instructions are more generic and may merge
# unrelated preferences into a single memory entry.
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

# enable_inserts=True  → allow creating brand-new memory entries
# enable_updates=True  → allow patching an existing memory when info changes
# enable_deletes=True  → allow removing a memory that is contradicted / outdated
memory_manager = create_memory_manager(
    model,
    instructions=MEMORY_INSTRUCTIONS,
    enable_inserts=True,
    enable_updates=True,
    enable_deletes=True,
)

# First conversation: four distinct preferences are stated.
# The manager should produce four separate memory entries.
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

# Second conversation: user changes their dark-mode preference to gray mode.
# By passing memories_1 as "existing", the manager knows to UPDATE the dark-mode
# entry rather than insert a duplicate.
conversation_messages_2 = [
    {"role": "user", "content": "I prefer gray mode in all my apps as it looks better than black."},
    {"role": "assistant", "content": "Got it, I'll keep that preference in mind."},
]

async def main():
    # First pass: no existing memories, so everything is freshly inserted.
    memories_1 = await memory_manager.ainvoke({"messages": conversation_messages_1})

    # Second pass: pass memories_1 as "existing" so the manager can detect
    # which entry to update instead of creating a duplicate.
    memories_2 = await memory_manager.ainvoke({"messages": conversation_messages_2, "existing": memories_1})

    print(memories_1)
    print(len(memories_1))
    print("----"*5)
    print(memories_2)
    print(len(memories_2))

asyncio.run(main())


# memories = await manager.ainvoke(
#     {"messages": conversation, "max_steps": max_steps}
# )

# if max_steps > 1:
#     session += f"\n\nYou have a maximum of {max_steps - 1} attempts"
#               " to form and consolidate memories from this session."


"""
You are a long-term memory manager maintaining a core store of semantic, procedural, and episodic memory. These memories power a life-long learning agent's core predictive model.

What should the agent learn from this interaction about the user, itself, or how it should act? Reflect on the input trajectory and current memories (if any).

1. Extract & Contextualize

* Identify essential facts, relationships, preferences, reasoning procedures, and context.
* Caveat uncertain or suppositional information with confidence levels (p(x)) and reasoning.
* Quote supporting information when necessary.

2. Compare & Update

* Attend to novel information that deviates from existing memories and expectations.
* Maintain information density within each memory entry, but preserve atomicity across distinct facts, preferences, relationships, procedures, and traits.
* Create separate memory entries for independent preferences or facts, even if they appear in the same conversation.
* Consolidate only truly redundant memories that represent the same underlying fact.
* Update only the memory directly affected by new information.
* Remove incorrect or obsolete memories while maintaining internal consistency.
* Never merge unrelated preferences, habits, traits, or facts into a single memory solely for compression.
* If uncertain whether information belongs in one memory or multiple memories, prefer multiple memories.

3. Synthesize & Reason

* What can you conclude about the user, agent ("I"), or environment using deduction, induction, and abduction?
* What patterns, relationships, and principles emerge about optimal responses?
* What generalizations can you make?
* Qualify conclusions with probabilistic confidence and justification.

Memory Formation Principles

* One independent preference, fact, relationship, procedure, or trait should generally correspond to one memory entry.
* Context that explains a preference may be included within that preference's memory when it improves future recall.
* Avoid conversation summaries, profile summaries, or catch-all memories that combine multiple unrelated topics.
* Prefer multiple information-rich atomic memories over a single compressed summary memory.
* Memories should be written exactly as they would be most useful to recall when predicting how to act or respond in the future.

As the agent, record memory content exactly as you'd want to recall it when predicting how to act or respond.

Prioritize retention of surprising (pattern deviation) and persistent (frequently reinforced) information, ensuring nothing worth remembering is forgotten and nothing false is remembered.

Prefer dense, information-rich memories about a single underlying fact over broad memories that combine multiple unrelated facts.

"""
