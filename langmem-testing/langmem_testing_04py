import asyncio

# =============================================================================
# FILE 04 — Structured Profile Memory (Pydantic schema + tight store control)
# =============================================================================
# PURPOSE:
#   Enforce a fixed typed schema (UserProfile) for every memory entry, and
#   guarantee exactly one profile record per user via enable_inserts=False.
#
# WHY enable_inserts=False IS ESSENTIAL FOR PROFILES:
#   A profile is a single document — there should only ever be ONE UserProfile
#   entry per user. If enable_inserts=True, every conversation could create a
#   second profile record instead of updating the existing one.
#   enable_inserts=False forces the manager to only update the existing entry.
#
# WHY THE default PARAMETER MATTERS:
#   On the very first ainvoke(), the store is empty — store.search() returns nothing.
#   Without a default, the manager has nothing to update and does nothing.
#   With default=UserProfile(name="Maria Santos"), the manager seeds the store
#   with a known starting state before the first extraction runs.
#   This is cleaner than manually calling store.aput() to initialise the profile.
#
# HOW SEARCH WORKS WITH SCHEMAS:
#   The full Pydantic object is serialised to a string before embedding.
#   e.g. "UserProfile name=Maria Santos age=41 city=Lisbon country_born=Brazil ..."
#   All fields contribute to the embedding — richer field values = better retrieval.
#   The `extra` dict fields are also embedded, making them searchable too.
#
# WHY extra: dict EXISTS:
#   Real conversations surface facts that don't fit predefined fields.
#   e.g. "I worked in Japan for 2 years", "I received an architecture award"
#   Rather than losing these or adding infinite Optional fields, they land in `extra`
#   as {"years_in_japan": "2 years", "award": "national architecture award 2018"}.
#   model_config=extra="allow" is a Pydantic-level safety net on top of this.
# =============================================================================

# Structured memory with a Pydantic schema and tighter store control.
# Previous files stored memories as free-form strings. Here we enforce a fixed
# schema (UserProfile) so every memory entry has predictable, typed fields.
from langmem import create_memory_store_manager
from langgraph.store.memory import InMemoryStore
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

from dotenv import load_dotenv
load_dotenv(r"D:\Projects\Agentic-AI\.env")

class UserProfile(BaseModel):
    """Biographical and demographic profile of a user."""
    # Groq requires every tool function to have a non-null description.
    # Pydantic's class docstring becomes the tool description via trustcall,
    # so without this docstring Groq returns a 400 "Value is not nullable" error.

    # extra="allow" lets the model populate fields that aren't declared below
    # without raising a validation error. Useful for one-off biographical facts.
    model_config = ConfigDict(extra="allow")

    # Identity
    name: Optional[str] = Field(None, description="Full name or preferred name of the user")
    date_of_birth: Optional[str] = Field(None, description="Date of birth in YYYY-MM-DD format")
    age: Optional[int] = Field(None, description="Age of the user")
    gender: Optional[str] = Field(None, description="Gender identity of the user")
    nationality: Optional[str] = Field(None, description="Nationality or citizenship")

    # Location history
    country_born: Optional[str] = Field(None, description="Country where the user was born")
    country_raised: Optional[str] = Field(None, description="Country where the user grew up")
    country_living: Optional[str] = Field(None, description="Country where the user currently lives")
    city: Optional[str] = Field(None, description="City where the user currently lives")

    # Education & Career
    occupation: Optional[str] = Field(None, description="Current job title or profession")
    employer: Optional[str] = Field(None, description="Current employer or company name")
    industry: Optional[str] = Field(None, description="Industry the user works in")
    education_level: Optional[str] = Field(None, description="Highest education level e.g. BSc, MSc, PhD")
    field_of_study: Optional[str] = Field(None, description="What the user studied")
    university: Optional[str] = Field(None, description="University or institution attended")

    # Language & Background
    native_language: Optional[str] = Field(None, description="User's first or native language")
    languages_spoken: Optional[list[str]] = Field(None, description="All languages the user speaks")
    ethnicity: Optional[str] = Field(None, description="Ethnic or cultural background if shared")
    religion: Optional[str] = Field(None, description="Religion if shared by the user")

    # Personal context
    marital_status: Optional[str] = Field(None, description="e.g. single, married, divorced")
    children: Optional[int] = Field(None, description="Number of children if mentioned")

    # Catch-all for biographical facts that don't fit the declared fields above.
    # Using a dict keeps the schema open-ended without polluting the typed fields.
    # Do NOT put preferences or opinions here — those belong in a separate memory type.
    extra: dict = Field(
        default_factory=dict,
        description=(
            "Any additional biographical or demographic facts that don't fit the fields above. "
            "Use snake_case keys with natural language values. "
            "e.g. {'country_studied': 'England', 'military_service': 'served 2 years', 'favorite_sport': 'Cricket', 'github_username': 'abd123'} "
            "Do NOT store preferences, opinions, or likes/dislikes here."
        )
    )


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


store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

# schemas=[UserProfile]: tells the manager every memory must conform to UserProfile.
#   The LLM receives UserProfile as a structured output tool (via trustcall) and
#   must fill its fields rather than writing free-form text.
#
# enable_inserts=False: the manager cannot create new memory entries on its own.
#   This gives tight control — exactly one UserProfile entry will ever exist,
#   the one seeded by the `default` parameter below.
#   Without this, the LLM could accidentally create duplicate profiles.
#
# enable_deletes=True: allows the manager to clear a field back to None if the
#   user explicitly retracts or contradicts information.
#
# default=UserProfile(name="Maria Santos"):
#   On the very first ainvoke call, if no memories exist yet, the manager
#   persists this value to the store under the key "default".
#   This is the clean built-in alternative to manually calling store.aput() —
#   no async seed helper needed. Only name is pre-filled; everything else
#   starts as None and gets populated as conversations are processed.
manager = create_memory_store_manager(
    model,
    namespace=("memories", ),
    store=store,
    instructions=MEMORY_INSTRUCTIONS,
    enable_inserts=False,
    enable_deletes=True,
    schemas=[UserProfile],
    default=UserProfile(name="Maria Santos"),
)

conversation_messages_1 = [
    # {"role": "user", "content": "My name is Maria Santos."},
    # {"role": "assistant", "content": "Hello Maria."},

    {"role": "user", "content": "Hi."},
    {"role": "assistant", "content": "Hello, how can i help you today?"},

    {"role": "user", "content": "I'm 41 years old and originally from Brazil."},
    {"role": "assistant", "content": "Thanks for sharing."},

    {"role": "user", "content": "I currently live in Lisbon, Portugal."},
    {"role": "assistant", "content": "Nice city."},

    {"role": "user", "content": "I work as an architect and run my own design studio."},
    {"role": "assistant", "content": "That sounds exciting."},

    {"role": "user", "content": "I studied Architecture at the University of São Paulo."},
    {"role": "assistant", "content": "Great background."},

    {"role": "user", "content": "I also spent two years working in Japan and received a national architecture award in 2018."},
    {"role": "assistant", "content": "Those are impressive achievements."},

    {"role": "user", "content": "Besides Portuguese, I speak English and Japanese."},
    {"role": "assistant", "content": "Very useful language combination."}
]

# Second conversation adds a new biographical fact.
# The manager patches the SAME "default" entry — no new entry is created
# because enable_inserts=False.
conversation_messages_2 = [
    {"role": "user", "content": "I also served in the military for 2 years."},
    {"role": "assistant", "content": "That's interesting, it shows your commitment and patriotism."},
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

    memories_2 = await manager.ainvoke({"messages": conversation_messages_2})
    print(memories_2)
    print(len(memories_2))

asyncio.run(main())
print("----"*5)
# Final state of the store — shows the single UserProfile entry with all
# fields accumulated across both conversation turns.
print(manager.search(query=None))
