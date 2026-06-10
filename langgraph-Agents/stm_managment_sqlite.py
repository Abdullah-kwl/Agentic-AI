import time
import uuid
import json
import re
from pathlib import Path
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv(r"D:\Projects\Agentic-AI\.env")

# Where the SQLite checkpoint DB lives — change path if you like
DB_PATH      = Path(r"D:\Projects\Agentic-AI\chat_checkpoints.db")
# File that remembers which thread_ids exist (so you can resume by name)
SESSIONS_PATH = Path(r"D:\Projects\Agentic-AI\chat_sessions.json")

# --------------------------------------------------
# Custom state
# --------------------------------------------------

class AgentState(TypedDict):
    messages:      Annotated[list[BaseMessage], add_messages]
    summary:       str
    summarize_at:  int
    context_start: int

# --------------------------------------------------
# LLM + tools
# --------------------------------------------------

llm = ChatGroq(model="openai/gpt-oss-120b")

sys_msg = SystemMessage(content="You are a helpful assistant tasked with using search and performing arithmetic on a set of inputs.")

def multiply(a: int, b: int) -> int:
    """Multiply a and b.
    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.
    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.
    Args:
        a: first int
        b: second int
    """
    return a / b

search = DuckDuckGoSearchRun()
tools  = [add, multiply, divide, search]
llm_with_tools = llm.bind_tools(tools)

# --------------------------------------------------
# STM thresholds
# --------------------------------------------------

THRESHOLD      = 15
KEEP_RECENT    = 5
SUMMARIZE_STEP = 10

# --------------------------------------------------
# STM helpers
# --------------------------------------------------

def _role(msg: BaseMessage) -> str:
    if isinstance(msg, HumanMessage):   return "User"
    if isinstance(msg, AIMessage):      return "Assistant"
    if isinstance(msg, SystemMessage):  return "System"
    return type(msg).__name__


def get_llm_context(state: AgentState) -> list[BaseMessage]:
    context: list[BaseMessage] = []
    if state["summary"]:
        context.append(SystemMessage(content=f"Conversation summary so far:\n{state['summary']}"))
    context.extend(state["messages"][state["context_start"]:])
    return context


def summarize_and_trim(state: AgentState) -> dict:
    total         = len(state["messages"])
    new_ctx_start = total - KEEP_RECENT
    to_summarize  = state["messages"][state["context_start"]: new_ctx_start]

    if not to_summarize:
        return {"summarize_at": total + SUMMARIZE_STEP}

    history_text = "\n".join(f"{_role(m)}: {m.content}" for m in to_summarize)

    if state["summary"]:
        prompt = (
            f"You are maintaining a rolling summary of a conversation.\n\n"
            f"Existing summary:\n{state['summary']}\n\n"
            f"New messages to incorporate:\n{history_text}\n\n"
            f"Write an updated, concise summary that captures all important "
            f"context from both the existing summary and the new messages. "
            f"Plain text only, no bullet points."
        )
    else:
        prompt = (
            f"Summarize the following conversation excerpt concisely, "
            f"capturing all important context. Plain text only.\n\n{history_text}"
        )

    response    = llm.invoke([HumanMessage(content=prompt)])
    new_summary = response.content.strip()
    print(f"  ↳ summarized msgs[{state['context_start']}:{new_ctx_start}], keeping msgs[{new_ctx_start}:]")

    return {
        "summary":       new_summary,
        "summarize_at":  total + SUMMARIZE_STEP,
        "context_start": new_ctx_start,
    }

# --------------------------------------------------
# Nodes
# --------------------------------------------------

def reasoner(state: AgentState) -> dict:
    total = len(state["messages"])
    print(f"[reasoner] total={total}  ctx_start={state['context_start']}  summarize_at={state['summarize_at']}")

    stm_update = {}
    if total >= state["summarize_at"]:
        print("  → threshold hit, summarizing...")
        stm_update = summarize_and_trim(state)

    effective_state = {**state, **stm_update}
    context  = get_llm_context(effective_state)
    response = llm_with_tools.invoke([sys_msg] + context)

    return {"messages": [response], **stm_update}

# --------------------------------------------------
# Graph — compiled with SqliteSaver checkpointer
# --------------------------------------------------

builder = StateGraph(AgentState)
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "reasoner")

# SqliteSaver is used as a context manager in the chat loop below
# so we keep the compiled graph as a factory function
def build_graph(checkpointer):
    return builder.compile(checkpointer=checkpointer)

# --------------------------------------------------
# Session management helpers
# --------------------------------------------------

def load_sessions() -> dict:
    """Load {name: thread_id} mapping from disk."""
    if SESSIONS_PATH.exists():
        return json.loads(SESSIONS_PATH.read_text())
    return {}


def save_sessions(sessions: dict) -> None:
    SESSIONS_PATH.write_text(json.dumps(sessions, indent=2))


def pick_or_create_session() -> tuple[str, str, bool]:
    """
    Ask the user whether to start a new session or resume an old one.
    Returns (session_name, thread_id, is_new).
    """
    sessions = load_sessions()

    print("\n" + "=" * 50)
    if sessions:
        print("Existing sessions:")
        for i, name in enumerate(sessions, 1):
            print(f"  {i}. {name}  (id: {sessions[name][:8]}…)")
        print("  n. Start a new session")
        print("=" * 50)

        choice = input("Choose a session number to resume, or 'n' for new: ").strip().lower()

        if choice != "n":
            names = list(sessions.keys())
            try:
                idx  = int(choice) - 1
                name = names[idx]
                print(f"\nResuming session '{name}'…")
                return name, sessions[name], False
            except (ValueError, IndexError):
                print("Invalid choice — starting a new session.")

    # New session
    name      = input("Name this session (or press Enter for a random name): ").strip()
    if not name:
        name  = f"session-{uuid.uuid4().hex[:6]}"
    thread_id = str(uuid.uuid4())
    sessions[name] = thread_id
    save_sessions(sessions)
    print(f"\nNew session '{name}' created (id: {thread_id[:8]}…)")
    return name, thread_id, True


# --------------------------------------------------
# Rate-limit-aware invoke with retry
# --------------------------------------------------

def invoke_with_retry(graph, user_msg: str, config: dict, is_new: bool, initial_state: dict) -> dict:
    """
    Invoke the graph, automatically retrying on Groq 413 rate-limit errors.
    Waits the number of seconds Groq tells us to wait (parsed from the error),
    or 60 seconds if no wait time is given.
    """
    input_state = (
        {**initial_state, "messages": [HumanMessage(content=user_msg)]}
        if is_new
        else {"messages": [HumanMessage(content=user_msg)]}
    )

    while True:
        try:
            return graph.invoke(input_state, config)
        except Exception as e:
            msg = str(e)
            if "413" in msg or "rate_limit_exceeded" in msg:
                # Try to parse "please try again in Xs" from Groq's message
                wait = 60
                match = re.search(r"try again in ([0-9.]+)s", msg)
                if match:
                    wait = int(float(match.group(1))) + 2
                print(f"\n⚠  Rate limit hit. Waiting {wait}s then retrying…")
                time.sleep(wait)
                # After the first attempt the checkpoint exists, so never pass initial_state again
                input_state = {"messages": [HumanMessage(content=user_msg)]}
            else:
                raise


# --------------------------------------------------
# Interactive chat loop
# --------------------------------------------------

if __name__ == "__main__":
    print(react_agent_ascii := builder.build() if False else "")  # suppress; ASCII printed below only on request

    session_name, thread_id, is_new = pick_or_create_session()

    # config is what identifies this conversation to the checkpointer
    config = {"configurable": {"thread_id": thread_id}}

    # Initial state is only used when creating a brand-new session
    initial_state: AgentState = {
        "messages":      [],
        "summary":       "",
        "summarize_at":  THRESHOLD,
        "context_start": 0,
    }

    print(f"\nChat started (session: '{session_name}'). Type 'quit' to exit.\n")

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_graph(checkpointer)

        # If resuming, show how many messages are already saved
        if not is_new:
            saved = graph.get_state(config)
            if saved and saved.values:
                n = len(saved.values.get("messages", []))
                print(f"  Loaded {n} messages from checkpoint.\n")

        first_turn = is_new

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ("quit", "exit", "q"):
                print(f"Session '{session_name}' saved. Goodbye!")
                break

            if not user_input:
                continue

            try:
                result = invoke_with_retry(
                    graph, user_input, config,
                    is_new=first_turn,
                    initial_state=initial_state,
                )
                first_turn = False  # initial_state only used on very first turn

                msgs = result["messages"]
                print(f"\nAssistant: {msgs[-1].content}")
                print(
                    f"[msgs={len(msgs)}  "
                    f"ctx_start={result.get('context_start', 0)}  "
                    f"next_summarize={result.get('summarize_at', THRESHOLD)}]\n"
                )

            except KeyboardInterrupt:
                print(f"\nInterrupted. Session '{session_name}' is saved — run again to resume.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")