from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv(r"D:\Projects\Agentic-AI\.env")

# --------------------------------------------------
# Custom state — extends the built-in messages list
# with our three STM tracking fields
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
tools = [add, multiply, divide, search]
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
    if isinstance(msg, HumanMessage):
        return "User"
    if isinstance(msg, AIMessage):
        return "Assistant"
    if isinstance(msg, SystemMessage):
        return "System"
    return type(msg).__name__


def get_llm_context(state: AgentState) -> list[BaseMessage]:
    """What the LLM actually sees: optional summary + recent messages."""
    context: list[BaseMessage] = []
    if state["summary"]:
        context.append(
            SystemMessage(content=f"Conversation summary so far:\n{state['summary']}")
        )
    context.extend(state["messages"][state["context_start"]:])
    return context


def summarize_and_trim(state: AgentState) -> dict:
    """
    Summarize messages that are leaving the context window,
    update the rolling summary, and advance both pointers.
    Returns a partial state dict.
    """
    total         = len(state["messages"])
    new_ctx_start = total - KEEP_RECENT

    to_summarize = state["messages"][state["context_start"]: new_ctx_start]

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
    print(f"[reasoner] total={total}  context_start={state['context_start']}  summarize_at={state['summarize_at']}")

    # Check threshold and merge summary update into the return dict
    stm_update = {}
    if total >= state["summarize_at"]:
        print("  → threshold hit, summarizing...")
        stm_update = summarize_and_trim(state)

    # Build context using the (possibly just-updated) summary and context_start
    # We apply stm_update locally so get_llm_context sees the new pointers
    effective_state = {**state, **stm_update}
    context = get_llm_context(effective_state)

    response = llm_with_tools.invoke([sys_msg] + context)

    # Return both the new message AND any STM pointer updates
    return {"messages": [response], **stm_update}

# --------------------------------------------------
# Graph
# --------------------------------------------------

builder = StateGraph(AgentState)
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "reasoner")

react_agent = builder.compile()

# print(react_agent.get_graph(xray=True).draw_ascii())

# --------------------------------------------------
# Interactive chat loop
# --------------------------------------------------

if __name__ == "__main__":
    state: AgentState = {
        "messages":      [],
        "summary":       "",
        "summarize_at":  THRESHOLD,
        "context_start": 0,
    }

    print("Chat started. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Append the new user message and invoke
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
        state = react_agent.invoke(state)

        print(f"\nAssistant: {state['messages'][-1].content}")
        print(f"[msgs={len(state['messages'])}  ctx_start={state['context_start']}  next_summarize={state['summarize_at']}]\n")