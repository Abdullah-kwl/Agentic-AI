import asyncio
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient # conceptual adapter usage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv(r"D:\Projects\Agentic-AI\.env")

# 1. Initialize your model
model = ChatOpenAI(model="gpt-4o")

# 2. Connect to your deployed MCP server endpoint
async def run_agent():
    client = MultiServerMCPClient(
        {
            "horizon": {
                "command": "npx",
                "args": ["-y", "mcp-remote@latest", "https://SampleMCP-Live.fastmcp.app/mcp"],
                "transport": "stdio",
            }
        }
    )

    # Automatically discover and fetch tools from the MCP server
    tools = await client.get_tools()

    # 3. Create the LangGraph ReAct agent with remote MCP tools
    agent = create_react_agent(model, tools)

    # 4. Invoke the graph
    result = await agent.ainvoke({"messages": [("user", "Use the tools to complete this task 'roll 3 dies'.")]})
    return result

result = asyncio.run(run_agent())
print(result)