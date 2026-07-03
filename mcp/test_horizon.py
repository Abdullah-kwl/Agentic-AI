import asyncio
from fastmcp import Client

async def main():
    transport = {
        "mcpServers": {
            "horizon": {
                "command": "npx",
                "args": ["-y", "mcp-remote@latest", "https://SampleMCP-Live.fastmcp.app/mcp"]
            }
        }
    }
    async with Client(transport) as client:
        print("Connected to Horizon!")
        tools = await client.list_tools()
        print("Tools:", [t.name for t in tools])

        result = await client.call_tool("add_numbers", {"a": 5, "b": 3})
        print("add_numbers(5, 3) =", result)

asyncio.run(main())

# just test the remote deployed mcp server on horizon's
