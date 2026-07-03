from fastmcp import Client
from fastmcp.server import create_proxy

mcp = create_proxy(
    Client({
        "mcpServers": {
            "horizon": {
                "command": "npx",
                "args": ["-y", "mcp-remote@latest", "https://SampleMCP-Live.fastmcp.app/mcp"]
            }
        }
    }),
    name="SampleMCP-Proxy",
)

if __name__ == "__main__":
    mcp.run()

# fastmcp dev inspector mcp/mcp_proxy_server.py
# "https://SampleMCP-Live.fastmcp.app/mcp" is deployed on Horizon's
# run deployed mcp server on local inspector
