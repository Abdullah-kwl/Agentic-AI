import random
from fastmcp import FastMCP

mcp = FastMCP(name="SampleMCP")

@mcp.tool
def roll_dice(n_dice: int = 1, sides: int = 6) -> list[int]:
    """Roll n_dice 6-sides dice and return the results."""
    return [random.randint(1, sides) for _ in range(n_dice)]
    

@mcp.tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b

if __name__ == "__main__":
    mcp.run()

# run command
# fastmcp dev inspector mcp/mcp_sample.py
# fastmcp run mcp/mcp_sample.py
# fastmcp install claude-desktop mcp/mcp_sample.py


# "mcpServers": {
#     "sample-mcp": {
#       "command": "D:\\Projects\\Agentic-AI\\.venv\\Scripts\\fastmcp.exe",
#       "args": ["run", "D:\\Projects\\Agentic-AI\\mcp\\mcp_sample.py"]
#     }
#   }
