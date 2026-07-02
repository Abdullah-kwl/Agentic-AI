import random
from fastmcp import FastMCP

mcp = FastMCP(name="SampleMCP-Live")

@mcp.tool
def roll_dice(n_dice: int = 1, sides: int = 6) -> list[int]:
    """Roll n_dice 6-sides dice and return the results."""
    return [random.randint(1, sides) for _ in range(n_dice)]
    

@mcp.tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)


# fastmcp run mcp/mcp_sample_live.py --transport http --port 9000 --reload
# OR
# python  mcp/mcp_sample_live.py 
# fastmcp dev inspector mcp/mcp_sample_live.py 