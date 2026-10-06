"""End-to-end MCP check: spawn the server over stdio, list tools, call search and read_file.

    python scripts/mcp_smoke_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main() -> None:
    env = {**os.environ, "KB_MCP_USER": os.environ.get("KB_MCP_USER", "mcp-local"), "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}
    params = StdioServerParameters(command=sys.executable, args=["-m", "kbassist.mcp_server"], env=env)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print("tools:", [t.name for t in tools.tools])

        res = await session.call_tool("search_knowledge_base", {"query": "how many redirects before TooManyRedirects", "k": 3})
        # A list result arrives as one text block per item.
        for block in res.content:
            if block.type == "text":
                h = json.loads(block.text)
                print(f"  {h['rank']}. {h['path']}:{h['lines']}  {h['url']}")

        res = await session.call_tool("read_file", {"path": "httpx/_config.py", "start_line": 246, "end_line": 248})
        print("read_file:", res.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
