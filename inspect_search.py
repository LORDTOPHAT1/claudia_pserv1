import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def main():
    options = ClaudeAgentOptions(allowed_tools=["WebSearch", "WebFetch"])

    async for message in query(prompt="Search the web for today's weather in Adelaide, Australia", options=options):
        print(f"--- Message type: {type(message).__name__} ---")
        if hasattr(message, "content"):
            for block in message.content:
                print(f"  Block type: {type(block).__name__}")
                print(f"  Block attributes: {dir(block)}")
                print(f"  Block repr: {repr(block)[:300]}")
                print()

asyncio.run(main())
