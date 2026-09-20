import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def main():
    options = ClaudeAgentOptions(allowed_tools=["WebSearch", "WebFetch"])

    async for message in query(prompt="Search the web for the cast of the newest Spider-Man movie", options=options):
        if hasattr(message, "content"):
            for block in message.content:
                block_type = type(block).__name__
                if block_type == "ToolUseBlock":
                    print(f"[TOOL USE] name={block.name}")
                elif block_type == "ToolResultBlock":
                    print(f"[TOOL RESULT] content type={type(block.content).__name__}")
                    print(f"[TOOL RESULT] content preview: {repr(block.content)[:400]}")
                    print()

asyncio.run(main())
