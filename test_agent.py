import asyncio
from claude_agent_sdk import query

async def main():
    async for message in query(prompt="Say hello and confirm you're running on LOADSERV-1"):
        print(message)

asyncio.run(main())
