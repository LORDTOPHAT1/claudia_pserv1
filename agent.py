import json
import re
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions
from memory_lib import get_existing_topics, save_memory, search_memory

async def ask_claude_agent(user_message):
    topics = get_existing_topics()
    topics_text = ", ".join(topics) if topics else "none yet"

    relevant = search_memory(user_message)
    if relevant:
        memory_context = "\n".join(
            f"- ({item['topic']}) {item['content']}" for item in relevant
        )
    else:
        memory_context = "No relevant memories found."

    system_prompt = f"""You are Claude Agent, running on Ben's personal server.

Existing memory topics: {topics_text}

Relevant memories for this message:
{memory_context}

Use the relevant memories above to inform your response if they're
actually useful, otherwise ignore them.

Respond to the user's message. Then decide if this exchange is
worth remembering long-term, and if so, which topic it belongs to.

Topics should represent genuinely distinct subject areas, not a
single catch-all. For example, "watch hardware" and "server setup"
are different topics even though both relate to the same overall
project. Only reuse an existing topic if the new content is truly
about the same specific subject, not just loosely related. When in
doubt, prefer creating a new, clearly-named topic over stuffing
content into a topic it doesn't precisely belong to.

Respond ONLY with valid JSON in this exact shape, nothing else
before or after it, no explanation outside the JSON:

{{
  "reply": "your actual response to the user goes here",
  "memory": {{
    "should_save": true or false,
    "content": "a condensed summary worth remembering, or empty string if should_save is false",
    "topic": "the topic name",
    "is_new_topic": true or false
  }}
}}"""

    options = ClaudeAgentOptions(system_prompt=system_prompt)

    full_response = ""
    async for message in query(prompt=user_message, options=options):
        if hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text"):
                    full_response += block.text

    return full_response

def extract_json(raw):
    # First, look for a fenced ```json ... ``` block anywhere in the text
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        return fenced.group(1)

    # Otherwise, fall back to grabbing everything from the first { to the last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]

    return raw

def handle_message(user_message):
    raw = asyncio.run(ask_claude_agent(user_message))
    candidate = extract_json(raw)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        print("Warning: Claude didn't return clean JSON. Raw output:")
        print(raw)
        return None

    print(f"\nClaude Agent: {parsed['reply']}\n")

    mem = parsed.get("memory", {})
    if mem.get("should_save"):
        save_memory(content=mem["content"], topic=mem["topic"])

    return parsed

if __name__ == "__main__":
    handle_message("Hey, just testing the connection. Can you confirm you're working?")
