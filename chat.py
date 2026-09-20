import asyncio
import json
import re
from datetime import datetime, timezone
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from memory_lib import get_existing_topics, save_memory, search_memory

with open("/home/linuxuser/claudia_persona.txt") as f:
    PERSONA = f.read()

SYSTEM_PROMPT = f"""{PERSONA}

You are running on Ben's personal server (pserv-1).

Each user message will come with extra context attached: a list of
existing memory topics, and any relevant past memories found for
that specific message, including how old each one is and whether
it came from a web search. Use relevant memories if they're
actually useful, otherwise ignore them.

You have access to WebSearch and WebFetch tools. Use them whenever
a question genuinely needs current or real-world information you
wouldn't already know.

After responding, decide if this exchange is worth remembering
long-term, and if so, which topic it belongs to. Topics should
represent genuinely distinct subject areas, not a single catch-all.
Only reuse an existing topic if the new content is truly about the
same specific subject. When in doubt, create a new, clearly-named
topic rather than stuffing content into one it doesn't precisely
belong to.

Respond ONLY with valid JSON in this exact shape, nothing else
before or after it, no explanation outside the JSON:

{{
  "reply": "your actual spoken response, no URLs or citations here",
  "sources": ["url1", "url2"],
  "memory": {{
    "should_save": true or false,
    "content": "a condensed summary worth remembering, or empty string if should_save is false",
    "topic": "the topic name",
    "is_new_topic": true or false
  }}
}}

If there are no sources, use an empty list for "sources"."""

def extract_json(raw):
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw

def age_description(timestamp_str):
    if not timestamp_str:
        return ""
    try:
        then = datetime.fromisoformat(timestamp_str)
        now = datetime.now(timezone.utc)
        days = (now - then).days
        if days == 0:
            return " [from today]"
        elif days == 1:
            return " [from 1 day ago]"
        else:
            return f" [from {days} days ago]"
    except ValueError:
        return ""

async def chat_loop():
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["WebSearch", "WebFetch"]
    )

    async with ClaudeSDKClient(options=options) as client:
        print("Claudia is ready. Type 'exit' to end the session.\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ("exit", "quit"):
                print("Ending session.")
                break
            if not user_input:
                continue

            topics = get_existing_topics()
            topics_text = ", ".join(topics) if topics else "none yet"

            relevant = search_memory(user_input)
            if relevant:
                lines = []
                for item in relevant:
                    tag = " (web result)" if item["source_type"] == "web_search" else ""
                    age = age_description(item.get("timestamp"))
                    lines.append(f"- ({item['topic']}){tag}{age} {item['content']}")
                memory_context = "\n".join(lines)
            else:
                memory_context = "No relevant memories found."

            augmented_message = f"""[Existing topics: {topics_text}]
[Relevant memories:
{memory_context}]

User message: {user_input}"""

            await client.query(augmented_message)

            full_response = ""
            used_web = False

            async for message in client.receive_response():
                if hasattr(message, "content"):
                    for block in message.content:
                        if hasattr(block, "text"):
                            full_response += block.text
                        if hasattr(block, "name") and block.name in ("WebSearch", "WebFetch"):
                            used_web = True

            candidate = extract_json(full_response)

            try:
                parsed = json.loads(candidate)
                reply = parsed.get("reply", full_response)
                sources = parsed.get("sources", [])
                mem = parsed.get("memory", {})
            except json.JSONDecodeError:
                # Fallback: JSON parsing failed, just show the raw
                # reply rather than dumping a scary warning message
                reply = full_response
                sources = []
                mem = {}

            print(f"\nClaudia: {reply}\n")

            if sources:
                print("Sources:")
                for s in sources:
                    print(f"  - {s}")
                print()

            if mem.get("should_save"):
                source_type = "web_search" if used_web else "conversation"
                save_memory(
                    content=mem["content"],
                    topic=mem["topic"],
                    source_type=source_type
                )

if __name__ == "__main__":
    asyncio.run(chat_loop())
