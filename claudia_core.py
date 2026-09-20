import json
import re
import asyncio
from datetime import datetime, timezone
from claude_agent_sdk import query, ClaudeAgentOptions
from memory_lib import get_existing_topics, save_memory, search_memory

with open("/home/linuxuser/claudia_persona.txt") as f:
    PERSONA = f.read()

with open("/home/linuxuser/voice_mode.txt") as f:
    VOICE_MODE_ADDON = f.read()

def build_system_prompt(mode):
    mode_addon = VOICE_MODE_ADDON if mode == "voice" else ""

    return f"""{PERSONA}

{mode_addon}

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
  "memory": {{
    "should_save": true or false,
    "content": "a condensed summary worth remembering, or empty string if should_save is false",
    "topic": "the topic name",
    "is_new_topic": true or false
  }}
}}"""

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

def find_json_array(text, start_idx):
    """Find the true end of a JSON array by tracking bracket
    depth, rather than assuming it runs to the end of the string."""
    depth = 0
    for i in range(start_idx, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[start_idx:i + 1]
    return None

def extract_sources_from_tool_result(content):
    sources = []
    try:
        marker = "Links: "
        idx = content.find(marker)
        if idx == -1:
            return sources

        array_start = idx + len(marker)
        links_json = find_json_array(content, array_start)
        if links_json is None:
            return sources

        links = json.loads(links_json)
        for link in links:
            url = link.get("url")
            if url:
                sources.append(url)
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return sources

async def _ask_claudia_async(user_message, mode):
    topics = get_existing_topics()
    topics_text = ", ".join(topics) if topics else "none yet"

    relevant = search_memory(user_message)
    if relevant:
        lines = []
        for item in relevant:
            tag = " (web result)" if item["source_type"] == "web_search" else ""
            age = age_description(item.get("timestamp"))
            lines.append(f"- ({item['topic']}){tag}{age} {item['content']}")
        memory_context = "\n".join(lines)
    else:
        memory_context = "No relevant memories found."

    full_prompt = f"""[Existing topics: {topics_text}]
[Relevant memories:
{memory_context}]

User message: {user_message}"""

    options = ClaudeAgentOptions(
        system_prompt=build_system_prompt(mode),
        allowed_tools=["WebSearch", "WebFetch"]
    )

    full_response = ""
    used_web = False
    all_sources = []

    async for message in query(prompt=full_prompt, options=options):
        if hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text"):
                    full_response += block.text

                if type(block).__name__ == "ToolUseBlock":
                    if block.name == "WebFetch":
                        url = block.input.get("url") if hasattr(block, "input") else None
                        if url:
                            used_web = True
                            all_sources.append(url)

                if type(block).__name__ == "ToolResultBlock":
                    content = block.content
                    if isinstance(content, str):
                        found = extract_sources_from_tool_result(content)
                        if found:
                            used_web = True
                            all_sources.extend(found)

    candidate = extract_json(full_response)

    try:
        parsed = json.loads(candidate)
        reply = parsed.get("reply", full_response)
        mem = parsed.get("memory", {})
    except json.JSONDecodeError:
        reply = full_response
        mem = {}

    unique_sources = list(dict.fromkeys(all_sources))

    if mem.get("should_save"):
        source_type = "web_search" if used_web else "conversation"
        save_memory(content=mem["content"], topic=mem["topic"], source_type=source_type)

    return {"reply": reply, "sources": unique_sources}

def ask_claudia(user_message, mode="text"):
    return asyncio.run(_ask_claudia_async(user_message, mode))
