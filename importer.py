import asyncio
import json
import re
from claude_agent_sdk import query, ClaudeAgentOptions
from import_conversation import load_conversation_turns
from memory_lib import get_existing_topics, chunk_turns, save_memory

def extract_json(raw):
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw

async def decide_topic(conversation_name, sample_text):
    topics = get_existing_topics()
    topics_text = ", ".join(topics) if topics else "none yet"

    prompt = f"""A conversation titled "{conversation_name}" is being imported into memory.

Existing topics: {topics_text}

Sample of the conversation content:
{sample_text[:2000]}

Decide the best topic tag for this content. Reuse an existing topic
only if this content is genuinely about that same subject, even
though the title differs. Otherwise, propose a new topic name,
using the conversation's own title as a strong guide since Ben
names his chats meaningfully.

Respond ONLY with valid JSON, nothing else:
{{"topic": "the topic name", "is_new_topic": true or false}}"""

    options = ClaudeAgentOptions()
    full_response = ""
    async for message in query(prompt=prompt, options=options):
        if hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text"):
                    full_response += block.text

    candidate = extract_json(full_response)
    try:
        parsed = json.loads(candidate)
        return parsed["topic"]
    except (json.JSONDecodeError, KeyError):
        return conversation_name

def process_import(filepath):
    name, turns = load_conversation_turns(filepath)

    if not turns:
        return f"No turns found in '{name}', nothing imported."

    chunks = chunk_turns(turns)
    topic = asyncio.run(decide_topic(name, turns[0]))

    for i, chunk in enumerate(chunks):
        save_memory(
            content=chunk,
            topic=topic,
            parent_conversation=name,
            chunk_index=i
        )

    return f"Imported '{name}' as topic '{topic}': {len(chunks)} chunk(s) saved."
