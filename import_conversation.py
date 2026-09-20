import json

def extract_text(content_blocks):
    text_parts = []
    for block in content_blocks:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    return "".join(text_parts)

def get_true_conversation_path(data):
    """Walk backward from the actual final message using
    parent_message_uuid links, to get only the real conversation
    path and skip any edited-away or regenerated branches."""
    messages_by_id = {msg["uuid"]: msg for msg in data.get("chat_messages", [])}
    leaf_id = data.get("current_leaf_message_uuid")

    path = []
    current_id = leaf_id
    while current_id and current_id in messages_by_id:
        msg = messages_by_id[current_id]
        path.append(msg)
        current_id = msg.get("parent_message_uuid")

    path.reverse()
    return path

def load_conversation_turns(filepath):
    with open(filepath) as f:
        data = json.load(f)

    conversation_name = data.get("name", "unnamed conversation")
    messages = get_true_conversation_path(data)

    turns = []
    pending_human = None

    for msg in messages:
        sender = msg.get("sender")
        text = extract_text(msg.get("content", []))
        timestamp = msg.get("created_at")

        if not text.strip():
            continue

        if sender == "human":
            pending_human = {"text": text, "timestamp": timestamp}

        elif sender == "assistant" and pending_human:
            turn_text = f"User ({pending_human['timestamp']}): {pending_human['text']}\n\nClaudia ({timestamp}): {text}"
            turns.append(turn_text)
            pending_human = None

    return conversation_name, turns

if __name__ == "__main__":
    name, turns = load_conversation_turns("/home/linuxuser/import_test.json")
    print(f"Conversation: {name}")
    print(f"Found {len(turns)} turn(s)\n")
    for i, t in enumerate(turns):
        print(f"--- Turn {i} ---")
        print(t[:300])
        print()
