from import_conversation import load_conversation_turns
from memory_lib import chunk_turns, save_memory

FILEPATH = "/home/linuxuser/import_test.json"
TOPIC = "vibe coding philosophy"

name, turns = load_conversation_turns(FILEPATH)
print(f"Loaded '{name}' with {len(turns)} turn(s)")

chunks = chunk_turns(turns)
print(f"Split into {len(chunks)} chunk(s)\n")

for i, chunk in enumerate(chunks):
    save_memory(
        content=chunk,
        topic=TOPIC,
        parent_conversation=name,
        chunk_index=i
    )
