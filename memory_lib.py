import chromadb
import uuid
from datetime import datetime, timezone

client = chromadb.PersistentClient(path="/home/linuxuser/hub_memory")
collection = client.get_or_create_collection(name="conversations")

old_client = chromadb.PersistentClient(path="/home/linuxuser/akagishiragt-chroma-backup/data")
old_sessions = old_client.get_collection("sessions")
old_codebase = old_client.get_collection("codebase")

def get_existing_topics():
    all_entries = collection.get(include=["metadatas"])
    topics = set()
    for metadata in all_entries["metadatas"]:
        if "topic" in metadata:
            topics.add(metadata["topic"])
    return sorted(topics)

def save_memory(content, topic, parent_conversation=None, chunk_index=None, source_type="conversation"):
    entry_id = str(uuid.uuid4())

    metadata = {
        "topic": topic,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_type": source_type
    }
    if parent_conversation is not None:
        metadata["parent_conversation"] = parent_conversation
    if chunk_index is not None:
        metadata["chunk_index"] = chunk_index

    collection.add(
        documents=[content],
        ids=[entry_id],
        metadatas=[metadata]
    )

    print(f"Saved to topic '{topic}' ({source_type}): {content[:60]}...")

def search_memory(query_text, n_results=3):
    found = []

    results = collection.query(query_texts=[query_text], n_results=n_results)
    if results["documents"][0]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            found.append({
                "content": doc,
                "topic": meta.get("topic", "unknown"),
                "source": "conversations",
                "source_type": meta.get("source_type", "conversation"),
                "timestamp": meta.get("timestamp")
            })

    session_results = old_sessions.query(query_texts=[query_text], n_results=2)
    if session_results["documents"][0]:
        for doc in session_results["documents"][0]:
            found.append({"content": doc, "topic": "dev history", "source": "old sessions", "source_type": "conversation", "timestamp": None})

    code_results = old_codebase.query(query_texts=[query_text], n_results=2)
    if code_results["documents"][0]:
        for doc, meta in zip(code_results["documents"][0], code_results["metadatas"][0]):
            filepath = meta.get("filepath", "unknown file")
            found.append({"content": doc, "topic": f"code: {filepath}", "source": "old codebase", "source_type": "conversation", "timestamp": None})

    return found

def chunk_text(text, chunk_size=20000, overlap=5000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks

def chunk_turns(turns, pad_chars=6000):
    chunks = []
    for i, turn in enumerate(turns):
        if i == 0:
            chunk = turn
        else:
            previous_turn = turns[i - 1]
            padding = previous_turn[-pad_chars:]
            chunk = padding + "\n\n[...continuing...]\n\n" + turn
        chunks.append(chunk)
    return chunks

def list_memories_in_topic(topic):
    results = collection.get(where={"topic": topic}, include=["documents", "metadatas"])
    for i, doc_id in enumerate(results["ids"]):
        print(f"[{doc_id}] {results['documents'][i][:80]}...")

def delete_memory(entry_id):
    collection.delete(ids=[entry_id])
    print(f"Deleted entry {entry_id}")
