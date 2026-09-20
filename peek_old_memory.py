import chromadb

client = chromadb.PersistentClient(path="/home/linuxuser/akagishiragt-chroma-backup/data")

for name in ["sessions", "codebase"]:
    col = client.get_collection(name)
    print(f"\n=== Sample from '{name}' ===")
    sample = col.peek(limit=1)
    print("Document text:")
    print(sample["documents"][0][:500])
    print("\nMetadata:")
    print(sample["metadatas"][0])
