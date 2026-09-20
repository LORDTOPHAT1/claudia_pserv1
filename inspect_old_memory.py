import chromadb

# Point directly at the backed-up AkagishiraGT database,
# not a new empty one this time
client = chromadb.PersistentClient(path="/home/linuxuser/akagishiragt-chroma-backup/data")

# List every collection that exists in this database
collections = client.list_collections()

print(f"Found {len(collections)} collection(s):\n")

for col in collections:
    count = col.count()
    print(f"- '{col.name}': {count} entries")
