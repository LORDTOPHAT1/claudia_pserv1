import chromadb

client = chromadb.PersistentClient(path="./chroma_data")

collection = client.get_or_create_collection(name="test_memory")

collection.add(
    documents=["Ben's hub server is called pserv-1, hosted on Vultr in Sydney."],
    ids=["test1"]
)

results = collection.query(
    query_texts=["What is the name of Ben's server?"],
    n_results=1
)

print("Found this:")
print(results["documents"])
