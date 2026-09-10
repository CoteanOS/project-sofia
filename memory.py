# memory.py — long-term memory via embeddings + Chroma
import pathlib
import chromadb
from llm import embed

DB = str(pathlib.Path("~/code/sofia/memory_db").expanduser())
client = chromadb.PersistentClient(path=DB)
col = client.get_or_create_collection("sofia")

def ingest(text, source="note"):
    """Chunk -> embed -> store. Call this on notes, docs, facts."""
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    if not chunks:
        return 0
    vectors = embed(chunks)
    ids = [f"{source}-{abs(hash(c))}" for c in chunks]
    col.add(ids=ids, documents=chunks, embeddings=vectors,
            metadatas=[{"source": source}] * len(chunks))
    return len(chunks)

def retrieve(query, k=4):
    """Embed the query, return the k most similar stored chunks."""
    qv = embed(query)[0]
    res = col.query(query_embeddings=[qv], n_results=k)
    docs = res.get("documents") or [[]]
    return docs[0]

if __name__ == "__main__":
    ingest("Sofia runs on an M-series MacBook.\n\nHer backbone model is gpt-oss:20b.", source="facts")
    print(retrieve("what hardware does Sofia run on?"))
