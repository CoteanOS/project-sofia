# memory.py - long-term memory via embeddings + Chroma
import pathlib
import chromadb
from llm import embed

DB = str(pathlib.Path(__file__).resolve().parent / "memory_db")
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
    # Quick read-only check. Does NOT write anything.
    # To store a fact:  python -c "from memory import ingest; ingest('...', source='facts')"
    import sys
    q = " ".join(sys.argv[1:]) or "what do you remember?"
    print(retrieve(q))
