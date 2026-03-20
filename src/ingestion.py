from src.chunking import chunk_documents
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore

def ingest_research_papers(raw_docs, v_store:VectorStore, e_manager:EmbeddingManager):
    """
    The main engine that turns uploaded PDFs into searchable math.
    Called directly by app.py when a user uploads files.
    """
    print("--- Starting Ingestion ---")

    # 1. Chunk the documents (using the new 1000/150 settings)
    new_chunks = chunk_documents(raw_docs)

    # 2. Extract text for embeddings
    texts = [c.page_content for c in new_chunks]

    # 3. Add to Vector Store
    # This automatically generates embeddings and saves to Chroma
    print(f"Embedding and saving {len(new_chunks)} chunks...")
    v_store.add_documents(new_chunks, e_manager)
    
    print(f"Success! Total chunks in DB: {v_store.collection.count()}")
    return len(new_chunks)
