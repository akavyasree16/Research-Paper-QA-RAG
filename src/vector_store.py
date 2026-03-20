import uuid
import chromadb
import numpy as np
from typing import List, Any

class VectorStore:
    """
    Manages private, in-memory research paper embeddings using unique UUIDs.
    Includes error handling to ensure the app doesn't crash during ingestion.
    """

    def __init__(self):
        self.client = None
        self.collection = None
        self.unique_id = uuid.uuid4().hex[:8]
        self.collection_name = f"research_{self.unique_id}"
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB as an Ephemeral (In-Memory) client with error handling."""
        try:
            self.client = chromadb.EphemeralClient()
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"DEBUG: Private session {self.collection_name} initialized.")
        except Exception as e:
            print(f"Failed to initialize Vector Store: {e}")
            raise # Re-raise so the app knows it can't start

    def add_documents(self, documents: List[Any], embedding_manager: Any):
        """Adds documents to the private collection with safety checks."""
        if not documents:
            return

        try:
            texts = [doc.page_content for doc in documents]
            embeddings = embedding_manager.generate_embeddings(texts)
            
            ids = [f"id_{uuid.uuid4().hex[:10]}" for _ in range(len(documents))]
            metadatas = [dict(doc.metadata) for doc in documents]

            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                documents=texts
            )
            print(f"Successfully added {len(documents)} chunks.")
        except Exception as e:
            print(f"Error during document ingestion: {e}")
            # We don't 'raise' here so the user can try uploading again

    def clear_collection(self):
        """Wipes the private collection safely."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Error clearing collection: {e}")

    def query(self, query_embedding: np.ndarray, n_results: int = 5):
        """Queries the private collection for relevant research segments."""
        try:
            return self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
        except Exception as e:
            print(f"Search error: {e}")
            return {"documents": [], "metadatas": [], "distances": []}