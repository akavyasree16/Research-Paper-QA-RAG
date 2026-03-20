# import os
# import uuid
# import chromadb
# import numpy as np
# from typing import List, Any

# class VectorStore:
#     """
#     Manages research paper embeddings in ChromaDB.
#     """

#     def __init__(
#         self,
#         collection_name: str = "research_papers",
#         persist_directory: str = "data/vector_store"
#     ):
#         self.collection_name = collection_name
#         self.persist_directory = persist_directory
#         self.client = None
#         self.collection = None
#         self._initialize_store()

#     def _initialize_store(self):
#         """Initialize ChromaDB client and collection"""
#         try:
#             os.makedirs(self.persist_directory, exist_ok=True)
#             self.client = chromadb.PersistentClient(path=self.persist_directory)
            
#             # Using Cosine Similarity as the distance metric
#             self.collection = self.client.get_or_create_collection(
#                 name=self.collection_name,
#                 metadata={"hnsw:space": "cosine"} 
#             )
#             print(f"Vector store ready. Current documents: {self.collection.count()}")
#         except Exception as e:
#             print(f"Error initializing vector store: {e}")
#             raise

#     def add_documents(self, documents: List[Any], embedding_manager: Any):
#         """
#         Takes LangChain documents, generates embeddings, and saves to Chroma.
#         """
#         if not documents:
#             return

#         texts = [doc.page_content for doc in documents]
#         # Generate embeddings using your manager
#         embeddings = embedding_manager.generate_embeddings(texts)
        
#         ids = [f"id_{uuid.uuid4().hex[:10]}" for _ in range(len(documents))]
#         metadatas = [dict(doc.metadata) for doc in documents]

#         try:
#             self.collection.add(
#                 ids=ids,
#                 embeddings=embeddings.tolist(),
#                 metadatas=metadatas,
#                 documents=texts
#             )
#             print(f"Added {len(documents)} chunks to the database.")
#         except Exception as e:
#             print(f"Error adding to Chroma: {e}")

#     def clear_collection(self):
#         """Wipes the database so you can start a fresh research session."""
#         try:
#             self.client.delete_collection(self.collection_name)
#             self.collection = self.client.create_collection(
#                 name=self.collection_name,
#                 metadata={"hnsw:space": "cosine"}
#             )
#             print("Database cleared for new session.")
#         except Exception as e:
#             print(f"Error clearing collection: {e}")

#     def query(self, query_embedding: np.ndarray, n_results: int = 5):
#         """Standard query method for the retriever."""
#         return self.collection.query(
#             query_embeddings=query_embedding.tolist(),
#             n_results=n_results
#         )
import os
import uuid
import chromadb
import numpy as np
from typing import List, Any

class VectorStore:
    """
    Manages research paper embeddings in an In-Memory (Ephemeral) ChromaDB.
    This ensures that each user session is private and data is not shared.
    """

    def __init__(
        self,
        collection_name: str = "research_papers"
    ):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB as an Ephemeral (In-Memory) client"""
        try:
            # SWITCHED: Using EphemeralClient instead of PersistentClient for privacy
            self.client = chromadb.EphemeralClient()
            
            # Using Cosine Similarity as the distance metric
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"} 
            )
            print(f"Private Vector Store initialized in memory.")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embedding_manager: Any):
        """
        Takes LangChain documents, generates embeddings, and saves to the in-memory collection.
        """
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings using your manager
        embeddings = embedding_manager.generate_embeddings(texts)
        
        # Create unique IDs for this session
        ids = [f"id_{uuid.uuid4().hex[:10]}" for _ in range(len(documents))]
        metadatas = [dict(doc.metadata) for doc in documents]

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                documents=texts
            )
            print(f"Added {len(documents)} chunks to the private session database.")
        except Exception as e:
            print(f"Error adding to Chroma: {e}")

    def clear_collection(self):
        """Wipes the session database for a fresh start."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print("Session database cleared.")
        except Exception as e:
            print(f"Error clearing collection: {e}")

    def query(self, query_embedding: np.ndarray, n_results: int = 5):
        """Standard query method for the retriever."""
        return self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )