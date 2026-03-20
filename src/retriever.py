from typing import List, Dict, Any
from src.vector_store import VectorStore
from src.embeddings import EmbeddingManager

# class RAGRetriever:
#     """Handles query-based retrieval from the vector store"""

#     def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
#         # Type hints make this look very professional for your project
#         self.vector_store = vector_store
#         self.embedding_manager = embedding_manager

#     def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         print(f"--- Retrieving for: {query} ---")
        
#         try:
#             # 1. Generate embedding
#             query_embedding = self.embedding_manager.generate_embeddings([query])
            
#             # 2. Query the Vector Store (ChromaDB)
#             results = self.vector_store.collection.query(
#                 query_embeddings=query_embedding.tolist(),
#                 n_results=top_k
#             )

#             retrieved_docs = []
            
#             # 3. Extract results safely
#             if results and results["documents"] and len(results["documents"][0]) > 0:
#                 for i in range(len(results["documents"][0])):
#                     retrieved_docs.append({
#                         "content": results["documents"][0][i],
#                         "metadata": results["metadatas"][0][i],
#                         "distance": results["distances"][0][i]
#                     })
            
#             return retrieved_docs

#         except Exception as e:
#             print(f" Error during retrieval: {e}")
#             return []
def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        print(f"--- Retrieving for: {query} ---")
        
        try:
            # 1. Generate embedding
            # Note: Ensure this returns a numpy array or list correctly
            query_embedding = self.embedding_manager.generate_embeddings([query])
            
            # 2. CALL THE VECTOR STORE'S OWN QUERY METHOD
            # This is better because VectorStore handles the tolist() and logic internally
            results = self.vector_store.query(query_embedding, n_results=top_k)

            retrieved_docs = []
            
            # 3. Extract results safely (ChromaDB returns lists of lists)
            if results and "documents" in results and len(results["documents"]) > 0:
                # Chroma returns [ [doc1, doc2] ], so we access index 0
                for i in range(len(results["documents"][0])):
                    retrieved_docs.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                    })
            
            return retrieved_docs

        except Exception as e:
            print(f" Error during retrieval: {e}")
            return []