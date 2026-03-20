def rag_advanced(query, retriever, llm):
    # 1. Fetch the top 5 matches
    results = retriever.retrieve(query, top_k=5)
    
    context_list = []
    sources = []
    
    for doc in results:
        # Distance to Similarity math
        # The most professional way to write it:
        similarity = max(0.0, 1 - doc.get("distance", 1.0))
        
        # WE USE 0.25 AND ROUND 3 
        if similarity > 0.25:
            context_list.append(doc["content"])
            
            # RAW PAGE LOGIC:
            sources.append({
                "source": doc["metadata"].get("source", "Unknown"),
                "page": doc["metadata"].get("page", 0), 
                "score": round(similarity, 3) 
            })

    context = "\n\n".join(context_list)
    
    # SIMPLE PROMPT: 
    prompt = f"""
    You are an academic research assistant. Use the context below to answer the question.
    If the context doesn't contain the answer, say you don't know.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {query}
    """
    
    response = llm.invoke(prompt)
    return {"answer": response.content, "sources": sources}