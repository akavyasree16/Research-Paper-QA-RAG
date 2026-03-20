from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    """
    Split research papers into larger chunks to maintain 
    academic context and citation integrity.
    """
    # Increased size for dense research text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages.")
    return chunks