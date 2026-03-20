import streamlit as st
import os
import time
from src.retriever import RAGRetriever
from src.vector_store import VectorStore
from src.embeddings import EmbeddingManager
from src.llm import llm
from src.rag_pipeline import rag_advanced
from src.data_loader import process_uploaded_file
from src.ingestion import ingest_research_papers

# --- 1. Page Configuration ---
st.set_page_config(page_title="Academic AI Assistant", layout="wide", page_icon="📚")

# --- 2. GLOBAL LOADING STATE ---
with st.spinner("Initializing Academic Research Engine..."):
    @st.cache_resource
    def init_system():
        # Initialize core components
        v_store = VectorStore()
        e_manager = EmbeddingManager()
        # The retriever now handles the try/except logic internally
        return RAGRetriever(v_store, e_manager)

    retriever = init_system()
    time.sleep(0.5) 

st.title("Academic Research Chatbot")

# --- Sidebar: Document Management & Status ---
with st.sidebar:
    st.header("Control Panel")
    
    # --- 3. DATABASE STATUS ---
    try:
        # Check chunk count directly from Chroma
        count = retriever.vector_store.collection.count()
        
        if count == 0:
            st.warning("⚠️ Database is empty.")
            st.info("💡 **Step 1:** Upload PDFs\n\n**Step 2:** Click 'Ingest Papers'")
        else:
            st.success(f"✅ Knowledge Base Active ({count} chunks)")
    except Exception:
        st.warning("⚠️ Database connection pending.")

    st.divider()
    
    # Upload Section
    files = st.file_uploader("Upload Research Papers", type="pdf", accept_multiple_files=True)
    
    col1, col2 = st.columns(2)
    
    if col1.button("Ingest Papers"):
        if files:
            raw_docs = []
            with st.spinner("Processing documents, generating embeddings..."):
                for f in files:
                    # Extracts text and original metadata from PDF
                    raw_docs.extend(process_uploaded_file(f))
                
                # The clean ingestion (no identity headers)
                ingest_research_papers(raw_docs, retriever.vector_store, retriever.embedding_manager)
                st.success(f"Successfully ingested {len(files)} paper(s)!")
                st.rerun() 
        else:
            st.error("Please drop PDFs first!")

    if col2.button("Clear DB"):
        # Wipes the Chroma collection
        retriever.vector_store.clear_collection()
        st.session_state.messages = [] 
        st.warning("Database wiped.")
        st.rerun()

# --- 4. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display sources attached to historical messages
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📝 Cited Sources"):
                for s in msg["sources"]:
                    c1, c2 = st.columns([3, 1])
                    with c1: st.markdown(f"📄 **{s['source']}**")
                    # NO +1 or +10 math here
                    with c2: st.markdown(f"📍 **Page {s['page']}**") 
                    
                    if "score" in s:
                        # Convert 0.519 to 51.9%
                        display_score = s['score'] * 100
                        st.caption(f"Relevance: {display_score:.1f}%")
                    st.divider()

# --- 5. User Query Input ---
if prompt := st.chat_input("Ask a question about the papers..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing research context..."):
            # Call the pipeline with the 0.25 similarity threshold
            result = rag_advanced(prompt, retriever, llm)
            st.markdown(result["answer"])
            
            # Display sources for the CURRENT response
            if result["sources"]:
                with st.expander("📝 Cited Sources"):
                    for s_new in result["sources"]:
                        c1, c2 = st.columns([3, 1])
                        with c1: st.markdown(f"📄 **{s_new['source']}**")
                        # Raw page number from metadata
                        with c2: st.markdown(f"📍 **Page {s_new['page']}**")
                        
                        if "score" in s_new:
                            # Clean percentage calculation
                            pct = s_new['score'] * 100
                            st.caption(f"Relevance: {pct:.1f}%")
                        st.divider()
                        
    # Save assistant response to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": result["answer"],
        "sources": result["sources"] 
    })
