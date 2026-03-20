Research Paper Question Answering System
Using Retrieval-Augmented Generation (RAG)
📌 Project Overview
Research papers are often lengthy and technical, making it time-consuming for students and faculty to locate specific information. This project provides an AI-powered solution that allows users to upload PDF research papers and ask questions in natural language. Instead of reading the entire document, the system identifies the most relevant sections and generates precise, grounded answers.

Target Users: * Undergraduate & Postgraduate Students

PhD Scholars & Faculty Members

Researchers needing quick insights into methodology, datasets, or findings.

🛠️ Technical Stack
Interface: Streamlit

PDF Parsing: PyMuPDFLoader (LangChain Community)

Orchestration: langchain, langchain-core, langchain-community

Embeddings: sentence-transformers

Vector Database: ChromaDB (or FAISS-cpu)

LLM Integration: langchain-groq (using Groq for high-speed inference)

Environment Management: python-dotenv

🏗️ System Architecture & Workflow
The system follows a standard RAG (Retrieval-Augmented Generation) pipeline to handle long documents efficiently:

Document Ingestion: PDFs are loaded using PyMuPDFLoader.

Text Chunking: Documents are split into smaller segments (chunks) to stay within the LLM's context limits.

Embedding Generation: Chunks are converted into vector representations using sentence-transformers.

Vector Storage: Embeddings are stored in ChromaDB for similarity searching.

Retrieval: When a user asks a question, the system retrieves the Top-K most relevant chunks.

Augmentation & Generation: The retrieved context is sent to the LLM (via Groq) to generate an answer grounded strictly in the document text.

📂 Project Structure
Plaintext
├── app.py                # Streamlit UI & Session Management
├── src/
│   ├── ingestion.py      # Logic for processing and embedding PDFs
│   ├── chunking.py       # Text splitting configurations
│   ├── rag_pipeline.py   # RAG logic and Prompt Engineering
│   ├── retriever.py      # Vector Store search interface
│   ├── vector_store.py   # Database setup (Chroma/FAISS)
│   ├── embeddings.py     # Sentence-transformer configuration
│   ├── llm.py            # LangChain-Groq model initialization
│   └── data_loader.py    # PDF text extraction utilities
├── requirements.txt      # List of dependencies
└── .env                  # API Keys (GROQ_API_KEY)
🔍 Key Challenges Solved
Context Limits: Solves the issue of research papers being too long for direct LLM input.

Accuracy & Hallucinations: By using RAG, the AI is forced to answer based only on the provided text, reducing "made-up" information.

Efficiency: Users get specific answers about methodology or results in seconds without manual reading.

🏁 How to Run
Install Requirements:

Bash
pip install -r requirements.txt
Set Environment Variables:
Create a .env file and add your GROQ_API_KEY.

Start the App:

Bash
streamlit run app.py
