## **🔍 Research Paper Question Answering System Using RAG**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://research-paper-app-rag-yszz5njyeefut2sfe2nrze.streamlit.app/)

> **Project Link:** [Click here to access research assistant](https://research-paper-app-rag-yszz5njyeefut2sfe2nrze.streamlit.app/)

---

### 📖 **Project Overview**
Research papers are often lengthy and technical, making it time-consuming for students and faculty to locate specific information. This project provides an **AI-powered solution** that allows users to upload **PDF research papers** and ask questions in natural language. 

Instead of reading the entire document, the system identifies the **most relevant sections** and generates precise, grounded answers.

---

### **Target Users:**
* **Undergraduate & Postgraduate Students**
* **PhD Scholars & Faculty Members**
* **Researchers** needing quick insights into methodology, datasets, or findings.

---

### 🛠️ **Technical Stack**
* **Interface:** Streamlit
* **PDF Parsing:** `PyMuPDFLoader` (LangChain Community)
* **Orchestration:** `langchain`, `langchain-core`, `langchain-community`
* **Embeddings:** `sentence-transformers`
* **Vector Database:** `ChromaDB`
* **LLM Integration:** `langchain-groq` (using **Groq** for high-speed inference)
* **Environment Management:** `python-dotenv`

---

### 🏗️ **System Architecture & Workflow**
The system follows a standard **RAG (Retrieval-Augmented Generation)** pipeline to handle long documents efficiently:

1. **Document Ingestion:** PDFs are loaded using `PyMuPDFLoader`.
2. **Text Chunking:** Documents are split into smaller segments (**chunks**) to stay within the LLM's context limits.
3. **Embedding Generation:** Chunks are converted into vector representations using `sentence-transformers`.
4. **Vector Storage:** Embeddings are stored in **ChromaDB** for similarity searching.
5. **Retrieval:** When a user asks a question, the system retrieves the **Top-K** most relevant chunks.
6. **Augmentation & Generation:** The retrieved context is sent to the LLM (via **Groq**) to generate an answer grounded strictly in the document text.

---

### 🚀 **Core Features & Source Transparency**
The system is designed for academic precision, ensuring every answer is backed by verifiable data from the uploaded document. For every query, the chatbot provides:

* **Top-5 Semantic Retrieval:** The engine identifies the five most relevant text segments from the research paper to ensure a comprehensive answer.
* **Precise Page Attribution:** Every retrieved chunk is tagged with its original **Page Number** from the PDF, allowing the user to verify the source instantly.
* **Similarity Relevance Scores:** The system displays a numerical **Relevance Score** for each source, showing the mathematical confidence of the search.
* **Grounded Generation:** The LLM is strictly constrained to the retrieved context, ensuring that every response is evidence-based and free from AI hallucinations.

---

### 🔍 **Key Challenges Solved**

* **⚡ Overcoming Context Limits:** Research papers are often too long for a single LLM prompt. By using **RAG**, we only send the relevant portions, staying within the model's token limits.
* **🛡️ Reducing Hallucinations:** Traditional AI might "make up" answers. This system is **grounded strictly in the PDF text**, ensuring it only answers based on provided evidence.
* **⏱️ Research Efficiency:** Users can extract specific **methodologies** or **results** in seconds, eliminating the need for hours of manual skimming and reading.

---

### 🚀 **How to Run (Local Development)**

 **1. Install Requirements**
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

 **2. Set Environment Variables**
Create a file named `.env` in the root directory and add your Groq API key:
```plaintext
GROQ_API_KEY=your_actual_key_here
```

**3. Start the App**
Launch the interface with the following command:
```bash
streamlit run app.py
```

