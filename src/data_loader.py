import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader

def process_uploaded_file(uploaded_file):
    """
    Takes a Streamlit UploadedFile object, saves it to a temp path, 
    and loads it using PyMuPDF.
    """
    # Create a temporary file to store the upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    try:
        # Load the PDF from the temp path
        loader = PyMuPDFLoader(temp_path)
        raw_docs = loader.load()
        
        # Metadata Fix: Human-readable page numbers and Filename
        for i, doc in enumerate(raw_docs):
            doc.metadata["page"] = i + 1  
            doc.metadata["source"] = uploaded_file.name
            
        return raw_docs
    finally:
        # Clean up the temp file after loading
        if os.path.exists(temp_path):
            os.remove(temp_path)