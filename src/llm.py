import os
import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# 1. Load local .env file (for local development only)
load_dotenv()

# 2. Smart Key Retrieval
# First, try to get it from Streamlit's secure Secrets vault
# Second, fall back to the environment variable (local .env)
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

# 3. Friendly Error Handling
if not groq_api_key:
    st.error("🔑 GROQ_API_KEY not found! Please add it to your .env or Streamlit Secrets.")
    st.stop() # Stops the app gracefully instead of crashing with a scary red error

# 4. Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1024
)