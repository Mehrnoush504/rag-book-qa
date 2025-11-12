import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Simple page config
st.set_page_config(page_title="RAG PDF QA", page_icon="📚")

# Title
st.title("📚 RAG PDF Question Answering")

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found in .env file")
    st.stop()

# File upload section
st.header("1. Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.success(f"✅ PDF uploaded: {uploaded_file.name}")
    
    # Show file info
    file_size = len(uploaded_file.getvalue()) / 1024  # KB
    st.write(f"File size: {file_size:.1f} KB")
    
    # Simple question input
    st.header("2. Ask Questions")
    question = st.text_input("Enter your question about the PDF:")
    
    if question:
        st.write(f"**Your question:** {question}")
        
        # Simulate processing (replace with actual RAG later)
        with st.spinner("Processing your question..."):
            # For now, just show a mock response
            st.success("**Answer:** This is a mock response. The RAG system would provide the real answer here.")
            
            # Show mock source documents
            with st.expander("View source documents"):
                st.write("**Source 1 (Page 3):**")
                st.write("This is where the actual PDF content would appear as evidence for the answer.")
                
                st.write("**Source 2 (Page 7):**")
                st.write("Another relevant passage from the PDF that supports the answer.")

else:
    st.info("👆 Please upload a PDF file to get started")

# Status section
st.header("3. System Status")
st.write("✅ Streamlit is working")
st.write("✅ OpenAI API key is set")

# Check if we can import the other modules
try:
    from ingest import PDFIngestor
    st.write("✅ PDF processing module loaded")
except ImportError as e:
    st.warning(f"⚠️ PDF processing module not available: {e}")

try:
    from rag_chain import RAGSystem
    st.write("✅ RAG chain module loaded")
except ImportError as e:
    st.warning(f"⚠️ RAG chain module not available: {e}")

# # Simple instructions
# st.header("Need Help?")
# st.write("""
# 1. Make sure you have a `.env` file with `OPENAI_API_KEY=your_key_here`
# 2. Install requirements: `pip install -r requirements.txt`
# 3. Run: `streamlit run src/ui_streamlit.py`
# """)