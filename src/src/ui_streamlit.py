import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG PDF QA", page_icon="📚")
st.title("📚 RAG PDF Question Answering")

if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found in environment variables")
    st.stop()

from ingest import ingest_pdf_to_qdrant, load_and_split_pdf
from rag_chain import query_qa_chain

st.header("1) Upload PDF")
uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    st.success(f"Uploaded: {uploaded.name}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(uploaded.getvalue())
        tmp_path = tf.name

    st.write(f"Saved to: `{tmp_path}`")

    if st.button("Ingest PDF into Qdrant (recreate collection)"):
        with st.spinner("Ingesting PDF..."):
            try:
                ingest_pdf_to_qdrant(tmp_path, collection_name="pdf_documents", force_recreate=True)
                st.success("Ingested into Qdrant ✅")
            except Exception as e:
                st.error(f"Ingest failed: {e}")

    st.header("2) Ask a question")
    question = st.text_input("Question about the uploaded PDF:")

    if question:
        with st.spinner("Querying RAG..."):
            try:
                answer, sources = query_qa_chain(question, collection_name="pdf_documents", k=3)
                st.markdown("### Answer")
                st.write(answer)

                st.markdown("### Source passages")
                for i, doc in enumerate(sources):
                    st.write(f"**Source #{i+1} — metadata:** {doc.metadata}")
                    st.write(doc.page_content)
            except Exception as e:
                st.error(f"Query failed: {e}")

else:
    st.info("Upload a PDF to begin.")
