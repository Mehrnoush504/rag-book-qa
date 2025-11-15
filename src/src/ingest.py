import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document


def get_embeddings():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY missing")
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=key)


def load_and_split_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = splitter.split_documents(docs)

    for i, c in enumerate(chunks):
        if not c.metadata:
            c.metadata = {}
        c.metadata["source"] = os.path.basename(pdf_path)
        c.metadata["chunk_id"] = i

    print(f"[ingest] Split into {len(chunks)} chunks")
    return chunks


def create_qdrant_from_documents(documents: List[Document], collection_name: str = "pdf_documents", force_recreate: bool = False):
    qdrant_url = os.getenv("QDRANT_URL")
    embeddings = get_embeddings()

    qdrant = Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        url=qdrant_url,
        prefer_grpc=False,
        force_recreate=force_recreate,
    )

    print(f"[ingest] Stored {len(documents)} docs in Qdrant collection '{collection_name}'")
    return qdrant


def ingest_pdf_to_qdrant(pdf_path: str, collection_name: str = "pdf_documents", force_recreate: bool = False):
    chunks = load_and_split_pdf(pdf_path)
    return create_qdrant_from_documents(chunks, collection_name=collection_name, force_recreate=force_recreate)
