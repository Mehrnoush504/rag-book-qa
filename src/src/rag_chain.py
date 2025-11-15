import os
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from qdrant_client import QdrantClient


def get_openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY missing in environment")
    return key


def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=get_openai_api_key()
    )


def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=get_openai_api_key()
    )


def get_qdrant_vectorstore(collection_name: str = "pdf_documents", qdrant_url: str | None = None):
    if qdrant_url is None:
        qdrant_url = os.getenv("QDRANT_URL")

    embeddings = get_embeddings()
    client = QdrantClient(url=qdrant_url)

    return Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )


PROMPT_TEMPLATE = """
Use ONLY the provided context to answer the question.
If the answer is not in the context, say: "I don't know".

Context:
{context}

Question:
{input}

Answer:
"""


def build_rag_chain(collection_name: str = "pdf_documents", k: int = 3):
    llm = get_llm()
    vectorstore = get_qdrant_vectorstore(collection_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "input"]
    )

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain


def query_qa_chain(question: str, collection_name: str = "pdf_documents", k: int = 3) -> Tuple[str, List[Document]]:
    chain = build_rag_chain(collection_name=collection_name, k=k)

    # IMPORTANT FIX
    out = chain.invoke({"input": question})

    answer = out.get("answer") or out.get("result") or ""
    sources = out.get("source_documents") or []

    return answer, sources
