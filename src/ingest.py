import os
import tempfile
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document  # CHANGED: from langchain.schema import Document

class PDFIngestor:
    def __init__(self):
        # Get OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load PDF and split into chunks"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Add metadata for tracking
            for i, doc in enumerate(documents):
                doc.metadata["chunk_id"] = i
                doc.metadata["source"] = os.path.basename(pdf_path)
            
            chunks = self.text_splitter.split_documents(documents)
            print(f"Split {len(documents)} pages into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            print(f"Error loading PDF: {e}")
            raise
    
    def create_vector_store(self, documents: List[Document], collection_name: str = "pdf_documents") -> Qdrant:
        """Create Qdrant vector store from documents"""
        
        # Get Qdrant URL from environment with fallback
        qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        print(f"Using Qdrant URL: {qdrant_url}")
        
        vector_store = Qdrant.from_documents(
            documents=documents,
            embedding=self.embeddings,
            url=qdrant_url,
            prefer_grpc=False,
            collection_name=collection_name,
            force_recreate=True
        )
        
        print(f"Created vector store with {len(documents)} documents in collection '{collection_name}'")
        return vector_store

def process_pdf(pdf_path: str, collection_name: str = "pdf_documents") -> Qdrant:
    """Main function to process PDF and create vector store"""
    ingestor = PDFIngestor()
    documents = ingestor.load_pdf(pdf_path)
    vector_store = ingestor.create_vector_store(documents, collection_name)
    return vector_store