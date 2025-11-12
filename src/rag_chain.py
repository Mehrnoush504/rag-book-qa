import os
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

try:
    # Try newer import style
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import Qdrant
    from langchain_core.documents import Document
    from langchain.chains import RetrievalQA
    from langchain_core.prompts import PromptTemplate
    print("✅ Using new LangChain imports")
except ImportError:
    try:
        # Fallback to older style
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.vectorstores import Qdrant
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        from langchain.schema import Document
        
        # Try different LLM imports
        try:
            from langchain_community.llms import OpenAI
        except ImportError:
            from langchain.llms import OpenAI
            
        print("✅ Using older LangChain imports")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise

class RAGSystem:
    def __init__(self, collection_name: str = "pdf_documents"):
        self.collection_name = collection_name
        
        # Get OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key
        )
        
        # Initialize LLM - handle different import styles
        try:
            # Try ChatOpenAI first (newer)
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                openai_api_key=openai_api_key
            )
        except (ImportError, NameError):
            # Fallback to OpenAI (older)
            self.llm = OpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.1,
                openai_api_key=openai_api_key
            )
        
        self.vector_store = self._initialize_vector_store()
        self.qa_chain = self._create_qa_chain()
    
    def _initialize_vector_store(self):
        """Initialize connection to Qdrant vector store"""
        qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        print(f"Connecting to Qdrant at: {qdrant_url}")
        
        return Qdrant(
            embedding_function=self.embeddings.embed_query,
            collection_name=self.collection_name,
            url=qdrant_url
        )
    
    def _create_qa_chain(self):
        """Create the QA chain"""
        prompt_template = """Answer the question based on the context below. If you don't know the answer, say so.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Create QA chain
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def query(self, question: str) -> Tuple[str, List[Document]]:
        """Query the RAG system"""
        try:
            result = self.qa_chain({"query": question})
            answer = result["result"]
            source_docs = result["source_documents"]
            return answer, source_docs
        except Exception as e:
            return f"Error: {str(e)}", []

_rag_system = None

def get_rag_system(collection_name: str = "pdf_documents"):
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem(collection_name)
    return _rag_system