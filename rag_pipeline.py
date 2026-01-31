from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import settings
import os
import uuid


class RAGPipeline:
    """Retrieval Augmented Generation pipeline for context-based responses"""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.text_splitter = None
        self.session_id = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAG components"""
        # Initialize embeddings (using free HuggingFace embeddings)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_documents(self, texts: List[str], metadata: Optional[List[dict]] = None) -> str:
        """
        Process documents and create vector store
        
        Args:
            texts: List of text content from files
            metadata: Optional metadata for each text
            
        Returns:
            Session ID for this vector store
        """
        # Create documents
        documents = []
        for i, text in enumerate(texts):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            documents.append(Document(page_content=text, metadata=meta))
        
        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(documents)
        
        if not split_docs:
            raise ValueError("No content extracted from documents")
        
        # Create vector store
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        
        # Generate session ID
        self.session_id = str(uuid.uuid4())
        
        return self.session_id
    
    def get_relevant_context(self, query: str, k: int = 4) -> List[Document]:
        """
        Get relevant context for a query
        
        Args:
            query: User query
            k: Number of relevant documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            raise ValueError("No documents processed. Please upload files first.")
        
        relevant_docs = self.vector_store.similarity_search(query, k=k)
        return relevant_docs
    
    def create_qa_chain(self, llm, retriever_k: int = 4, chat_history: list = None):
        """
        Create a conversational QA chain with retrieval using LCEL
        
        Args:
            llm: Language model instance
            retriever_k: Number of documents to retrieve
            chat_history: Previous conversation messages
            
        Returns:
            LCEL chain
        """
        if not self.vector_store:
            raise ValueError("No documents processed. Please upload files first.")
        
        # Build conversation history for context
        history_text = ""
        if chat_history and len(chat_history) > 0:
            history_messages = []
            for msg in chat_history[-6:]:  # Last 3 exchanges (6 messages)
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_messages.append(f"{role}: {msg.get('content')}")
            history_text = "\n".join(history_messages)
        
        # Create conversational prompt
        if history_text:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a friendly, intelligent AI assistant having a natural conversation with the user about their uploaded documents.

Your conversational guidelines:
- For casual greetings (hi, hello, how are you, etc.): Respond warmly and naturally
- For questions about the documents: Use ONLY the provided context to answer
- Be warm, helpful, and engaging in all responses
- Remember and reference previous parts of the conversation when relevant
- Provide clear, well-structured answers with good formatting
- If a factual question can't be answered from the documents, politely say so
- Maintain context awareness throughout the conversation

You can have natural small talk, but for any factual questions, strictly use only the document context."""),
                ("human", """Previous conversation:
{history}

Context from the uploaded documents:
{context}

Current question: {question}

Please provide a natural, conversational response.""")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a friendly, intelligent AI assistant helping users understand information from their uploaded documents.

Your conversational guidelines:
- For casual greetings (hi, hello, how are you, etc.): Respond warmly and naturally
- For questions about the documents: Use ONLY the provided context to answer
- Be warm, helpful, and engaging in your responses
- Provide clear, well-structured answers with good formatting
- If a factual question can't be answered from the documents, politely say so
- Use bullet points, numbering, or paragraphs as appropriate for clarity

You can have natural small talk, but for any factual questions, strictly use only the document context."""),
                ("human", """Context from the uploaded documents:
{context}

Question: {question}

Please provide a natural, conversational response.""")
            ])
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": retriever_k}
        )
        
        # Helper function to format documents
        def format_docs(docs):
            return "\n\n---\n\n".join(doc.page_content for doc in docs)
        
        # Create LCEL chain with or without history
        if history_text:
            chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough(),
                    "history": lambda x: history_text
                }
                | prompt
                | llm
                | StrOutputParser()
            )
        else:
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
        
        return chain, retriever
    
    def query(self, llm, question: str, retriever_k: int = 4, chat_history: list = None) -> dict:
        """
        Query the conversational RAG system
        
        Args:
            llm: Language model instance
            question: User question
            retriever_k: Number of documents to retrieve
            chat_history: Previous conversation messages
            
        Returns:
            Dictionary with answer and source documents
        """
        chain, retriever = self.create_qa_chain(llm, retriever_k, chat_history)
        
        # Get answer using LCEL chain
        answer = chain.invoke(question)
        
        # Get source documents separately using invoke
        source_docs = retriever.invoke(question)
        
        return {
            "answer": answer,
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in source_docs
            ]
        }
    
    def clear(self):
        """Clear the current vector store"""
        self.vector_store = None
        self.session_id = None
