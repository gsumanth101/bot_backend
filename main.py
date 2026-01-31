from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import uuid
import shutil
from pathlib import Path

# Import local modules
from config import settings
from models import (
    FileUploadResponse, ChatRequest, ChatResponse,
    ModelsListResponse, ModelInfo, HealthResponse, ErrorResponse
)
from file_processor import FileProcessor
from llm_models import LLMModelManager
from rag_pipeline import RAGPipeline


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="ChatBot Backend with multiple LLM models and RAG capabilities"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
file_processor = FileProcessor()
model_manager = LLMModelManager()
rag_pipeline = RAGPipeline()

# Store uploaded files metadata
uploaded_files_data = {}
uploaded_files_texts = []


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to ChatBot Backend API",
        "version": settings.app_version,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        groq_configured=settings.groq_api_key is not None,
        huggingface_configured=settings.huggingfacehub_api_token is not None
    )


@app.get("/models", response_model=ModelsListResponse, tags=["Models"])
async def get_available_models():
    """Get list of available LLM models"""
    models_dict = model_manager.get_available_models()
    models_list = [
        ModelInfo(
            key=key,
            name=info["name"],
            provider=info["provider"],
            description=info["description"],
            max_tokens=info["max_tokens"]
        )
        for key, info in models_dict.items()
    ]
    
    return ModelsListResponse(
        models=models_list,
        current_model=model_manager.get_current_model_key()
    )


@app.post("/upload", response_model=List[FileUploadResponse], tags=["Files"])
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple files for processing
    
    Accepts: text files, PDFs, Word documents, images
    """
    global uploaded_files_data, uploaded_files_texts
    
    # Clear previous session
    uploaded_files_data = {}
    uploaded_files_texts = []
    
    responses = []
    
    try:
        for file in files:
            # Validate file extension
            if not file_processor.validate_file_extension(file.filename, settings.allowed_extensions):
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not allowed: {file.filename}. Allowed: {settings.allowed_extensions}"
                )
            
            # Read file into memory
            file_content = await file.read()
            file_size_mb = len(file_content) / (1024 * 1024)
            
            # Check file size
            if file_size_mb > settings.max_file_size_mb:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large: {file.filename}. Max size: {settings.max_file_size_mb}MB"
                )
            
            # Extract text from memory
            try:
                # Save temporarily for processing
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                    tmp.write(file_content)
                    tmp_path = tmp.name
                
                extracted_text = await file_processor.extract_text_from_file(tmp_path)
                os.unlink(tmp_path)  # Delete temp file
            except Exception as e:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise HTTPException(
                    status_code=500,
                    detail=f"Error extracting text from {file.filename}: {str(e)}"
                )
            
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Store metadata (no file path)
            uploaded_files_data[file_id] = {
                "filename": file.filename,
                "extracted_text": extracted_text,
                "size_mb": file_size_mb
            }
            
            uploaded_files_texts.append(extracted_text)
            
            # Create response
            text_preview = extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
            responses.append(FileUploadResponse(
                filename=file.filename,
                file_id=file_id,
                size_mb=round(file_size_mb, 2),
                text_preview=text_preview
            ))
        
        # Process documents with RAG pipeline
        if uploaded_files_texts:
            metadata = [{"filename": data["filename"]} for data in uploaded_files_data.values()]
            session_id = rag_pipeline.process_documents(uploaded_files_texts, metadata)
            print(f"Created RAG session: {session_id}")
        
        return responses
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with the AI - works with or without uploaded files
    
    - With files: Answers based on uploaded documents using RAG
    - Without files: General conversational AI assistant
    """
    try:
        # Get LLM model
        try:
            llm = model_manager.get_model(
                model_key=request.model_key,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Check if files are uploaded
        if uploaded_files_texts:
            # Use RAG pipeline with uploaded documents
            try:
                # Convert chat_history to dict format if needed
                history_list = None
                if request.chat_history:
                    history_list = [msg.dict() for msg in request.chat_history]
                
                result = rag_pipeline.query(
                    llm=llm,
                    question=request.prompt,
                    retriever_k=request.retriever_k,
                    chat_history=history_list
                )
                
                return ChatResponse(
                    answer=result["answer"],
                    model_used=request.model_key,
                    source_documents=result["source_documents"]
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error generating response: {str(e)}"
                )
        else:
            # No files uploaded - use direct LLM for general conversation
            try:
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.output_parsers import StrOutputParser
                
                # Build conversation history
                history_text = ""
                if request.chat_history:
                    history_messages = []
                    for msg in request.chat_history[-6:]:  # Last 3 exchanges
                        role = "User" if msg.role == "user" else "Assistant"
                        history_messages.append(f"{role}: {msg.content}")
                    history_text = "\n".join(history_messages)
                
                # Create conversational prompt
                if history_text:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are a friendly, helpful AI assistant. Have natural conversations with users, answer questions, and provide assistance.
                        
Be warm, engaging, and conversational. You can discuss various topics, answer general questions, and have casual conversations."""),
                        ("human", f"""Previous conversation:
{history_text}

Current message: {{question}}

Please respond naturally and helpfully.""")
                    ])
                else:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are a friendly, helpful AI assistant. Have natural conversations with users, answer questions, and provide assistance.
                        
Be warm, engaging, and conversational. You can discuss various topics, answer general questions, and have casual conversations."""),
                        ("human", "{question}")
                    ])
                
                # Create chain
                chain = prompt | llm | StrOutputParser()
                
                # Get response
                answer = chain.invoke({"question": request.prompt})
                
                return ChatResponse(
                    answer=answer,
                    model_used=request.model_key,
                    source_documents=[]
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error generating response: {str(e)}"
                )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/clear", tags=["Files"])
async def clear_session():
    """Clear current session (files in memory only)"""
    global uploaded_files_data, uploaded_files_texts
    
    try:
        # Clear data (no files to delete - they're in memory)
        uploaded_files_data = {}
        uploaded_files_texts = []
        
        # Clear RAG pipeline
        rag_pipeline.clear()
        
        return {"message": "Session cleared successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")


@app.get("/files", tags=["Files"])
async def list_uploaded_files():
    """List currently uploaded files"""
    files_info = [
        {
            "file_id": file_id,
            "filename": data["filename"],
            "size_mb": round(data["size_mb"], 2),
            "text_length": len(data["extracted_text"])
        }
        for file_id, data in uploaded_files_data.items()
    ]
    
    return {
        "total_files": len(files_info),
        "files": files_info
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc.detail)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.debug else None
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
