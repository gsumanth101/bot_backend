from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class FileUploadResponse(BaseModel):
    """Response model for file upload"""
    filename: str
    file_id: str
    size_mb: float
    text_preview: str = Field(..., description="First 200 characters of extracted text")


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    prompt: str = Field(..., description="User's question or prompt")
    model_key: str = Field(default="groq_llama3_70b", description="Model to use for response")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Temperature for generation")
    max_tokens: Optional[int] = Field(default=None, ge=100, le=8192, description="Maximum tokens to generate (min: 100 for complete responses)")
    retriever_k: Optional[int] = Field(default=4, ge=1, le=10, description="Number of relevant documents to retrieve")
    chat_history: Optional[List[ChatMessage]] = Field(default=None, description="Previous conversation history")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str
    model_used: str
    source_documents: List[Dict[str, Any]] = Field(default_factory=list)


class ModelInfo(BaseModel):
    """Model information"""
    key: str
    name: str
    provider: str
    description: str
    max_tokens: int


class ModelsListResponse(BaseModel):
    """Response model for listing available models"""
    models: List[ModelInfo]
    current_model: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    app_name: str
    version: str
    groq_configured: bool
    huggingface_configured: bool


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
