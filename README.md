# ChatBot Backend with RAG and Multiple LLM Models

A FastAPI backend for a chatbot that supports multiple free LLM models and processes various file types (images, PDFs, text files) to provide context-based responses without hallucination using Retrieval Augmented Generation (RAG).

## Features

- **Multiple Free LLM Models**: Choose from several free models including:
  - Mixtral 8x7B (Groq)
  - Llama 3 70B & 8B (Groq)
  - Gemma 7B (Groq)
  - Mistral 7B (HuggingFace)

- **File Processing**: Upload and process multiple file types:
  - Text files (.txt)
  - PDF documents (.pdf)
  - Word documents (.doc, .docx)
  - Images (.png, .jpg, .jpeg, .gif, .bmp) with OCR

- **RAG Pipeline**: Ensures responses are based only on uploaded files, preventing hallucination

- **Vector Store**: Uses FAISS for efficient similarity search

- **Embeddings**: Free HuggingFace embeddings (all-MiniLM-L6-v2)

## Installation

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Install Tesseract OCR** (for image text extraction):
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

3. **Configure environment variables**:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
- Get Groq API key (free): https://console.groq.com
- Get HuggingFace token (optional): https://huggingface.co/settings/tokens

## Running the Application

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: http://localhost:8000

API Documentation: http://localhost:8000/docs

## API Endpoints

### Health Check
```
GET /health
```

### List Available Models
```
GET /models
```

### Upload Files
```
POST /upload
Content-Type: multipart/form-data
Body: files (multiple files)
```

### Chat
```
POST /chat
Content-Type: application/json
Body:
{
  "prompt": "Your question here",
  "model_key": "groq_mixtral",
  "temperature": 0.7,
  "max_tokens": 2048,
  "retriever_k": 4
}
```

### List Uploaded Files
```
GET /files
```

### Clear Session
```
POST /clear
```

## Usage Example

1. **Upload files**:
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document.pdf" \
  -F "files=@image.png"
```

2. **Ask a question**:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is mentioned in the uploaded documents?",
    "model_key": "groq_mixtral"
  }'
```

## Available Models

- `groq_mixtral` - Mixtral 8x7B (recommended)
- `groq_llama3_70b` - Llama 3 70B
- `groq_llama3_8b` - Llama 3 8B (faster)
- `groq_gemma_7b` - Gemma 7B
- `hf_mistral` - Mistral 7B (requires HuggingFace token)

## Project Structure

```
backend/
├── main.py              # FastAPI application and endpoints
├── config.py            # Configuration and settings
├── models.py            # Pydantic models for API
├── file_processor.py    # File processing utilities
├── llm_models.py        # LLM model management
├── rag_pipeline.py      # RAG implementation
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── uploads/             # Uploaded files directory
└── vector_store/        # Vector store directory
```

## Configuration

Edit `.env` file to customize:

- `GROQ_API_KEY` - Your Groq API key
- `HUGGINGFACEHUB_API_TOKEN` - Your HuggingFace token (optional)
- `MAX_FILE_SIZE_MB` - Maximum file size (default: 10MB)
- `CHUNK_SIZE` - Text chunk size for RAG (default: 1000)
- `CHUNK_OVERLAP` - Chunk overlap (default: 200)
- `DEFAULT_MODEL` - Default LLM model
- `TEMPERATURE` - Default temperature (default: 0.7)
- `MAX_TOKENS` - Default max tokens (default: 2048)

## Notes

- The chatbot only responds based on uploaded files to prevent hallucination
- Files are processed using RAG (Retrieval Augmented Generation)
- Each session maintains its own vector store
- Use `/clear` endpoint to reset session and delete files
- OCR is used to extract text from images (requires Tesseract)

## License

MIT
