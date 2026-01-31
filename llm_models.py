from typing import Optional, Dict, Any

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.language_models import BaseLanguageModel

from config import settings


class LLMModelManager:
    """Manage multiple LLM models with selection capability"""

    AVAILABLE_MODELS = {
        # ================= GROQ (FREE) - CONVERSATIONAL =================
        # "groq_llama3_70b": {
        #     "name": "Llama 3.1 70B Versatile (Groq)",
        #     "model_id": "llama-3.1-70b-versatile",
        #     "provider": "groq",
        #     "description": "Best conversational model - Natural dialogue, reasoning & context understanding",
        #     "max_tokens": 32768,
        # },
        # "groq_llama3_8b": {
        #     "name": "Llama 3.1 8B Instant (Groq)",
        #     "model_id": "llama-3.1-8b-instant",
        #     "provider": "groq",
        #     "description": "Fast conversational responses with good context retention",
        #     "max_tokens": 8192,
        # },
        "groq_llama3_3_70b": {
            "name": "Llama 3.3 70B Versatile (Groq)",
            "model_id": "llama-3.3-70b-versatile",
            "provider": "groq",
            "description": "Latest Llama model - Superior conversational abilities & nuanced understanding",
            "max_tokens": 32768,
        },
        # "groq_mixtral": {
        #     "name": "Mixtral 8x7B (Groq)",
        #     "model_id": "mixtral-8x7b-32768",
        #     "provider": "groq",
        #     "description": "Excellent for detailed conversations and multi-turn dialogue",
        #     "max_tokens": 32768,
        # },
        # "groq_gemma2_9b": {
        #     "name": "Gemma 2 9B IT (Groq)",
        #     "model_id": "gemma2-9b-it",
        #     "provider": "groq",
        #     "description": "Instruction-tuned for helpful, conversational responses",
        #     "max_tokens": 8192,
        # },

        # ================= HUGGINGFACE (FREE) =================
        "hf_mistral_7b": {
            "name": "Mistral 7B Instruct (HF)",
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "provider": "huggingface",
            "description": "Conversational instruction-following model via HuggingFace",
            "max_tokens": 4096,
        },
    }

    def __init__(self):
        self.current_model: Optional[BaseLanguageModel] = None
        self.current_model_key: Optional[str] = None

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Return available model metadata"""
        return self.AVAILABLE_MODELS

    def get_model(
        self,
        model_key: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> BaseLanguageModel:
        """
        Create and return an LLM instance
        """

        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{model_key}' not found. "
                f"Available models: {list(self.AVAILABLE_MODELS.keys())}"
            )

        model_info = self.AVAILABLE_MODELS[model_key]
        provider = model_info["provider"]
        model_id = model_info["model_id"]

        temperature = temperature if temperature is not None else settings.temperature
        max_tokens = max_tokens if max_tokens is not None else settings.max_tokens
        max_tokens = min(max_tokens, model_info["max_tokens"])

        # ================= GROQ =================
        if provider == "groq":
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY not set")

            llm = ChatGroq(
                groq_api_key=settings.groq_api_key,
                model_name=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # ================= HUGGINGFACE =================
        elif provider == "huggingface":
            if not settings.huggingfacehub_api_token:
                raise ValueError("HUGGINGFACEHUB_API_TOKEN not set")

            llm = HuggingFaceEndpoint(
                repo_id=model_id,
                huggingfacehub_api_token=settings.huggingfacehub_api_token,
                temperature=temperature,
                max_new_tokens=max_tokens,
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.current_model = llm
        self.current_model_key = model_key
        return llm

    def get_current_model(self) -> Optional[BaseLanguageModel]:
        return self.current_model

    def get_current_model_key(self) -> Optional[str]:
        return self.current_model_key
