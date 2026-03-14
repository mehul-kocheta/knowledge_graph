from llm.base import LLMClient
from llm.ollama_client import OllamaClient
from llm.groq_client import GroqClient


def get_client(name: str) -> LLMClient:
    if name == "ollama":
        return OllamaClient()
    elif name == "groq":
        return GroqClient()
    else:
        raise ValueError(f"Unknown client '{name}'. Choose 'ollama' or 'groq'.")
