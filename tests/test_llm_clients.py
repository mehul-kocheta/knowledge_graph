"""
Unit tests for LLM client factory and GroqClient env var handling.
Tasks: 1.1
"""
import os
import pytest
from unittest.mock import patch

from llm import get_client
from llm.ollama_client import OllamaClient
from llm.groq_client import GroqClient


def test_get_client_ollama_returns_ollama_client():
    client = get_client("ollama")
    assert isinstance(client, OllamaClient)


def test_get_client_groq_returns_groq_client():
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        client = get_client("groq")
    assert isinstance(client, GroqClient)


def test_get_client_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unknown client"):
        get_client("unknown_backend")


def test_groq_client_raises_when_api_key_missing():
    env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            GroqClient()
