"""
Unit tests for the FastAPI /extract endpoint.
Tasks: 3.1
"""
import io
import os
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_GRAPH_JSON = json.dumps([
    {"node_1": "Alice", "node_2": "Bob", "edge": "knows"}
])


def _mock_llm_client(response_json: str):
    """Return a mock LLMClient whose generate() always returns response_json."""
    mock = MagicMock()
    mock.generate.return_value = response_json
    return mock


def test_empty_file_returns_422():
    response = client.post(
        "/extract",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 422


def test_whitespace_only_file_returns_422():
    response = client.post(
        "/extract",
        files={"file": ("ws.txt", b"   \n\t  ", "text/plain")},
    )
    assert response.status_code == 422


def test_valid_upload_returns_200_with_csv_content_type():
    mock_client = _mock_llm_client(VALID_GRAPH_JSON)
    with patch("api.main.get_client", return_value=mock_client):
        response = client.post(
            "/extract",
            files={"file": ("sample.txt", b"Alice knows Bob.", "text/plain")},
        )
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]


def test_valid_upload_csv_has_required_columns():
    mock_client = _mock_llm_client(VALID_GRAPH_JSON)
    with patch("api.main.get_client", return_value=mock_client):
        response = client.post(
            "/extract",
            files={"file": ("sample.txt", b"Alice knows Bob.", "text/plain")},
        )
    assert response.status_code == 200
    first_line = response.text.splitlines()[0]
    columns = [c.strip() for c in first_line.split(",")]
    for col in ("node_1", "node_2", "edge", "chunk_id"):
        assert col in columns, f"Missing column: {col}"


def test_client_groq_query_param_is_accepted():
    mock_client = _mock_llm_client(VALID_GRAPH_JSON)
    with patch("api.main.get_client", return_value=mock_client) as mock_get:
        response = client.post(
            "/extract?client=groq",
            files={"file": ("sample.txt", b"Alice knows Bob.", "text/plain")},
        )
    mock_get.assert_called_once_with("groq")
    assert response.status_code == 200
