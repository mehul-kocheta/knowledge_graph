"""
Property-based tests for kg-api-groq.
Tasks: 1.2, 3.2, 3.3
"""
import io
import json
import os
import pytest
from unittest.mock import patch, MagicMock
from hypothesis import given, settings
import hypothesis.strategies as st
from fastapi.testclient import TestClient

from llm import get_client
from api.main import app

api_client = TestClient(app)

VALID_NAMES = ("ollama", "groq")

# ---------------------------------------------------------------------------
# Property 1: Invalid client name always raises ValueError
# Feature: kg-api-groq, Property 1: Invalid client name always raises ValueError
# ---------------------------------------------------------------------------

@given(st.text().filter(lambda s: s not in VALID_NAMES))
@settings(max_examples=100)
def test_property1_invalid_client_name_raises_value_error(name):
    """For any string that is not 'ollama' or 'groq', get_client() raises ValueError."""
    with pytest.raises(ValueError):
        get_client(name)


# ---------------------------------------------------------------------------
# Property 2: CSV output always contains required columns
# Feature: kg-api-groq, Property 2: CSV output always contains required columns
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"node_1", "node_2", "edge", "chunk_id"}

def _make_mock_client(response_json: str):
    mock = MagicMock()
    mock.generate.return_value = response_json
    return mock


@given(st.text(min_size=1, alphabet=st.characters(blacklist_categories=("Cs",))))
@settings(max_examples=100)
def test_property2_csv_output_has_required_columns(text_input):
    """For any non-empty text, the CSV returned always has the required columns."""
    valid_json = json.dumps([{"node_1": "a", "node_2": "b", "edge": "rel"}])
    mock_client = _make_mock_client(valid_json)
    with patch("api.main.get_client", return_value=mock_client):
        response = api_client.post(
            "/extract",
            files={"file": ("input.txt", text_input.encode("utf-8", errors="replace"), "text/plain")},
        )
    assert response.status_code == 200
    first_line = response.text.splitlines()[0]
    columns = {c.strip() for c in first_line.split(",")}
    assert REQUIRED_COLUMNS.issubset(columns)


# ---------------------------------------------------------------------------
# Property 3: Bad LLM chunks are skipped, output remains valid CSV
# Feature: kg-api-groq, Property 3: Bad LLM chunks are skipped, output remains valid CSV
# ---------------------------------------------------------------------------

@given(st.lists(st.booleans(), min_size=1, max_size=10))
@settings(max_examples=100)
def test_property3_bad_chunks_skipped_output_is_valid_csv(good_flags):
    """
    For any mix of valid/invalid LLM responses, the pipeline never raises an
    unhandled exception and always returns a valid CSV with the required columns.
    """
    valid_json = json.dumps([{"node_1": "x", "node_2": "y", "edge": "rel"}])
    call_count = [0]

    def side_effect(**kwargs):
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(good_flags) and good_flags[idx]:
            return valid_json
        return "NOT VALID JSON {{{"

    mock = MagicMock()
    mock.generate.side_effect = side_effect

    # Build a text long enough to produce at least len(good_flags) chunks
    text = ("Alice knows Bob. " * 100)

    with patch("api.main.get_client", return_value=mock):
        response = api_client.post(
            "/extract",
            files={"file": ("input.txt", text.encode("utf-8"), "text/plain")},
        )

    assert response.status_code == 200
    first_line = response.text.splitlines()[0]
    columns = {c.strip() for c in first_line.split(",")}
    assert REQUIRED_COLUMNS.issubset(columns)
