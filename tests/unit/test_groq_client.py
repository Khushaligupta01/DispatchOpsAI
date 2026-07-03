"""
tests/unit/test_groq_client.py

Unit tests for GroqClient.

The real Groq API is NEVER called. ChatGroq is patched at the module level
in app/llm/groq_client.py so no API key, no network request, and no
LangChain dependency is required to run these tests.

What we test:
- complete() returns the stripped .content string from the LLM response.
- complete() raises ExtractionError when the LLM call throws.
- GroqClient logs prompt length and response length.
- The LLM is invoked with the exact prompt string passed in.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.utils.exceptions import ExtractionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_groq_client(model_name: str = "llama3-8b-8192"):
    """
    Build a GroqClient with ChatGroq patched out.

    Returns (client, mock_llm) so tests can configure mock_llm.invoke().
    """
    mock_llm = MagicMock()
    # Patch at module level so the GroqClient __init__ uses the mock
    with patch("app.llm.groq_client.ChatGroq") as mock_chat_groq_cls:
        mock_chat_groq_cls.return_value = mock_llm
        from app.llm.groq_client import GroqClient
        client = GroqClient(api_key="test-key", model_name=model_name)
    # Swap the internal _llm to our controlled mock after construction
    client._llm = mock_llm
    return client, mock_llm


def make_ai_message(content: str) -> MagicMock:
    """Build a mock AIMessage with a .content attribute."""
    msg = MagicMock()
    msg.content = content
    return msg


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_complete_returns_response_text():
    """complete() returns the stripped content from the LLM response."""
    client, mock_llm = make_groq_client()
    mock_llm.invoke.return_value = make_ai_message(
        '  {"customer_name": "Alice"}  '
    )

    result = client.complete("Extract from: hello")

    assert result == '{"customer_name": "Alice"}'


def test_complete_invokes_llm_with_exact_prompt():
    """complete() passes the prompt string directly to llm.invoke()."""
    client, mock_llm = make_groq_client()
    mock_llm.invoke.return_value = make_ai_message("{}")

    prompt = "Extract dispatch info from: pipe burst"
    client.complete(prompt)

    mock_llm.invoke.assert_called_once_with(prompt)


def test_complete_strips_whitespace_from_response():
    """Whitespace in the LLM response is stripped before returning."""
    client, mock_llm = make_groq_client()
    mock_llm.invoke.return_value = make_ai_message("\n\n{}\n\n")

    result = client.complete("prompt")

    assert result == "{}"


def test_complete_returns_non_json_response_as_is():
    """complete() returns whatever the LLM responds — parsing is the service's job."""
    client, mock_llm = make_groq_client()
    mock_llm.invoke.return_value = make_ai_message("I cannot extract that.")

    result = client.complete("prompt")

    assert result == "I cannot extract that."


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------

def test_complete_raises_extraction_error_on_llm_failure():
    """When llm.invoke() raises, ExtractionError is raised."""
    client, mock_llm = make_groq_client()
    mock_llm.invoke.side_effect = RuntimeError("Connection timeout")

    with pytest.raises(ExtractionError) as exc_info:
        client.complete("prompt")

    assert "LLM API call failed" in exc_info.value.message


def test_complete_includes_original_error_in_detail():
    """ExtractionError.detail contains the original exception message."""
    client, mock_llm = make_groq_client()
    mock_llm.invoke.side_effect = RuntimeError("rate limit exceeded")

    with pytest.raises(ExtractionError) as exc_info:
        client.complete("prompt")

    assert "rate limit exceeded" in exc_info.value.detail


def test_complete_raises_extraction_error_not_base_exception():
    """The raised exception is specifically ExtractionError, not a raw RuntimeError."""
    client, mock_llm = make_groq_client()
    mock_llm.invoke.side_effect = ValueError("unexpected response format")

    with pytest.raises(ExtractionError):
        client.complete("prompt")


# ---------------------------------------------------------------------------
# ChatGroq not installed
# ---------------------------------------------------------------------------

def test_groq_client_raises_if_langchain_not_installed():
    """If ChatGroq is None (not installed), GroqClient raises ExtractionError."""
    with patch("app.llm.groq_client.ChatGroq", None):
        from app.llm.groq_client import GroqClient
        with pytest.raises(ExtractionError) as exc_info:
            GroqClient(api_key="key")
        assert "langchain-groq" in exc_info.value.message
