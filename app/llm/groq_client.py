"""
app/llm/groq_client.py

Groq API client for DispatchOps AI.

WHAT THIS FILE DOES:
--------------------
One class, one responsibility: send a prompt to the Groq API and return
the raw text response. Nothing else.

This client knows nothing about jobs, transcripts, or extraction schemas.
It is a thin, reusable wrapper around the Groq API that any future LLM
feature can use.

WHY GROQ?
----------
Groq provides the fastest available inference for open-weight models
(Llama 3, Mixtral). For a dispatch system where latency matters —
a technician can't be assigned while we wait for a slow LLM response —
Groq's hardware acceleration gives us sub-second extraction times.

WHY NOT USE THE GROQ PYTHON SDK DIRECTLY IN THE SERVICE?
---------------------------------------------------------
The service would become tightly coupled to Groq's SDK. If we switch to
OpenAI or a self-hosted model, we'd change every service that calls LLMs.

GroqClient is the only place that knows about the Groq SDK. The service
receives a plain string. The client is easily mocked in tests.

WHY LANGCHAIN HERE?
--------------------
LangChain's ChatGroq provides:
- A clean, consistent interface (invoke/ainvoke) regardless of provider.
- Automatic message formatting (system/human roles).
- Provider switching without changing calling code.

We use ONLY ChatGroq and the invoke call — no agents, no chains, no memory.

Interview talking point:
"GroqClient is a thin wrapper. The service passes a formatted prompt string
and gets back a plain string. If we switch from Groq to OpenAI tomorrow,
only this file changes. Tests mock this class entirely — they never hit
the real API."
"""

from __future__ import annotations

from app.utils.exceptions import ExtractionError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# LangChain's ChatGroq imported at module level so tests can patch it.
# If langchain-groq is not installed, set to None gracefully.
try:
    from langchain_groq import ChatGroq  # type: ignore[import]
except ImportError:
    ChatGroq = None  # type: ignore[assignment]


class GroqClient:
    """
    Thin wrapper around the Groq API via LangChain's ChatGroq.

    Accepts a prompt string, returns a response string.
    All LLM concerns (retries, temperature, model name) are configured here.

    Args:
        api_key:    Groq API key.
        model_name: Groq model to use (e.g. "llama3-8b-8192").
        temperature: Sampling temperature. 0.0 = deterministic output,
                     which is what we want for structured extraction.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "llama3-8b-8192",
        temperature: float = 0.0,
    ) -> None:
        if ChatGroq is None:
            raise ExtractionError(
                message="langchain-groq is not installed.",
                detail="Run: pip install langchain-groq",
            )
        self._model_name = model_name
        self._llm = ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )

    def complete(self, prompt: str) -> str:
        """
        Send a prompt to the Groq API and return the response text.

        Uses temperature=0.0 so structured extraction is deterministic.
        The same transcript should always produce the same JSON output.

        Args:
            prompt: The fully formatted prompt string to send.

        Returns:
            The raw text content of the LLM response.

        Raises:
            ExtractionError: If the API call fails for any reason.
        """
        logger.info(
            "Sending prompt to Groq",
            extra={
                "model": self._model_name,
                "prompt_length": len(prompt),
            },
        )

        try:
            # LangChain invoke: accepts a string, returns an AIMessage.
            # .content gives us the plain text response string.
            response = self._llm.invoke(prompt)
            text = response.content.strip()

            logger.info(
                "Groq response received",
                extra={
                    "model": self._model_name,
                    "response_length": len(text),
                },
            )
            return text

        except Exception as exc:
            logger.error(
                "Groq API call failed",
                extra={
                    "model": self._model_name,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            raise ExtractionError(
                message="LLM API call failed.",
                detail=str(exc),
            ) from exc
