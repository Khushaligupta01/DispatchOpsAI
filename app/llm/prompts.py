"""
app/llm/prompts.py

Prompt loader for DispatchOps AI LLM operations.

WHY STORE PROMPTS IN MARKDOWN FILES?
--------------------------------------
Prompts are natural language instructions — not Python logic. Storing
them in .md files means:

1. A prompt engineer can edit the prompt without touching Python code.
2. The prompt gets syntax highlighting and readable formatting in any editor.
3. Git diffs on prompt changes are clean and human-readable.
4. The prompt file path is an artefact you can reference in Langfuse traces
   alongside the version string, making regression analysis straightforward.

PROMPT VERSIONING:
------------------
EXTRACTION_PROMPT_VERSION is a string constant that identifies the active
prompt. Include this in Langfuse span metadata so you can filter traces by
prompt version and compare output quality across versions.

Interview talking point:
"Prompts live in .md files under prompts/, loaded at startup. When I change
a prompt, I bump the version string here. Langfuse traces reference the
version, so I can compare output quality before and after any prompt change."
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Prompt version — increment when the prompt content changes
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT_VERSION = "extraction_v1"

# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

def _load_prompt(relative_path: str) -> str:
    """
    Load a prompt template from a file relative to the project root.

    Args:
        relative_path: Path from the project root, e.g. "prompts/dispatch_extraction.md"

    Returns:
        The raw prompt string, stripped of leading/trailing whitespace.

    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
    """
    # __file__ is app/llm/prompts.py → project root is two levels up
    project_root = Path(__file__).parent.parent.parent
    prompt_path = project_root / relative_path

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: '{prompt_path}'. "
            f"Expected at: prompts/dispatch_extraction.md"
        )

    return prompt_path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Loaded prompt templates
# ---------------------------------------------------------------------------

# Loaded once at module import time — no per-request file I/O.
# <<TRANSCRIPT>> is the placeholder — replaced by ExtractionService
# using str.replace() to avoid conflicts with JSON curly braces.
EXTRACTION_PROMPT: str = _load_prompt("prompts/dispatch_extraction.md")
