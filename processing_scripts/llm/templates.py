"""Parse prompt .txt files and fill {payload} placeholders."""

import re
from pathlib import Path
from functools import lru_cache

from .registry import PromptSpec

# Project root (three levels up: llm/ → processing_scripts/ → project-root/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Regex to detect prompt section headers like:
#   1) PAGE TITLE EVALUATION PROMPT
#   2) HEADING STRUCTURE EVALUATION PROMPT
_HEADER_RE = re.compile(r"^\s*-+\s*\n\s*(\d+)\)\s+.+?\n\s*-+\s*$", re.MULTILINE)


@lru_cache(maxsize=None)
def _parse_prompt_file(file_path: Path) -> dict[int, str]:
    """Parse a prompt .txt file into {prompt_number: template_text}.

    Each prompt is delimited by a dashed header block containing a number:
        ---...---
        N) PROMPT NAME
        ---...---

    Returns a dict keyed by the 1-based prompt number.
    """
    text = file_path.read_text(encoding="utf-8")

    # Find all header positions
    headers = list(_HEADER_RE.finditer(text))
    if not headers:
        raise ValueError(f"No prompt headers found in {file_path}")

    prompts: dict[int, str] = {}
    for i, match in enumerate(headers):
        prompt_num = int(match.group(1))
        # Template text starts after the header block
        start = match.end()
        # Ends at the next header, or end of file
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        template = text[start:end].strip()
        prompts[prompt_num] = template

    return prompts


def get_template(spec: PromptSpec) -> str:
    """Load the raw template text for a PromptSpec (before payload substitution)."""
    file_path = PROJECT_ROOT / spec.prompt_file
    prompts = _parse_prompt_file(file_path)
    if spec.prompt_index not in prompts:
        raise KeyError(
            f"Prompt index {spec.prompt_index} not found in {spec.prompt_file}. "
            f"Available: {sorted(prompts.keys())}"
        )
    return prompts[spec.prompt_index]


def fill_template(spec: PromptSpec, payload_json: str) -> str:
    """Load the template for a PromptSpec and replace {payload} with the given JSON string."""
    template = get_template(spec)
    if "{payload}" not in template:
        raise ValueError(
            f"Template for '{spec.name}' (index {spec.prompt_index} in "
            f"{spec.prompt_file}) does not contain a {{payload}} placeholder."
        )
    return template.replace("{payload}", payload_json)
