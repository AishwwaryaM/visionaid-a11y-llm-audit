"""
Parses the numbered prompt sections from the LLM prompt-library text files.

File format (see processing_scripts/llm/*.txt):
    ---------------------------------
    N) PROMPT TITLE
    ---------------------------------
    <prompt body>
    Data: {payload}

    ---------------------------------
    N+1) NEXT PROMPT TITLE
    ...
"""

import re
from pathlib import Path


_HEADER_RE = re.compile(r"^\s{2,}(\d+)\)\s+.+", re.MULTILINE)
_DIVIDER_RE = re.compile(r"^\s*-{5,}\s*$", re.MULTILINE)


def load_prompts(filepath: str | Path) -> dict[int, str]:
    """
    Parse *filepath* and return ``{prompt_number: prompt_text}`` for every
    numbered section found.  Divider lines (``---...``) are stripped.

    The returned text for each prompt still contains the ``{payload}``
    placeholder so the caller can fill it later.
    """
    text = Path(filepath).read_text(encoding="utf-8")

    # Find all header positions and their prompt numbers
    headers = list(_HEADER_RE.finditer(text))
    if not headers:
        return {}

    prompts: dict[int, str] = {}

    for i, header in enumerate(headers):
        num = int(header.group(1))
        start = header.end()  # character just after the header line
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)

        body = text[start:end]

        # Remove divider lines
        body = _DIVIDER_RE.sub("", body)

        prompts[num] = body.strip()

    return prompts


def load_all_prompts(prompts_dir: str | Path) -> dict[str, dict[int, str]]:
    """
    Load all ``*.txt`` prompt files in *prompts_dir*.

    Returns ``{stem: {prompt_number: prompt_text}}``, e.g.::

        {
            "semantic_checklist_01": {1: "...", 2: "...", ...},
            "forms_checklist_02":    {1: "...", ...},
            "nontext_checklist_03":  {1: "...", ...},
        }
    """
    prompts_dir = Path(prompts_dir)
    result: dict[str, dict[int, str]] = {}
    for txt_file in sorted(prompts_dir.glob("*.txt")):
        result[txt_file.stem] = load_prompts(txt_file)
    return result
