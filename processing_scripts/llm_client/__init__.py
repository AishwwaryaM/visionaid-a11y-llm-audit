"""
llm_client — Claude API client for the VisionAid accessibility audit pipeline.

Usage
-----
    from llm_client import AuditClient, load_all_prompts, run_all

    client = AuditClient(model="claude-sonnet-4-6", temperature=0.1)
    prompts = load_all_prompts(PROMPTS_DIR)
    report  = run_all(client, prompts, all_slices)
"""

from .client import AuditClient
from .prompt_loader import load_prompts, load_all_prompts
from .runner import run_checklist, run_all

__all__ = [
    "AuditClient",
    "load_prompts",
    "load_all_prompts",
    "run_checklist",
    "run_all",
]
