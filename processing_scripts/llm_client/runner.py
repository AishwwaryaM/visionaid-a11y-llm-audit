"""
High-level runner that executes one or all checklist prompt sets against
pre-built JSON payloads, collects responses, and aggregates token usage.
"""

from __future__ import annotations

from typing import Any

from .client import AuditClient


def run_checklist(
    client: AuditClient,
    prompts: dict[int, str],
    slices: dict[str, Any | None],
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run every prompt in *prompts* against the matching payload slice in *slices*.

    Parameters
    ----------
    client : AuditClient
        Configured API client.
    prompts : dict[int, str]
        ``{prompt_number: prompt_text}`` from ``load_prompts()``.
    slices : dict[str, payload | None]
        Mapping of label → payload dict (or ``None`` to skip).
        Labels are used as keys in the returned results dict.
    verbose : bool
        Print progress to stdout while running.

    Returns
    -------
    dict with keys:
        - ``"results"``  : ``{label: response_json}``
        - ``"skipped"``  : list of skipped labels
        - ``"usage"``    : ``{"input_tokens": int, "output_tokens": int}``
        - ``"errors"``   : ``{label: error_message}``
    """
    results: dict[str, Any] = {}
    skipped: list[str] = []
    errors: dict[str, str] = {}
    total_input = 0
    total_output = 0

    # Map labels → prompt numbers by position (label order == prompt order)
    labels = list(slices.keys())

    for idx, label in enumerate(labels):
        prompt_num = idx + 1
        payload = slices[label]

        if payload is None:
            if verbose:
                print(f"  [{prompt_num}] {label} — SKIPPED (filtered by Pass 1)")
            skipped.append(label)
            continue

        # Skip if the payload is empty after filtering
        if _is_empty_payload(payload):
            if verbose:
                print(f"  [{prompt_num}] {label} — SKIPPED (empty payload after filtering)")
            skipped.append(label)
            continue

        prompt_text = prompts.get(prompt_num)
        if prompt_text is None:
            if verbose:
                print(f"  [{prompt_num}] {label} — SKIPPED (no prompt text found)")
            skipped.append(label)
            continue

        if verbose:
            print(f"  [{prompt_num}] {label} ...", end=" ", flush=True)

        try:
            response, usage = client.call(prompt_text, payload)
            results[label] = response
            total_input += usage["input_tokens"]
            total_output += usage["output_tokens"]
            if verbose:
                print(f"✓ ({usage['input_tokens']}in / {usage['output_tokens']}out tokens)")
        except Exception as exc:
            errors[label] = str(exc)
            if verbose:
                print(f"✗ ERROR: {exc}")

    return {
        "results": results,
        "skipped": skipped,
        "usage": {"input_tokens": total_input, "output_tokens": total_output},
        "errors": errors,
    }


def run_all(
    client: AuditClient,
    all_prompts: dict[str, dict[int, str]],
    all_slices: dict[str, dict[str, Any | None]],
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run all checklists in *all_slices* and return a consolidated report dict.

    Parameters
    ----------
    client : AuditClient
    all_prompts : dict[str, dict[int, str]]
        ``{checklist_stem: {prompt_num: prompt_text}}`` from ``load_all_prompts()``.
    all_slices : dict[str, dict[str, payload|None]]
        ``{checklist_stem: {label: payload|None}}``.
    verbose : bool

    Returns
    -------
    dict with keys:
        - ``"checklists"`` : ``{stem: checklist_result}``
        - ``"total_usage"`` : ``{"input_tokens": int, "output_tokens": int}``
    """
    checklists: dict[str, Any] = {}
    grand_input = 0
    grand_output = 0

    for stem, slices in all_slices.items():
        prompts = all_prompts.get(stem, {})
        if verbose:
            print(f"\n--- {stem} ---")

        result = run_checklist(client, prompts, slices, verbose=verbose)
        checklists[stem] = result
        grand_input += result["usage"]["input_tokens"]
        grand_output += result["usage"]["output_tokens"]

    return {
        "checklists": checklists,
        "total_usage": {"input_tokens": grand_input, "output_tokens": grand_output},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_empty_payload(payload: Any) -> bool:
    """Return True if every list-valued entry in the payload is empty."""
    if not isinstance(payload, dict):
        return False
    for v in payload.values():
        if isinstance(v, list) and v:
            return False
        if isinstance(v, dict) and v:
            return False
        if isinstance(v, str) and v.strip():
            return False
    return True
