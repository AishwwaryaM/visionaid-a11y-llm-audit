"""Pass 1 → Pass 2 filtering: exclude programmatic findings from LLM payloads.

Items that failed a binary programmatic check (missing alt, no label, empty title)
are excluded from LLM evaluation — there is no value in judging the quality of
something that doesn't exist.

Usage:
    flags = build_filter_flags(sem_findings, form_findings, ntext_findings)
    cl01_payload = apply_cl01_filters(cl01_payload, flags)
    cl02_payload = apply_cl02_filters(cl02_payload, flags)
    cl03_payload = apply_cl03_filters(cl03_payload, flags)
"""

from __future__ import annotations

import copy


def build_filter_flags(
    sem_findings: list[dict],
    form_findings: list[dict],
    ntext_findings: list[dict],
) -> dict:
    """Build a dict of boolean flags and a skip_prompts set from programmatic findings.

    Args:
        sem_findings: Results from semantic_checklist_01.audit_html_file().
        form_findings: Results from forms_checklist_02.audit_forms().
        ntext_findings: Results from nontext_checklist_03.audit_nontext().

    Returns:
        Dict with boolean filter flags and a ``skip_prompts`` set of prompt
        names that should be skipped entirely.
    """
    sem_rules = {f["rule_id"] for f in sem_findings}
    form_rules = {f["rule_id"] for f in form_findings}
    ntext_rules = {f["rule_id"] for f in ntext_findings}

    skip_title_prompt = bool(sem_rules & {"PAGE_TITLE_001", "PAGE_TITLE_003"})
    head_empty_fired = "HEAD_004" in sem_rules
    iframe_rules_fired = bool(sem_rules & {"IFRAME_001", "IFRAME_002"})
    form_label_fired = "FORM_LABEL_001" in form_rules
    ntext_action_fired = "NON_TEXT_002" in ntext_rules

    skip_prompts: set[str] = set()
    if skip_title_prompt:
        skip_prompts.add("page_title")

    return {
        "skip_title_prompt": skip_title_prompt,
        "head_empty_fired": head_empty_fired,
        "iframe_rules_fired": iframe_rules_fired,
        "form_label_fired": form_label_fired,
        "ntext_action_fired": ntext_action_fired,
        "skip_prompts": skip_prompts,
    }


def apply_cl01_filters(payload: dict, flags: dict) -> dict:
    """Return a deep copy of the CL01 payload with Pass 1 items removed.

    Filters applied:
    - HEAD_004: remove headings with empty text from ``headings``.
    - IFRAME_001/002: remove iframes with missing/empty title from ``iframes``.
    """
    payload = copy.deepcopy(payload)

    if flags["head_empty_fired"]:
        payload["headings"] = [
            h for h in payload.get("headings", [])
            if h.get("text", "").strip()
        ]

    if flags["iframe_rules_fired"]:
        payload["iframes"] = [
            f for f in payload.get("iframes", [])
            if f.get("title") and str(f["title"]).strip()
        ]

    return payload


def apply_cl02_filters(payload: dict, flags: dict) -> dict:
    """Return a deep copy of the CL02 payload with Pass 1 items removed.

    Filters applied:
    - FORM_LABEL_001: remove fields with ``label_source == "none"`` from each
      form's ``fields`` list.
    """
    payload = copy.deepcopy(payload)

    if flags["form_label_fired"]:
        for form in payload.get("forms", []):
            form["fields"] = [
                f for f in form.get("fields", [])
                if f.get("label_source") != "none"
            ]

    return payload


def apply_cl03_filters(payload: dict, flags: dict) -> dict:
    """Return a deep copy of the CL03 payload with Pass 1 items removed.

    Filters applied:
    - NON_TEXT_002: remove actionable images with missing/empty alt from
      ``images.actionable``.
    """
    payload = copy.deepcopy(payload)

    if flags["ntext_action_fired"]:
        payload["images"]["actionable"] = [
            im for im in payload["images"].get("actionable", [])
            if im.get("alt") and str(im["alt"]).strip()
        ]

    return payload
