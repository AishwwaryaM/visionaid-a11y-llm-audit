"""Functions that extract the right JSON slice for each prompt from extractor payloads.

Each function takes a full extractor payload dict and returns a JSON string
ready to be substituted into the {payload} placeholder in a prompt template.
"""

import json


def _dumps(obj: object) -> str:
    """Serialize to compact but readable JSON."""
    return json.dumps(obj, indent=2, ensure_ascii=False)


def is_empty_slice(payload_json: str) -> bool:
    """Return True if the payload JSON represents an empty collection.

    Handles [], {}, "[]", "{}", and null/empty-string edge cases.
    """
    stripped = payload_json.strip()
    if stripped in ("[]", "{}", "null", ""):
        return True
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return False
    if isinstance(parsed, (list, dict)) and len(parsed) == 0:
        return True
    return False


# ── CL01: Semantic Structure ─────────────────────────────────────────────────

def slice_page_title(payload: dict) -> str:
    """Return the page_title object."""
    return _dumps(payload["page_title"])


def slice_headings(payload: dict) -> str:
    """Return page_title + headings for hierarchy evaluation."""
    return _dumps({
        "page_title": payload["page_title"],
        "headings": payload["headings"],
    })


def slice_flagged_links(payload: dict) -> str:
    """Return the flagged links list."""
    return _dumps(payload["flagged_links"])


def slice_tables(payload: dict) -> str:
    """Return the tables list."""
    return _dumps(payload["tables"])


def slice_iframes(payload: dict) -> str:
    """Return the iframes list."""
    return _dumps(payload["iframes"])


def slice_landmarks(payload: dict) -> str:
    """Return the landmarks list."""
    return _dumps(payload["landmarks"])


def slice_cl01_full(payload: dict) -> str:
    """Return the full CL01 payload for the semantic summary prompt."""
    return _dumps(payload)


# ── CL02: Forms ──────────────────────────────────────────────────────────────

def slice_fields_with_labels(payload: dict) -> str:
    """Return all fields that have an effective_label."""
    fields = [
        f
        for form in payload["forms"]
        for f in form["fields"]
        if f.get("effective_label")
    ]
    return _dumps(fields)


def slice_placeholder_only_fields(payload: dict) -> str:
    """Return fields where label_source is 'placeholder_only'."""
    fields = [
        f
        for form in payload["forms"]
        for f in form["fields"]
        if f.get("label_source") == "placeholder_only"
    ]
    return _dumps(fields)


def slice_form_groups(payload: dict) -> str:
    """Return all fieldset/legend groups across all forms."""
    groups = [
        g
        for form in payload["forms"]
        for g in form["groups"]
    ]
    return _dumps(groups)


def slice_required_fields(payload: dict) -> str:
    """Return fields where required is True."""
    fields = [
        f
        for form in payload["forms"]
        for f in form["fields"]
        if f.get("required")
    ]
    return _dumps(fields)


def slice_fields_with_instructions(payload: dict) -> str:
    """Return fields that have non-null instructions (aria-describedby text)."""
    fields = [
        f
        for form in payload["forms"]
        for f in form["fields"]
        if f.get("instructions")
    ]
    return _dumps(fields)


def slice_cl02_full(payload: dict) -> str:
    """Return the full CL02 payload for the forms summary prompt."""
    return _dumps(payload)


# ── CL03: Non-text Content ───────────────────────────────────────────────────

def slice_informative_images(payload: dict) -> str:
    """Return informative images (non-empty alt text)."""
    return _dumps(payload["images"]["informative"])


def slice_decorative_images(payload: dict) -> str:
    """Return decorative images (empty alt)."""
    return _dumps(payload["images"]["decorative"])


def slice_actionable_images(payload: dict) -> str:
    """Return actionable images (inside links or buttons)."""
    return _dumps(payload["images"]["actionable"])


def slice_complex_images(payload: dict) -> str:
    """Return complex images (charts, diagrams, etc.)."""
    return _dumps(payload["images"]["complex"])


def slice_svgs(payload: dict) -> str:
    """Return non-hidden SVG elements."""
    return _dumps(payload["svgs"])


def slice_icon_fonts(payload: dict) -> str:
    """Return icon font elements."""
    return _dumps(payload["icon_fonts"])


def slice_media(payload: dict) -> str:
    """Return video and audio elements."""
    return _dumps(payload["media"])


def slice_cl03_full(payload: dict) -> str:
    """Return the full CL03 payload for the non-text summary prompt."""
    return _dumps(payload)


# ── Slicer lookup ─────────────────────────────────────────────────────────────

def get_slicer(name: str):
    """Return the slicer function by name.

    Raises KeyError if the name is not found.
    """
    slicer = _SLICER_MAP.get(name)
    if slicer is None:
        raise KeyError(f"Unknown slicer: {name!r}. Available: {sorted(_SLICER_MAP)}")
    return slicer


_SLICER_MAP = {
    # CL01
    "slice_page_title": slice_page_title,
    "slice_headings": slice_headings,
    "slice_flagged_links": slice_flagged_links,
    "slice_tables": slice_tables,
    "slice_iframes": slice_iframes,
    "slice_landmarks": slice_landmarks,
    "slice_cl01_full": slice_cl01_full,
    # CL02
    "slice_fields_with_labels": slice_fields_with_labels,
    "slice_placeholder_only_fields": slice_placeholder_only_fields,
    "slice_form_groups": slice_form_groups,
    "slice_required_fields": slice_required_fields,
    "slice_fields_with_instructions": slice_fields_with_instructions,
    "slice_cl02_full": slice_cl02_full,
    # CL03
    "slice_informative_images": slice_informative_images,
    "slice_decorative_images": slice_decorative_images,
    "slice_actionable_images": slice_actionable_images,
    "slice_complex_images": slice_complex_images,
    "slice_svgs": slice_svgs,
    "slice_icon_fonts": slice_icon_fonts,
    "slice_media": slice_media,
    "slice_cl03_full": slice_cl03_full,
}
