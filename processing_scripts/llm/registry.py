"""Registry mapping each evaluation task to its prompt template, payload slicer, and metadata."""

from dataclasses import dataclass, field


@dataclass
class PromptSpec:
    """Defines one element-specific evaluation task.

    Attributes:
        name: Unique identifier, e.g. "link_clarity".
        checklist: Which extractor payload to use — "CL01", "CL02", or "CL03".
        prompt_file: Relative path to the .txt file containing the prompt template.
        prompt_index: 1-based index of the prompt within the .txt file.
        payload_slicer: Name of the slicer function in slicers.py.
        wcag_criteria: WCAG success criteria this prompt evaluates.
        element_types: HTML element tags this prompt is relevant to.
        output_type: Expected JSON response shape — "array", "object", or "summary".
        is_summary: If True, only run when --include-summaries is set.
        skip_if_empty: If True, skip the API call when the payload slice is empty.
    """

    name: str
    checklist: str
    prompt_file: str
    prompt_index: int
    payload_slicer: str
    wcag_criteria: list[str] = field(default_factory=list)
    element_types: list[str] = field(default_factory=list)
    output_type: str = "array"
    is_summary: bool = False
    skip_if_empty: bool = True


PROMPT_REGISTRY: list[PromptSpec] = [
    # ── CL01: Semantic Structure ──────────────────────────────────────────────
    PromptSpec(
        name="page_title",
        checklist="CL01",
        prompt_file="processing_scripts/llm/semantic_checklist_01.txt",
        prompt_index=1,
        payload_slicer="slice_page_title",
        wcag_criteria=["2.4.2"],
        element_types=["title", "h1"],
        output_type="object",
    ),
    PromptSpec(
        name="heading_structure",
        checklist="CL01",
        prompt_file="processing_scripts/llm/semantic_checklist_01.txt",
        prompt_index=2,
        payload_slicer="slice_headings",
        wcag_criteria=["1.3.1", "2.4.6"],
        element_types=["h1", "h2", "h3", "h4", "h5", "h6"],
        output_type="object",
    ),
    PromptSpec(
        name="link_clarity",
        checklist="CL01",
        prompt_file="processing_scripts/llm/semantic_checklist_01.txt",
        prompt_index=3,
        payload_slicer="slice_flagged_links",
        wcag_criteria=["2.4.4"],
        element_types=["a"],
        output_type="array",
    ),
    PromptSpec(
        name="table_semantics",
        checklist="CL01",
        prompt_file="processing_scripts/llm/semantic_checklist_01.txt",
        prompt_index=4,
        payload_slicer="slice_tables",
        wcag_criteria=["1.3.1"],
        element_types=["table", "th", "caption"],
        output_type="array",
    ),
    PromptSpec(
        name="iframe_titles",
        checklist="CL01",
        prompt_file="processing_scripts/llm/semantic_checklist_01.txt",
        prompt_index=5,
        payload_slicer="slice_iframes",
        wcag_criteria=["4.1.2"],
        element_types=["iframe"],
        output_type="array",
    ),
    PromptSpec(
        name="landmark_structure",
        checklist="CL01",
        prompt_file="processing_scripts/llm/semantic_checklist_01.txt",
        prompt_index=6,
        payload_slicer="slice_landmarks",
        wcag_criteria=["1.3.1"],
        element_types=["main", "nav", "header", "footer", "aside"],
        output_type="object",
    ),
    PromptSpec(
        name="semantic_summary",
        checklist="CL01",
        prompt_file="processing_scripts/llm/semantic_checklist_01.txt",
        prompt_index=7,
        payload_slicer="slice_cl01_full",
        wcag_criteria=["1.3.1", "2.4.2", "2.4.4", "2.4.6", "3.1.1", "4.1.2"],
        element_types=[],
        output_type="summary",
        is_summary=True,
    ),
    # ── CL02: Forms ───────────────────────────────────────────────────────────
    PromptSpec(
        name="label_quality",
        checklist="CL02",
        prompt_file="processing_scripts/llm/forms_checklist_02.txt",
        prompt_index=1,
        payload_slicer="slice_fields_with_labels",
        wcag_criteria=["1.3.1", "2.4.6"],
        element_types=["input", "select", "textarea", "label"],
        output_type="array",
    ),
    PromptSpec(
        name="placeholder_as_label",
        checklist="CL02",
        prompt_file="processing_scripts/llm/forms_checklist_02.txt",
        prompt_index=2,
        payload_slicer="slice_placeholder_only_fields",
        wcag_criteria=["1.3.1"],
        element_types=["input", "select", "textarea"],
        output_type="array",
    ),
    PromptSpec(
        name="group_labels",
        checklist="CL02",
        prompt_file="processing_scripts/llm/forms_checklist_02.txt",
        prompt_index=3,
        payload_slicer="slice_form_groups",
        wcag_criteria=["1.3.1"],
        element_types=["fieldset", "legend"],
        output_type="array",
    ),
    PromptSpec(
        name="required_field_indicators",
        checklist="CL02",
        prompt_file="processing_scripts/llm/forms_checklist_02.txt",
        prompt_index=4,
        payload_slicer="slice_required_fields",
        wcag_criteria=["3.3.2"],
        element_types=["input", "select", "textarea"],
        output_type="array",
    ),
    PromptSpec(
        name="form_instructions",
        checklist="CL02",
        prompt_file="processing_scripts/llm/forms_checklist_02.txt",
        prompt_index=5,
        payload_slicer="slice_fields_with_instructions",
        wcag_criteria=["3.3.2"],
        element_types=["input", "select", "textarea"],
        output_type="array",
    ),
    PromptSpec(
        name="form_summary",
        checklist="CL02",
        prompt_file="processing_scripts/llm/forms_checklist_02.txt",
        prompt_index=6,
        payload_slicer="slice_cl02_full",
        wcag_criteria=["1.3.1", "2.4.6", "3.3.2"],
        element_types=[],
        output_type="summary",
        is_summary=True,
    ),
    # ── CL03: Non-text Content ────────────────────────────────────────────────
    PromptSpec(
        name="informative_alt_quality",
        checklist="CL03",
        prompt_file="processing_scripts/llm/nontext_checklist_03.txt",
        prompt_index=1,
        payload_slicer="slice_informative_images",
        wcag_criteria=["1.1.1"],
        element_types=["img"],
        output_type="array",
    ),
    PromptSpec(
        name="decorative_verification",
        checklist="CL03",
        prompt_file="processing_scripts/llm/nontext_checklist_03.txt",
        prompt_index=2,
        payload_slicer="slice_decorative_images",
        wcag_criteria=["1.1.1"],
        element_types=["img"],
        output_type="array",
    ),
    PromptSpec(
        name="actionable_image_alt",
        checklist="CL03",
        prompt_file="processing_scripts/llm/nontext_checklist_03.txt",
        prompt_index=3,
        payload_slicer="slice_actionable_images",
        wcag_criteria=["1.1.1", "2.4.4"],
        element_types=["img", "a", "button"],
        output_type="array",
    ),
    PromptSpec(
        name="complex_descriptions",
        checklist="CL03",
        prompt_file="processing_scripts/llm/nontext_checklist_03.txt",
        prompt_index=4,
        payload_slicer="slice_complex_images",
        wcag_criteria=["1.1.1"],
        element_types=["img"],
        output_type="array",
    ),
    PromptSpec(
        name="svg_accessibility",
        checklist="CL03",
        prompt_file="processing_scripts/llm/nontext_checklist_03.txt",
        prompt_index=5,
        payload_slicer="slice_svgs",
        wcag_criteria=["1.1.1"],
        element_types=["svg"],
        output_type="array",
    ),
    PromptSpec(
        name="icon_font_accessibility",
        checklist="CL03",
        prompt_file="processing_scripts/llm/nontext_checklist_03.txt",
        prompt_index=6,
        payload_slicer="slice_icon_fonts",
        wcag_criteria=["1.1.1"],
        element_types=["i", "span"],
        output_type="array",
    ),
    PromptSpec(
        name="media_captions",
        checklist="CL03",
        prompt_file="processing_scripts/llm/nontext_checklist_03.txt",
        prompt_index=7,
        payload_slicer="slice_media",
        wcag_criteria=["1.2.1", "1.2.2", "1.2.3"],
        element_types=["video", "audio"],
        output_type="array",
    ),
    PromptSpec(
        name="nontext_summary",
        checklist="CL03",
        prompt_file="processing_scripts/llm/nontext_checklist_03.txt",
        prompt_index=8,
        payload_slicer="slice_cl03_full",
        wcag_criteria=["1.1.1", "1.2.1", "1.2.2", "1.2.3"],
        element_types=[],
        output_type="summary",
        is_summary=True,
    ),
]
