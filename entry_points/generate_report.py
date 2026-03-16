"""Combine programmatic + LLM findings into a single flat CSV report.

Reads output/manifest.json, output/programmatic_findings.json, and
output/prompts/*.json, normalizes all findings into flat CSV rows, and
writes to test_results/claude/report_YYYY-MM-DD.csv.

Usage:
    python entry_points/generate_report.py
    python entry_points/generate_report.py --output-dir ./output --report-dir ./test_results/claude/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, fields
from pathlib import Path


# ── CSV schema ──────────────────────────────────────────────────────────────

@dataclass
class ReportRow:
    """One row in the final CSV report (13 columns from CLAUDE.md spec)."""

    ID: int = 0
    element_name: str = ""
    browser_combination: str = "N/A"
    page_title: str = ""
    issue_title: str = ""
    steps_to_reproduce: str = ""
    actual_result: str = ""
    expected_result: str = ""
    recommendation: str = ""
    wcag_sc: str = ""
    category: str = ""
    log_date: str = ""
    reported_by: str = ""


CSV_COLUMNS = [f.name for f in fields(ReportRow)]


# ── Helpers ─────────────────────────────────────────────────────────────────

def strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` wrappers from an LLM response string."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _repair_json(text: str) -> str:
    """Attempt to repair common LLM JSON malformations.

    Handles the pattern where the model writes alternatives inline:
      "value": "some text" or similar alternative text
    by truncating at the first unquoted `or` after a closing quote.
    """
    # Fix: "string value" or alternative text  →  "string value"
    # This matches a quoted string followed by unquoted text before a comma/bracket.
    text = re.sub(
        r'("(?:[^"\\]|\\.)*")\s+or\s+[^,\]\}]+',
        r'\1',
        text,
    )
    return text


def safe_parse_json(text: str):
    """Parse JSON from an LLM response, stripping code fences first.

    Falls back to a repair pass if initial parsing fails due to common
    LLM malformations like inline alternative text.
    """
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = _repair_json(cleaned)
        return json.loads(repaired)


def load_prompt_file(path: Path) -> dict | None:
    """Load a prompt output JSON file and return the parsed structure."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


def extract_page_title_from_payload(prompt_data: dict) -> str:
    """Pull the page title string from the page_title prompt's payload_slice."""
    try:
        payload = json.loads(prompt_data["payload_slice"])
        return payload.get("title", "")
    except (json.JSONDecodeError, KeyError):
        return ""


def _get_recommendation(item: dict) -> str:
    """Extract the best fix recommendation from an LLM finding.

    Checks recommended_fix first (the new field from updated prompts),
    then falls back to existing field names for backwards compatibility.
    """
    return (
        item.get("recommended_fix")
        or item.get("suggested_improvement")
        or item.get("recommendation")
        or item.get("improved_example")
        or item.get("fix")
        or ""
    )


# ── Programmatic findings normalizer ───────────────────────────────────────

def normalize_programmatic(findings: list[dict], page_title: str,
                           log_date: str) -> list[ReportRow]:
    """Convert programmatic_findings.json entries to ReportRow objects.

    Supports both the new schema (rule_id/rule_name/location/description)
    and the legacy schema (issue_code/checklist_item/element/wcag) for
    backwards compatibility.
    """
    rows = []
    for f in findings:
        # ── Resolve element / location ──────────────────────────────────
        location = f.get("location") or f.get("element") or {}
        tag = location.get("tag", "")
        el_id = location.get("id", "")
        el_classes = location.get("class")
        snippet = location.get("text_preview") or location.get("snippet", "")

        if el_id:
            element_name = f"<{tag} id=\"{el_id}\">"
        elif el_classes:
            class_str = " ".join(el_classes) if isinstance(el_classes, list) else str(el_classes)
            element_name = f"<{tag} class=\"{class_str}\">"
        elif tag:
            element_name = f"<{tag}>"
        else:
            element_name = "(unknown)"

        # ── Resolve identifiers ─────────────────────────────────────────
        rule_id = f.get("rule_id") or f.get("issue_code", "")
        rule_name = f.get("rule_name") or f.get("checklist_item", "")
        description = f.get("description", "")

        # ── Resolve WCAG criterion (legacy schema has it, new one may not)
        wcag = f.get("wcag", {})
        criterion = wcag.get("criterion", "")
        wcag_name = wcag.get("name", "")

        # ── Build row ───────────────────────────────────────────────────
        issue_title = f"{rule_id}: {rule_name}" if rule_id else rule_name
        actual = description or rule_name
        if criterion:
            expected = f"Element should meet WCAG {criterion} ({wcag_name})"
        else:
            expected = f"Element should pass rule {rule_id}"
        category_suffix = wcag_name if wcag_name else rule_id
        category = f"Programmatic / {category_suffix}" if category_suffix else "Programmatic"

        rows.append(ReportRow(
            element_name=element_name,
            page_title=page_title,
            issue_title=issue_title,
            steps_to_reproduce=f"Inspect element: {snippet[:200]}",
            actual_result=actual,
            expected_result=expected,
            recommendation=description or rule_name,
            wcag_sc=criterion,
            category=category,
            log_date=log_date,
            reported_by="Programmatic",
        ))
    return rows


# ── LLM prompt normalizers ─────────────────────────────────────────────────

def _norm_page_title(data: dict, wcag: str, **ctx) -> list[ReportRow]:
    """Normalize page_title prompt response."""
    rows = []
    for issue in data.get("issues", []):
        rows.append(ReportRow(
            element_name="<title>",
            issue_title=f"Page Title: {issue}",
            actual_result=issue,
            expected_result="Page title should be descriptive and match H1 content",
            recommendation=_get_recommendation(data),
            wcag_sc=wcag,
            category="Semantic Structure / Page Title",
        ))
    return rows


def _norm_heading_structure(data: dict, wcag: str, **ctx) -> list[ReportRow]:
    """Normalize heading_structure prompt response."""
    rows = []
    for issue in data.get("issues", []):
        rows.append(ReportRow(
            element_name="<h1>-<h6>",
            issue_title=f"Heading Structure: {issue}",
            actual_result=issue,
            expected_result="Headings should form a logical content outline",
            recommendation=issue,
            wcag_sc=wcag,
            category="Semantic Structure / Headings",
        ))
    for heading in data.get("vague_headings", []):
        rows.append(ReportRow(
            element_name="<h1>-<h6>",
            issue_title=f"Vague heading: \"{heading}\"",
            actual_result=f"Heading \"{heading}\" is vague or unclear",
            expected_result="Headings should meaningfully describe their sections",
            recommendation=f"Replace \"{heading}\" with a more descriptive heading",
            wcag_sc=wcag,
            category="Semantic Structure / Headings",
        ))
    return rows


def _norm_link_clarity(data: list, wcag: str, **ctx) -> list[ReportRow]:
    """Normalize link_clarity prompt response."""
    rows = []
    for item in data:
        if item.get("is_clear", True):
            continue
        text = item.get("text") or "(no text)"
        rows.append(ReportRow(
            element_name=f"<a> \"{text}\"",
            issue_title=f"Unclear link: \"{text}\"",
            actual_result=item.get("reason", ""),
            expected_result="Link text should clearly describe its destination when read alone",
            recommendation=_get_recommendation(item),
            wcag_sc=wcag,
            category="Semantic Structure / Links",
        ))
    return rows


def _norm_iframe_titles(data: list, wcag: str, **ctx) -> list[ReportRow]:
    """Normalize iframe_titles prompt response."""
    rows = []
    for item in data:
        if item.get("is_descriptive", True):
            continue
        title = item.get("title") or "(no title)"
        rows.append(ReportRow(
            element_name=f"<iframe> \"{title}\"",
            issue_title=f"Non-descriptive iframe title: \"{title}\"",
            actual_result=item.get("reason", ""),
            expected_result="Iframe title should clearly describe the iframe content",
            recommendation=_get_recommendation(item),
            wcag_sc=wcag,
            category="Semantic Structure / Iframes",
        ))
    return rows


def _norm_landmark_structure(data: dict, wcag: str, **ctx) -> list[ReportRow]:
    """Normalize landmark_structure prompt response."""
    rows = []
    for issue in data.get("issues", []):
        rows.append(ReportRow(
            element_name="<main>/<nav>/<header>/<footer>",
            issue_title=f"Landmark issue: {issue}",
            actual_result=issue,
            expected_result="Landmark structure should be appropriate and balanced",
            recommendation=issue,
            wcag_sc=wcag,
            category="Semantic Structure / Landmarks",
        ))
    return rows


def _norm_label_quality(data: list, wcag: str, **ctx) -> list[ReportRow]:
    """Normalize label_quality prompt response."""
    rows = []
    for item in data:
        if item.get("is_descriptive", True):
            continue
        field_id = item.get("field_id") or "unknown"
        field_type = item.get("field_type", "input")
        label = item.get("effective_label") or "(no label)"
        issues = item.get("issues", [])
        rows.append(ReportRow(
            element_name=f"<{field_type} id=\"{field_id}\">",
            issue_title=f"Poor label quality: \"{label}\"",
            actual_result="; ".join(issues),
            expected_result="Form field labels should be descriptive and meaningful",
            recommendation=_get_recommendation(item),
            wcag_sc=wcag,
            category="Forms / Label Quality",
        ))
    return rows


def _norm_required_field_indicators(data: list, wcag: str, **ctx) -> list[ReportRow]:
    """Normalize required_field_indicators prompt response."""
    rows = []
    for item in data:
        issues = item.get("issues", [])
        if not issues:
            continue
        field_id = item.get("field_id") or "unknown"
        label = item.get("effective_label") or "(no label)"
        rows.append(ReportRow(
            element_name=f"<input id=\"{field_id}\">",
            issue_title=f"Required field not clearly indicated: \"{label}\"",
            actual_result="; ".join(issues),
            expected_result="Required field status should be communicated visually and programmatically",
            recommendation=_get_recommendation(item),
            wcag_sc=wcag,
            category="Forms / Required Fields",
        ))
    return rows


def _norm_informative_alt_quality(data: list, wcag: str, **ctx) -> list[ReportRow]:
    """Normalize informative_alt_quality prompt response."""
    rows = []
    for item in data:
        issues = item.get("issues", [])
        if not issues:
            continue
        src = item.get("src", "")
        alt = item.get("alt", "")
        rows.append(ReportRow(
            element_name=f"<img src=\"{src}\">",
            issue_title=f"Poor alt text quality ({item.get('quality', 'poor')}): \"{alt}\"",
            actual_result="; ".join(issues),
            expected_result="Alt text should accurately and concisely describe image content",
            recommendation=_get_recommendation(item),
            wcag_sc=wcag,
            category="Non-text Content / Informative Images",
        ))
    return rows


def _norm_decorative_verification(data: list, wcag: str, **ctx) -> list[ReportRow]:
    """Normalize decorative_verification prompt response."""
    rows = []
    for item in data:
        if item.get("likely_decorative", True):
            continue
        src = item.get("src", "")
        rows.append(ReportRow(
            element_name=f"<img src=\"{src}\" alt=\"\">",
            issue_title=f"Possibly mis-marked as decorative: {src}",
            actual_result=item.get("reason", ""),
            expected_result="Image marked as decorative (alt=\"\") should truly be decorative",
            recommendation=_get_recommendation(item),
            wcag_sc=wcag,
            category="Non-text Content / Decorative Verification",
        ))
    return rows


def _norm_actionable_image_alt(data: list, wcag: str, **ctx) -> list[ReportRow]:
    """Normalize actionable_image_alt prompt response."""
    rows = []
    for item in data:
        issues = item.get("issues", [])
        if not issues:
            continue
        src = item.get("src", "")
        context = item.get("context", "in_link")
        alt = item.get("alt") or "(empty)"
        rows.append(ReportRow(
            element_name=f"<img src=\"{src}\"> ({context})",
            issue_title=f"Actionable image alt issue: \"{alt}\"",
            actual_result="; ".join(issues),
            expected_result="Images in links/buttons should describe the action/destination, not appearance",
            recommendation=_get_recommendation(item),
            wcag_sc=wcag,
            category="Non-text Content / Actionable Images",
        ))
    return rows


def _norm_svg_accessibility(data: list, wcag: str, **ctx) -> list[ReportRow]:
    """Normalize svg_accessibility prompt response."""
    rows = []
    for item in data:
        issues = item.get("issues", [])
        if not issues:
            continue
        label = item.get("aria_label") or item.get("title") or "(unlabeled)"
        rows.append(ReportRow(
            element_name=f"<svg> \"{label}\"",
            issue_title=f"SVG accessibility issue: {label}",
            actual_result="; ".join(issues),
            expected_result="SVGs should have role=\"img\" and an accessible name via title + aria-labelledby",
            recommendation=_get_recommendation(item),
            wcag_sc=wcag,
            category="Non-text Content / SVGs",
        ))
    return rows


def _norm_icon_font_accessibility(data: list, wcag: str, **ctx) -> list[ReportRow]:
    """Normalize icon_font_accessibility prompt response."""
    rows = []
    for item in data:
        issues = item.get("issues", [])
        if not issues:
            continue
        classes = item.get("classes", "")
        pattern = item.get("pattern", "")
        rows.append(ReportRow(
            element_name=f"<i class=\"{classes}\">",
            issue_title=f"Icon font issue ({pattern}): {classes}",
            actual_result="; ".join(issues),
            expected_result="Icon fonts should be properly labeled or hidden from assistive technology",
            recommendation=_get_recommendation(item),
            wcag_sc=wcag,
            category="Non-text Content / Icon Fonts",
        ))
    return rows


# ── Normalizer registry ────────────────────────────────────────────────────

NORMALIZERS: dict[str, callable] = {
    "page_title": _norm_page_title,
    "heading_structure": _norm_heading_structure,
    "link_clarity": _norm_link_clarity,
    "iframe_titles": _norm_iframe_titles,
    "landmark_structure": _norm_landmark_structure,
    "label_quality": _norm_label_quality,
    "required_field_indicators": _norm_required_field_indicators,
    "informative_alt_quality": _norm_informative_alt_quality,
    "decorative_verification": _norm_decorative_verification,
    "actionable_image_alt": _norm_actionable_image_alt,
    "svg_accessibility": _norm_svg_accessibility,
    "icon_font_accessibility": _norm_icon_font_accessibility,
}


# ── Main report generation ─────────────────────────────────────────────────

def generate_report(output_dir: Path, report_dir: Path) -> Path:
    """Generate a unified CSV report from pipeline output.

    Args:
        output_dir: Directory containing manifest.json, programmatic_findings.json,
                    and prompts/ subdirectory.
        report_dir: Directory to write the CSV report into.

    Returns:
        Path to the written CSV file.
    """
    # 1. Load manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    run_timestamp = manifest.get("run_timestamp", "")
    log_date = run_timestamp[:10] if run_timestamp else "unknown"
    model = manifest.get("model", "unknown")

    # 2. Extract page title from page_title prompt payload
    page_title_data = load_prompt_file(output_dir / "prompts" / "page_title.json")
    page_title = ""
    if page_title_data:
        page_title = extract_page_title_from_payload(page_title_data)
    if not page_title:
        # Fallback to HTML filename
        page_title = Path(manifest.get("html_file", "unknown")).stem

    # 3. Normalize programmatic findings
    prog_path = output_dir / "programmatic_findings.json"
    prog_findings = []
    if prog_path.exists():
        with open(prog_path, encoding="utf-8") as f:
            prog_raw = json.load(f)
        prog_findings = normalize_programmatic(prog_raw, page_title, log_date)

    # 4. Normalize LLM prompt results
    llm_findings: list[ReportRow] = []
    prompts_dir = output_dir / "prompts"

    for prompt_entry in manifest.get("prompts_executed", []):
        name = prompt_entry["name"]
        wcag_list = prompt_entry.get("wcag_criteria", [])
        wcag_str = ", ".join(wcag_list)

        normalizer = NORMALIZERS.get(name)
        if normalizer is None:
            continue

        prompt_file = prompts_dir / f"{name}.json"
        prompt_data = load_prompt_file(prompt_file)
        if prompt_data is None:
            continue

        api_result = prompt_data.get("api_result", {})
        if not api_result.get("success", False):
            continue

        response_text = api_result.get("response", "")
        try:
            parsed = safe_parse_json(response_text)
        except (json.JSONDecodeError, ValueError):
            print(f"  WARNING: Could not parse JSON response for {name}, skipping")
            continue

        new_rows = normalizer(parsed, wcag=wcag_str)

        # Fill in shared fields
        for row in new_rows:
            row.page_title = page_title
            row.log_date = log_date
            row.reported_by = model

        llm_findings.extend(new_rows)

    # 5. Combine and assign sequential IDs
    all_rows = prog_findings + llm_findings
    for i, row in enumerate(all_rows, start=1):
        row.ID = i

    # 6. Write CSV
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"report_{log_date}.csv"

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({col: getattr(row, col) for col in CSV_COLUMNS})

    # 7. Print summary
    print(f"Report generated: {report_path}")
    print(f"  Programmatic findings: {len(prog_findings)}")
    print(f"  LLM findings:         {len(llm_findings)}")
    print(f"  Total rows:           {len(all_rows)}")
    print(f"  Log date:             {log_date}")
    print(f"  Model:                {model}")

    return report_path


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point for report generation."""
    parser = argparse.ArgumentParser(
        description="Generate a unified CSV report from pipeline output."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Directory containing pipeline output (default: ./output)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("./test_results/claude"),
        help="Directory to write the CSV report (default: ./test_results/claude/)",
    )
    args = parser.parse_args()
    generate_report(args.output_dir, args.report_dir)


if __name__ == "__main__":
    main()
