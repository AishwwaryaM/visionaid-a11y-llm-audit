# Project: P5 – Automating Digital Accessibility

## What This Project Does

This project builds LLM-powered tools that analyze websites for WCAG accessibility issues and generate structured remediation reports. It is a Computing for Good course project at Georgia Tech (OMSCS) partnered with the Vision Aid Digital Accessibility Testing Team.

The system has three planned stages:
1. Automated webpage accessibility analysis → structured CSV reports
2. Code remediation suggestions based on those reports
3. (Stretch) Chrome extension for on-the-fly accessibility fixes

## Architecture Overview

```
project-root/
├── index.html                       # Team/project website (served via GitHub Pages)
├── styles.css                       # Website styles
├── processing_scripts/
│   ├── llm/                         # Modular prompt system
│   │   ├── registry.py              # PromptSpec dataclass + PROMPT_REGISTRY (21 specs)
│   │   ├── slicers.py               # Payload slicer functions (one per prompt)
│   │   ├── templates.py             # Prompt .txt parser + {payload} filler
│   │   ├── semantic_checklist_01.txt # 7 prompts for semantic structure (CL01)
│   │   ├── forms_checklist_02.txt   # 6 prompts for forms (CL02)
│   │   └── nontext_checklist_03.txt # 8 prompts for non-text content (CL03)
│   ├── llm_preprocessing/           # HTML → structured JSON extractors
│   │   ├── semantic_checklist_01.py  # CL01 extractor
│   │   ├── forms_checklist_02.py    # CL02 extractor
│   │   └── nontext_checklist_03.py  # CL03 extractor
│   └── programmatic/                # Rule-based checks (no API needed)
│       └── semantic_checklist_01.py  # Missing alt, duplicate IDs, etc.
├── entry_points/
│   ├── run_pipeline.py              # Full pipeline orchestrator
│   ├── generate_report.py           # JSON → CSV report generator
│   └── get_visionaid_home.py        # HTML downloader
├── test_files/                      # HTML files to analyze (may be very large, 500K+ tokens)
└── docs/
    └── modular-prompts-plan.md      # Architecture plan
```

## Website (GitHub Pages)

The team project website lives at the repo root (`index.html` + `styles.css`). It is a pure HTML5/CSS3 static site with no build tools or JavaScript frameworks — served directly via GitHub Pages from the main branch root.

- Design: DM Serif Display + DM Sans, teal/amber palette, fully responsive
- Accessibility: skip link, semantic HTML, ARIA landmarks, WCAG AA contrast
- Lighthouse scores (last measured): Performance 92 · Accessibility 96 · Best Practices 100 · SEO 100
- Do not add JavaScript or external CSS frameworks to the website

## Tech Stack

- Python 3.x (use virtual environment: `source venv/bin/activate`)
- Anthropic API (Claude Sonnet 4 preferred for testing — `claude-sonnet-4-20250514`)
- HTML preprocessing: BeautifulSoup / custom parsers (check imports in preprocessing files)
- Accessibility tools context: axe-core, Lighthouse, WAVE (hybrid approach)
- Output format: CSV with columns — ID, element_name, browser_combination, page_title, issue_title, steps_to_reproduce, actual_result, expected_result, recommendation, wcag_sc, category, log_date, reported_by

## Pipeline Usage

```bash
# Dry run (no API key needed) — generates prompts and payloads
python entry_points/run_pipeline.py --html test_files/home.html --dry-run

# Full run with API
python entry_points/run_pipeline.py --html test_files/home.html

# Generate CSV report from pipeline output
python entry_points/generate_report.py
```

## Code Patterns

- Prompt templates are `.txt` files with `{payload}` placeholders, parsed by `processing_scripts/llm/templates.py`
- The registry (`processing_scripts/llm/registry.py`) maps each evaluation task to its prompt file, slicer, and WCAG criteria
- Slicer functions (`processing_scripts/llm/slicers.py`) extract targeted JSON slices from extractor payloads
- The pipeline runner (`entry_points/run_pipeline.py`) orchestrates: programmatic checks → extraction → slicing → prompt filling → API calls
- File encoding: use `encoding='utf-8', errors='replace'` when reading HTML files

## Style Rules

- Python code: standard PEP 8
- Docstrings for all new functions
- Use `pathlib.Path` for file paths (consistent with existing scripts)
- Commit messages: descriptive, prefixed with area (e.g., "prompts: add element-specific template system")

## Do NOT

- Do not call the Anthropic API or spend any money unless explicitly testing
- Do not install new dependencies without team agreement
- Do not modify test_files/ or any HTML input files

## Common Issues

- HTML test files can be enormous (500K+ tokens) — do not try to read them fully into context
- Some files may have non-UTF-8 encoding; always use `errors='replace'`
