import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Project root and sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load Environment / Imports
from entry_points.run_pipeline import run_pipeline
from entry_points.generate_report import generate_report
from vision_aid.ingestion.file_crawler import fetch_page, fetch_pages_nested
from processing_scripts.llm_client.client import is_openai_model

app = FastAPI()

# Automatic CORS handling
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REUSE YOUR EXISTING LOGIC FUNCTIONS ---
# Copy-paste split_pages, _resolve_api_key, run_audit, and _try_parse_json
# EXACTLY as they are in your current script.

def _resolve_api_key(data: dict, model: str) -> str:
    """Return the appropriate API key for *model* from the request body or env.

    OpenAI models use the ``openai_api_key`` field / ``OPENAI_API_KEY`` env.
    Anthropic models use the ``api_key`` field / ``ANTHROPIC_API_KEY`` env.
    Per-request keys take priority over environment variables.
    """
    if is_openai_model(model):
        return (
            data.get("openai_api_key", "").strip()
            or os.getenv("OPENAI_API_KEY", "")
        )
    return (
        data.get("api_key", "").strip()
        or os.getenv("ANTHROPIC_API_KEY", "")
    )

def run_audit(html_content: str, api_key: str, model: str, progress_callback=None) -> dict:
    """Write *html_content* to a temp file, run the pipeline, return results.

    If *api_key* is empty, the pipeline runs in dry-run mode (programmatic
    checks only, no LLM calls).
    """
    dry_run = not api_key
    # Vercel allows writing to /tmp for ephemeral storage
    tmp_dir = tempfile.mkdtemp(prefix="visionaid_audit_")
    try:
        html_path = Path(tmp_dir) / "input.html"
        html_path.write_text(html_content, encoding="utf-8")
        tmp_dir = tempfile.mkdtemp(prefix="visionaid_audit_", dir="/tmp")

        manifest = run_pipeline(
            html_path=str(html_path),
            output_dir=output_dir,
            api_key=api_key if api_key else None,
            model=model,
            dry_run=dry_run,
            include_summaries=False,
            progress_callback=progress_callback,
        )

        # Read programmatic findings
        prog_path = output_dir / "programmatic_findings.json"
        programmatic_findings = (
            json.loads(prog_path.read_text(encoding="utf-8"))
            if prog_path.exists()
            else []
        )

        # Read per-prompt LLM results
        llm_results = {}
        prompts_dir = output_dir / "prompts"
        if prompts_dir.exists():
            for prompt_file in sorted(prompts_dir.glob("*.json")):
                data = json.loads(prompt_file.read_text(encoding="utf-8"))
                name = data.get("prompt_name", prompt_file.stem)
                api_result = data.get("api_result", {})
                parsed = None
                if api_result.get("success"):
                    parsed = _try_parse_json(api_result.get("response", ""))
                usage = api_result.get("usage", {})
                llm_results[name] = {
                    "checklist": data.get("checklist"),
                    "wcag_criteria": data.get("wcag_criteria", []),
                    "status": "success" if api_result.get("success") else "dry_run",
                    "parsed": parsed,
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "duration_seconds": api_result.get("duration_seconds"),
                }

        # Generate CSV report (only meaningful when LLM ran)
        csv_content = None
        if not dry_run:
            try:
                if progress_callback:
                    progress_callback({
                        "type": "progress",
                        "stage": "report_generating",
                        "message": "Generating report…",
                    })
                report_dir = Path(tmp_dir) / "reports"
                report_path = generate_report(output_dir, report_dir)
                csv_content = report_path.read_text(encoding="utf-8")
                if progress_callback:
                    progress_callback({
                        "type": "progress",
                        "stage": "report_complete",
                        "message": "Report ready",
                    })
            except Exception as csv_err:
                print(f"  Warning: CSV generation failed: {csv_err}")

        return {
            "success": True,
            "programmatic_findings": programmatic_findings,
            "llm_results": llm_results,
            "csv_report": csv_content,
            "skipped_prompts": manifest.get("prompts_skipped", []),
            "summary": {
                "programmatic_count": manifest.get("programmatic_findings_count", 0),
                "programmatic_by_checker": manifest.get(
                    "programmatic_findings_by_checker", {}
                ),
                "llm_prompts_run": len(llm_results),
                "llm_prompts_skipped": len(manifest.get("prompts_skipped", [])),
                "total_input_tokens": manifest.get("total_input_tokens", 0),
                "total_output_tokens": manifest.get("total_output_tokens", 0),
                "estimated_cost_usd": manifest.get("estimated_cost_usd"),
                "model": model,
                "dry_run": dry_run,
            },
        }

    except Exception as exc:
        return {"success": False, "error": str(exc)}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)



def _try_parse_json(text: str):
    """Parse JSON from an LLM response, stripping markdown code fences."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    cleaned = m.group(1).strip() if m else text.strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {"raw": text}

def split_pages(html: str) -> list[tuple[str, str]]:
    """Split concatenated HTML from ``fetch_pages_nested`` into per-page chunks.

    Returns a list of ``(url, html)`` tuples.  If no PAGE markers are found
    the entire string is returned as a single page with url ``"unknown"``.
    """
    markers = list(_PAGE_MARKER.finditer(html))
    if not markers:
        return [("unknown", html)]

    pages: list[tuple[str, str]] = []
    for i, m in enumerate(markers):
        url = m.group(1)
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(html)
        pages.append((url, html[start:end].strip()))
    return pages



# --- NEW FASTAPI ENDPOINTS ---

@app.post("/api/audit")
async def audit_html(request: Request):
    data = await request.json()

    async def event_generator():
        html_content = data.get("html_content", "").strip()
        model = data.get("model", "claude-haiku-4-5-20251001")
        api_key = _resolve_api_key(data, model)

        def send_event(obj):
            return (json.dumps(obj) + "\n").encode("utf-8")

        yield send_event({"type": "progress", "stage": "starting", "message": "Starting audit…"})

        # Use your existing run_audit
        result = run_audit(html_content, api_key, model,
                           progress_callback=lambda x: event_generator.yield_queue.append(x))
        # Note: run_audit is synchronous. For a simple migration, we'll run
        # the audit and return the result as the final line of the stream.

        result["type"] = "result"
        yield send_event(result)

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.post("/api/audit/url")
async def handle_url_audit(request: Request):
    try:
        data = await request.json()
        url = data.get("url", "").strip()
        if not url.startswith('http'): url = f"https://{url}"

        # Log for debugging
        print(f"Starting audit for: {url}")

        html_content = fetch_page(url)
        model = data.get("model", "claude-haiku-4-5-20251001")
        api_key = _resolve_api_key(data, model)

        # Pass a specific tmp directory to run_audit
        result = run_audit(html_content, api_key, model)

        return result

    except Exception as e:
        import traceback
        # This will print the EXACT error (like "Read-only file system") to Vercel Logs
        error_details = traceback.format_exc()
        print(error_details)
        return {
            "success": False,
            "error": str(e),
            "traceback": error_details if not os.getenv("PROD") else None
        }

# --- STATIC FILE SERVING ---
# Vercel serves index.html and styles.css automatically if they are in the root.
# You don't need code for this anymore!