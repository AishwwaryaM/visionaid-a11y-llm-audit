#!/usr/bin/env python3
"""Serve the team website and run the accessibility audit API locally.

This server has two responsibilities:
  1. Serve static files (index.html, styles.css) from the project root.
  2. Handle POST /api/audit — receive raw HTML, run the pipeline, return JSON.

Usage:
    python entry_points/api_server.py
    python entry_points/api_server.py --port 8080

Then open http://localhost:8000 in your browser.

The ANTHROPIC_API_KEY can be supplied per-request in the POST body, or set in
the .env file / environment. Per-request keys take priority.
"""

import json
import os
import re
import shutil
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

# ── Project root and sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from entry_points.run_pipeline import run_pipeline  # noqa: E402
from entry_points.generate_report import generate_report  # noqa: E402

STATIC_DIR = PROJECT_ROOT  # index.html and styles.css live at the repo root


# ── Audit logic ───────────────────────────────────────────────────────────────

def run_audit(html_content: str, api_key: str, model: str) -> dict:
    """Write *html_content* to a temp file, run the pipeline, return results.

    If *api_key* is empty, the pipeline runs in dry-run mode (programmatic
    checks only, no LLM calls).
    """
    dry_run = not api_key
    tmp_dir = tempfile.mkdtemp(prefix="visionaid_audit_")
    try:
        html_path = Path(tmp_dir) / "input.html"
        html_path.write_text(html_content, encoding="utf-8")
        output_dir = Path(tmp_dir) / "output"

        manifest = run_pipeline(
            html_path=str(html_path),
            output_dir=output_dir,
            api_key=api_key if api_key else None,
            model=model,
            dry_run=dry_run,
            include_summaries=False,
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
                report_dir = Path(tmp_dir) / "reports"
                report_path = generate_report(output_dir, report_dir)
                csv_content = report_path.read_text(encoding="utf-8")
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


# ── HTTP handler ──────────────────────────────────────────────────────────────

class AuditHandler(BaseHTTPRequestHandler):
    """Handle static file serving and the /api/audit endpoint."""

    def log_message(self, fmt, *args):  # noqa: N802
        sys.stderr.write(f"[{self.address_string()}] {fmt % args}\n")

    # CORS helpers ─────────────────────────────────────────────────────────────

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):  # noqa: N802
        """Handle CORS preflight requests."""
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    # GET ──────────────────────────────────────────────────────────────────────

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/styles.css":
            self._serve_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
        else:
            self.send_error(404, "Not Found")

    def _serve_file(self, file_path: Path, content_type: str):
        if not file_path.exists():
            self.send_error(404, "Not Found")
            return
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    # POST ─────────────────────────────────────────────────────────────────────

    def do_POST(self):  # noqa: N802
        if self.path == "/api/audit":
            self._handle_audit()
        else:
            self.send_error(404, "Not Found")

    def _handle_audit(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"success": False, "error": "Invalid JSON body"}, 400)
            return

        html_content = data.get("html_content", "").strip()
        if not html_content:
            self._send_json(
                {"success": False, "error": "html_content is required"}, 400
            )
            return

        # Per-request key takes priority; fall back to environment
        api_key = (
            data.get("api_key", "").strip()
            or os.getenv("ANTHROPIC_API_KEY", "")
        )
        model = data.get("model", "claude-haiku-4-5-20251001")

        print(
            f"  Audit request: {len(html_content):,} chars, "
            f"model={model}, api_key={'set' if api_key else 'not set'}"
        )

        result = run_audit(html_content, api_key, model)
        self._send_json(result)

    def _send_json(self, obj: dict, status: int = 200):
        body = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Start the HTTP server.

    Host and port can be set via CLI flags or environment variables:
      HOST  (default: 127.0.0.1, use 0.0.0.0 for cloud deployment)
      PORT  (default: 8000, set automatically by Render/Railway)
    """
    import argparse

    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Run the accessibility audit web server."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", 8000)),
        help="Port to listen on (default: $PORT or 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "127.0.0.1"),
        help="Host to bind to (default: $HOST or 127.0.0.1; use 0.0.0.0 for deployment)",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), AuditHandler)
    display_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print("Accessibility Audit Server")
    print(f"  URL  : http://{display_host}:{args.port}")
    print(f"  API  : POST http://{display_host}:{args.port}/api/audit")
    print("  Press Ctrl+C to stop.")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
