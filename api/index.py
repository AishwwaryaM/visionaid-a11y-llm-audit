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
async def audit_url(request: Request):
    data = await request.json()
    url = data.get("url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    async def event_generator():
        # Port your logic from _handle_url_audit here using 'yield'
        # instead of 'self.wfile.write()'
        yield (json.dumps({"type": "progress", "message": f"Fetching {url}..."}) + "\n").encode("utf-8")

        html_content = fetch_page(url)
        # ... call run_audit and yield result ...

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

# --- STATIC FILE SERVING ---
# Vercel serves index.html and styles.css automatically if they are in the root.
# You don't need code for this anymore!