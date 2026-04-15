import re
import json
from bs4 import BeautifulSoup

GENERIC_LINK_TERMS = {
    "click here", "here", "read more", "more",
    "learn more", "details", "link"
}

ARIA_LANDMARK_ROLES = {
    "banner", "navigation", "main", "complementary",
    "contentinfo", "search", "form", "region"
}

SKIP_INPUT_TYPES = {"hidden", "submit", "button", "reset", "image"}

_LANDMARK_TAGS = {"main", "nav", "header", "footer", "aside"}
_LANDMARK_ROLES = {"main", "navigation", "banner", "contentinfo", "complementary", "region", "search"}


def clean(text):
    return re.sub(r"\s+", " ", text.strip()) if text else ""


def get_nearest_heading(el):
    """Return the text of the nearest preceding heading in the DOM."""
    prev_headings = el.find_all_previous(re.compile(r"^h[1-6]$"))
    return clean(prev_headings[0].get_text()) if prev_headings else None


def get_landmark_region(el):
    """Return the label or tag name of the nearest landmark ancestor."""
    for ancestor in el.parents:
        if not ancestor.name:
            continue
        if ancestor.name in _LANDMARK_TAGS:
            return clean(ancestor.get("aria-label", "")) or ancestor.name
        role = ancestor.get("role", "").lower()
        if role in _LANDMARK_ROLES:
            return clean(ancestor.get("aria-label", "")) or role
    return None


def estimate_tokens(text):
    """Rough estimate: ~4 chars per token for English/HTML text."""
    return max(1, len(text) // 4)


def extract(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    payload = {}

    # ── LANGUAGE ──────────────────────────────────────────────────────────────
    html_tag = soup.find("html")
    payload["language"] = html_tag.get("lang", "") if html_tag else ""

    # ── TITLE + H1 ────────────────────────────────────────────────────────────
    payload["page_title"] = {
        "title": clean(soup.title.string) if soup.title else "",
        "h1": clean(soup.find("h1").get_text()) if soup.find("h1") else ""
    }

    # ── HEADINGS ──────────────────────────────────────────────────────────────
    payload["headings"] = [
        {"level": int(h.name[1]), "text": clean(h.get_text())}
        for h in soup.find_all(re.compile(r"^h[1-6]$"))
    ]

    # ── IMAGES ────────────────────────────────────────────────────────────────
    # Categorised by alt-text status so the LLM can evaluate quality
    images = {"missing_alt": [], "empty_alt": [], "has_alt": []}
    for img in soup.find_all("img"):
        alt = img.get("alt")
        src = img.get("src", "").split("/")[-1][:60]
        if alt is None:
            images["missing_alt"].append({"src": src})
        elif alt.strip() == "":
            images["empty_alt"].append({"src": src})
        else:
            images["has_alt"].append({"src": src, "alt": clean(alt)})
    payload["images"] = images

    # ── FLAGGED LINKS ─────────────────────────────────────────────────────────
    # Collects links that may lack a clear accessible name:
    #   • no visible text
    #   • generic term (click here, read more, …)
    #   • very short label (≤2 words) — often context-dependent
    flagged_links = []
    seen = set()
    for a in soup.find_all("a"):
        text = clean(a.get_text())
        aria = clean(a.get("aria-label", ""))
        effective = aria or text
        is_generic = effective.lower() in GENERIC_LINK_TERMS
        is_short = bool(effective) and len(effective.split()) <= 2
        if not effective or is_generic or is_short:
            key = (text, aria)
            if key not in seen:
                seen.add(key)
                flagged_links.append({
                    "text": text or None,
                    "aria_label": aria or None,
                    "href": a.get("href", "")[:120] or None,
                    "nearest_heading": get_nearest_heading(a),
                    "landmark_region": get_landmark_region(a),
                })
    payload["flagged_links"] = flagged_links

    # ── FORMS ─────────────────────────────────────────────────────────────────
    forms = []
    for form in soup.find_all("form"):
        fields = []
        for inp in form.find_all(["input", "select", "textarea"]):
            inp_type = inp.get("type", "text").lower()
            if inp_type in SKIP_INPUT_TYPES:
                continue
            inp_id = inp.get("id", "")
            label_el = soup.find("label", attrs={"for": inp_id}) if inp_id else None
            label_text = clean(label_el.get_text()) if label_el else None
            aria_label = clean(inp.get("aria-label", ""))
            aria_labelledby = inp.get("aria-labelledby", "")
            fields.append({
                "type": inp_type,
                "id": inp_id or None,
                "label": label_text,
                "aria_label": aria_label or None,
                "placeholder": clean(inp.get("placeholder", "")) or None,
                "has_accessible_name": bool(label_text or aria_label or aria_labelledby)
            })
        if fields:
            forms.append({"action": form.get("action", ""), "fields": fields})
    payload["forms"] = forms

    # ── BUTTONS ───────────────────────────────────────────────────────────────
    buttons = []
    seen_btns = set()
    for btn in soup.select("button, [role='button']"):
        text = clean(btn.get_text())
        aria = clean(btn.get("aria-label", ""))
        key = (text, aria)
        if key not in seen_btns:
            seen_btns.add(key)
            buttons.append({
                "text": text or None,
                "aria_label": aria or None,
                "has_label": bool(text or aria)
            })
    payload["buttons"] = buttons

    # ── LANDMARKS ─────────────────────────────────────────────────────────────
    landmarks = []
    for el in soup.find_all(["main", "nav", "header", "footer", "aside"]):
        landmarks.append({
            "tag": el.name,
            "aria_label": clean(el.get("aria-label", "")) or None,
        })
    for el in soup.find_all(attrs={"role": True}):
        role = el.get("role", "").lower()
        if role in ARIA_LANDMARK_ROLES:
            landmarks.append({
                "tag": el.name,
                "role": role,
                "aria_label": clean(el.get("aria-label", "")) or None,
            })
    payload["landmarks"] = landmarks

    # ── TABLES ────────────────────────────────────────────────────────────────
    tables = []
    for table in soup.find_all("table"):
        caption = clean(table.caption.get_text()) if table.caption else ""
        headers = [clean(th.get_text()) for th in table.find_all("th")]
        tables.append({"caption": caption, "headers": headers[:20]})
    payload["tables"] = tables

    # ── IFRAMES ───────────────────────────────────────────────────────────────
    payload["iframes"] = [
        {
            "title": clean(i.get("title", "")) or None,
            "src": i.get("src", "")[:80],
            "nearest_heading": get_nearest_heading(i),
            "landmark_region": get_landmark_region(i),
        }
        for i in soup.find_all("iframe")
    ]

    return payload


if __name__ == "__main__":
    import sys
    result = extract(sys.argv[1])
    output = json.dumps(result, indent=2)
    print(output)
    print(f"\n# Estimated tokens: {estimate_tokens(output):,}", file=__import__("sys").stderr)
