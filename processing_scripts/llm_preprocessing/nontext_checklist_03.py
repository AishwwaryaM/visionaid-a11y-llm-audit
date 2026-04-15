import re
import json
from bs4 import BeautifulSoup

# Icon font class prefixes — covers FA v4/v5/v6, Elementor, Dashicons, Glyphicons
ICON_CLASS_RE = re.compile(
    r"\b(fa|fas|far|fab|fal|fad|fa-|eicon|dashicons|glyphicon)\b", re.IGNORECASE
)

# Hints that an image is complex (chart, diagram, etc.) — needs long description
COMPLEX_HINT_RE = re.compile(
    r"(chart|graph|diagram|infographic|figure|map|plot)", re.IGNORECASE
)

# Alt text quality anti-patterns — checkable without semantic understanding
ALT_FILENAME_RE     = re.compile(r"\.(jpe?g|png|gif|svg|webp|bmp|ico)$", re.IGNORECASE)
ALT_REDUNDANT_RE    = re.compile(
    r"^(image of|photo of|picture of|graphic of|icon of|screenshot of)", re.IGNORECASE
)


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


def css_path(el, max_depth=4):
    """Short CSS-like path to help locate an element in the HTML."""
    parts = []
    current = el
    for _ in range(max_depth):
        if not current or not current.name:
            break
        name = current.name
        el_id = current.get("id")
        if el_id:
            parts.append(f"{name}#{el_id}")
            break
        el_class = current.get("class")
        if el_class:
            parts.append(f"{name}.{el_class[0]}")
        else:
            parts.append(name)
        current = current.parent
    return " > ".join(reversed(parts))


def resolve_id(soup, el_id):
    """Return text of a single element by ID."""
    if not el_id:
        return None
    target = soup.find(id=el_id.strip())
    return clean(target.get_text()) if target else None


def alt_flags(alt_text):
    """Return a list of quick programmatic quality flags on alt text."""
    flags = []
    if not alt_text:
        return flags
    if ALT_FILENAME_RE.search(alt_text):
        flags.append("looks_like_filename")
    if ALT_REDUNDANT_RE.match(alt_text):
        flags.append("redundant_phrase")
    if len(alt_text) > 150:
        flags.append("too_long")
    return flags


def extract(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    payload = {}

    # ── IMAGES ────────────────────────────────────────────────────────────────
    # Split into four categories because the LLM evaluation criteria differ for each.
    images = {
        "informative": [],   # has alt text → LLM judges quality
        "decorative":  [],   # empty alt="" → LLM verifies truly decorative
        "actionable":  [],   # inside <a> or <button> → alt must describe action/destination
        "complex":     [],   # likely chart/diagram → needs long description
    }

    for img in soup.find_all("img"):
        alt     = img.get("alt")          # None = missing; "" = empty (decorative)
        src     = img.get("src", "").split("/")[-1][:60]
        classes = " ".join(img.get("class", []))
        src_full = img.get("src", "")

        parent_link   = img.find_parent("a")
        parent_button = img.find_parent("button")
        is_actionable = bool(parent_link or parent_button)

        # Hints in filename or class name suggest a complex image
        is_complex = bool(
            COMPLEX_HINT_RE.search(src_full) or
            COMPLEX_HINT_RE.search(classes)
        )

        # Base entry — always present
        entry = {
            "src": src,
            "alt": clean(alt) if alt else None,
            "alt_flags": alt_flags(clean(alt) if alt else None),
            "selector": css_path(img),
            "nearest_heading": get_nearest_heading(img),
            "landmark_region": get_landmark_region(img),
        }

        if alt is None:
            # Missing alt entirely — not added to any category here;
            # programmatic/semantic_checklist_01.py already catches these.
            continue

        if is_actionable:
            # The accessible name must describe the link destination or button action,
            # not the visual appearance of the image.
            if parent_link:
                entry["context"]          = "in_link"
                entry["link_aria_label"]  = clean(parent_link.get("aria-label", "")) or None
                entry["link_text"]        = clean(parent_link.get_text()) or None
                entry["link_href"]        = parent_link.get("href", "")[:80] or None
            else:
                entry["context"]         = "in_button"
                entry["button_text"]     = clean(parent_button.get_text()) or None
                entry["button_aria_label"] = clean(parent_button.get("aria-label", "")) or None
            images["actionable"].append(entry)

        elif is_complex:
            # Complex images need a long description. Capture any linked description.
            describedby = img.get("aria-describedby", "")
            entry["aria_describedby_text"] = resolve_id(soup, describedby)
            entry["longdesc"]              = img.get("longdesc") or None
            images["complex"].append(entry)

        elif alt.strip() == "":
            # Empty alt — should be truly decorative; LLM judges from context
            parent_text = clean(img.find_parent().get_text())[:100] if img.find_parent() else None
            entry["surrounding_text"] = parent_text
            images["decorative"].append(entry)

        else:
            # Has non-empty alt — LLM judges quality
            images["informative"].append(entry)

    payload["images"] = images

    # ── SVG ───────────────────────────────────────────────────────────────────
    # Only non-decorative SVGs (i.e. not aria-hidden). The LLM evaluates whether
    # the title/description is sufficient.
    svgs = []
    for svg in soup.find_all("svg"):
        if svg.get("aria-hidden") == "true":
            continue  # intentionally decorative — skip

        title_el = svg.find("title")
        desc_el  = svg.find("desc")

        parent = svg.parent
        parent_text = clean(parent.get_text())[:60] if parent else None

        svgs.append({
            "selector":        css_path(svg),
            "role":            svg.get("role") or None,
            "aria_label":      clean(svg.get("aria-label", "")) or None,
            "aria_labelledby": svg.get("aria-labelledby") or None,
            "title":           clean(title_el.get_text()) if title_el else None,
            "desc":            clean(desc_el.get_text()) if desc_el else None,
            "parent_tag":      parent.name if parent else None,
            "parent_text":     parent_text,
            "nearest_heading": get_nearest_heading(svg),
            "landmark_region": get_landmark_region(svg),
        })
    payload["svgs"] = svgs

    # ── ICON FONTS ────────────────────────────────────────────────────────────
    # Icon fonts (Font Awesome, Elementor icons, Dashicons) are a common source
    # of accessibility failures. The LLM determines whether each icon conveys
    # information that requires an accessible name.
    icon_fonts = []
    seen_icons: set = set()

    for el in soup.find_all(["i", "span"]):
        classes = " ".join(el.get("class", []))
        if not ICON_CLASS_RE.search(classes):
            continue

        aria_hidden   = el.get("aria-hidden") == "true"
        aria_label    = clean(el.get("aria-label", "")) or None
        visible_text  = clean(el.get_text()) or None

        # Deduplicate by class + hidden state + label
        key = (classes[:60], aria_hidden, aria_label)
        if key in seen_icons:
            continue
        seen_icons.add(key)

        # Get sibling/parent text to help LLM decide if icon is supplementary
        parent = el.parent
        sibling_text = clean(parent.get_text())[:80] if parent else None

        # Is this icon the sole content of a link or button?
        parent_link   = el.find_parent("a")
        parent_button = el.find_parent("button")
        sole_content  = False
        if parent_link and not clean(parent_link.get_text().replace(visible_text or "", "")):
            sole_content = True
        if parent_button and not clean(parent_button.get_text().replace(visible_text or "", "")):
            sole_content = True

        icon_fonts.append({
            "classes":        classes[:80],
            "aria_hidden":    aria_hidden,
            "aria_label":     aria_label,
            "visible_text":   visible_text,
            "sibling_text":   sibling_text,
            "sole_content":   sole_content,   # True = icon is the only content of a link/button
            "nearest_heading": get_nearest_heading(el),
            "landmark_region": get_landmark_region(el),
        })

    payload["icon_fonts"] = icon_fonts

    # ── VIDEO / AUDIO ─────────────────────────────────────────────────────────
    media = []

    for video in soup.find_all("video"):
        src = (
            video.get("src") or
            (video.find("source") or {}).get("src", "")
        )
        tracks = [
            {
                "kind":  t.get("kind"),
                "label": t.get("label"),
                "srclang": t.get("srclang"),
            }
            for t in video.find_all("track")
        ]
        media.append({
            "type":         "video",
            "src":          src[:80] if src else None,
            "has_controls": video.has_attr("controls"),
            "autoplay":     video.has_attr("autoplay"),
            "tracks":       tracks,
            "aria_label":   clean(video.get("aria-label", "")) or None,
        })

    for audio in soup.find_all("audio"):
        tracks = [
            {"kind": t.get("kind"), "label": t.get("label")}
            for t in audio.find_all("track")
        ]
        media.append({
            "type":         "audio",
            "has_controls": audio.has_attr("controls"),
            "autoplay":     audio.has_attr("autoplay"),
            "tracks":       tracks,
            "aria_label":   clean(audio.get("aria-label", "")) or None,
        })

    payload["media"] = media

    return payload


if __name__ == "__main__":
    import sys
    print(json.dumps(extract(sys.argv[1]), indent=2))
