# Written using the assist of ChatGPT

import sys
import os
from bs4 import BeautifulSoup


# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def css_path(el):
    path = []
    while el and el.name:
        sibling_index = 1
        sibling = el
        while sibling.previous_sibling:
            sibling = sibling.previous_sibling
            if getattr(sibling, "name", None) == el.name:
                sibling_index += 1

        if sibling_index > 1:
            path.append(f"{el.name}:nth-of-type({sibling_index})")
        else:
            path.append(el.name)

        el = el.parent

    return " > ".join(reversed(path))


def element_location(element):
    if element is None:
        return None

    attrs = dict(element.attrs)

    loc = {
        "tag": element.name,
        "id": attrs.get("id"),
        "class": attrs.get("class"),
        "css_path": css_path(element),
        "attributes": attrs,
        "text_preview": element.get_text(strip=True)[:80]
    }

    # Include src for images/iframes so findings can be located
    if element.name in ("img", "iframe", "input"):
        loc["src"] = element.get("src", "")[:120]

    return loc


def issue(rule_id, rule_name, element, description):
    return {
        "rule_id": rule_id,
        "rule_name": rule_name,
        "location": element_location(element),
        "description": description
    }


# ==========================================================
# MAIN AUDIT FUNCTION
# ==========================================================

def audit_nontext(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    results = []

    # ==========================================================
    # Image Alt Text
    # ==========================================================

    # Images that convey content MUST have programmatically-discernible alternative text.
    for img in soup.find_all("img"):

        # NON_TEXT_001 — Missing alt attribute
        if img.get("alt") is None:
            results.append(issue(
                "NON_TEXT_001",
                "Image missing alt attribute",
                img,
                "<img> element does not contain an alt attribute."
            ))

    # All actionable images MUST have alternative text.
    for a in soup.find_all("a"):
        img = a.find("img")
        if img:
            alt = img.get("alt")
            if alt is None or alt.strip() == "":
                results.append(issue(
                    "NON_TEXT_002",
                    "Actionable image missing alt text",
                    img,
                    "Image inside a link must have non-empty alt text."
                ))

    # Form inputs with type="image" MUST have alternative text.
    for inp in soup.find_all("input", attrs={"type": "image"}):
        alt = inp.get("alt")
        if alt is None or alt.strip() == "":
            results.append(issue(
                "NON_TEXT_003",
                "Image input missing alt text",
                inp,
                "Form input type='image' must have non-empty alt text."
            ))

    # ==========================================================
    # Image Maps
    # ==========================================================

    # The alternative text for the <area> element MUST be available.
    for area in soup.find_all("area"):
        alt = area.get("alt")
        if alt is None or alt.strip() == "":
            results.append(issue(
                "NON_TEXT_004",
                "Image map area missing alt text",
                area,
                "<area> element must have non-empty alt text."
            ))

    # ==========================================================
    # SVG
    # ==========================================================

    # SVG SHOULD NOT be embedded via <object> or <iframe>.
    for obj in soup.find_all(["object", "iframe"]):
        src = obj.get("data") or obj.get("src")
        if src and src.lower().endswith(".svg"):
            results.append(issue(
                "NON_TEXT_005",
                "SVG embedded via object or iframe",
                obj,
                "SVG should not be embedded using <object> or <iframe>."
            ))

    # ==========================================================
    # HTML 5 <canvas>
    # ==========================================================

    # All <canvas> elements MUST have a text alternative.
    for canvas in soup.find_all("canvas"):
        if not canvas.get_text(strip=True):
            results.append(issue(
                "NON_TEXT_006",
                "Canvas missing fallback text",
                canvas,
                "<canvas> element must contain fallback text content."
            ))

    # ==========================================================
    # Plug-ins
    # ==========================================================

    # All <object> elements MUST have alternative text.
    for obj in soup.find_all("object"):
        if not obj.get_text(strip=True):
            results.append(issue(
                "NON_TEXT_007",
                "Object missing alternative text",
                obj,
                "<object> element must contain alternative text content."
            ))

    return results


# ==========================================================
# CLI ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python audit_nontext.py <file.html>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    results = audit_nontext(file_path)

    print("\n=== NON-TEXT ACCESSIBILITY AUDIT RESULTS ===\n")

    if not results:
        print("No issues found.")
        sys.exit(0)

    for r in results:
        print(f"[{r['rule_id']}] {r['rule_name']}")

        location = r.get("location")

        if location:
            if location.get("css_path"):
                print(f"  Location: {location['css_path']}")
            if location.get("id"):
                print(f"  ID: {location['id']}")
            if location.get("attributes"):
                print(f"  Attributes: {location['attributes']}")
            if location.get("text_preview"):
                print(f"  Text Preview: \"{location['text_preview']}\"")

        print(f"  Description: {r['description']}")
        print()

    sys.exit(2)