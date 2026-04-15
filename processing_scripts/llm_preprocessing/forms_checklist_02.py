import re
import json
from bs4 import BeautifulSoup

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


def resolve_ids(soup, id_string):
    """Resolve a space-separated list of IDs to their combined text content."""
    if not id_string:
        return None
    texts = [
        clean(el.get_text())
        for ref_id in id_string.split()
        if (el := soup.find(id=ref_id.strip()))
    ]
    return " ".join(texts) if texts else None


def label_source(label_el, wrapping_label, aria_label, aria_labelledby_text,
                 title, placeholder):
    """Return the primary source of the accessible name for a field."""
    if label_el:
        return "label_for"
    if wrapping_label:
        return "wrapping_label"
    if aria_labelledby_text:
        return "aria_labelledby"
    if aria_label:
        return "aria_label"
    if title:
        return "title"
    if placeholder:
        return "placeholder_only"   # disappears on input — not a real label
    return "none"


def extract(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    payload = {}

    # ── FORMS ─────────────────────────────────────────────────────────────────
    forms = []

    for form in soup.find_all("form"):
        form_data = {
            "action": form.get("action", "") or None,
            "aria_label": clean(form.get("aria-label", "")) or None,
            "aria_labelledby_text": resolve_ids(soup, form.get("aria-labelledby", "")),
            "fields": [],
            "groups": [],
        }

        # ── Fields ────────────────────────────────────────────────────────────
        for inp in form.find_all(["input", "select", "textarea"]):
            inp_type = inp.get("type", "text").lower()
            if inp_type in SKIP_INPUT_TYPES:
                continue

            inp_id = inp.get("id", "")

            # Label association
            label_el = soup.find("label", attrs={"for": inp_id}) if inp_id else None
            wrapping = inp.find_parent("label") if not label_el else None
            label_text = clean((label_el or wrapping).get_text()) if (label_el or wrapping) else None

            aria_label        = clean(inp.get("aria-label", "")) or None
            aria_labelledby   = inp.get("aria-labelledby", "")
            aria_lby_text     = resolve_ids(soup, aria_labelledby)
            title             = clean(inp.get("title", "")) or None
            placeholder       = clean(inp.get("placeholder", "")) or None

            # Instructions (aria-describedby)
            instructions = resolve_ids(soup, inp.get("aria-describedby", ""))

            # Fieldset group context
            fieldset   = inp.find_parent("fieldset")
            group_label = None
            if fieldset:
                legend = fieldset.find("legend")
                group_label = clean(legend.get_text()) if legend else None

            # Effective label (what the screen reader will announce)
            effective = label_text or aria_lby_text or aria_label or title

            src = label_source(label_el, wrapping, aria_label, aria_lby_text,
                               title, placeholder)

            form_data["fields"].append({
                "type":                   inp_type,
                "id":                     inp_id or None,
                "name":                   inp.get("name") or None,
                "label":                  label_text,
                "aria_label":             aria_label,
                "aria_labelledby_text":   aria_lby_text,
                "title":                  title,
                "effective_label":        effective,
                "label_source":           src,
                "placeholder":            placeholder,
                "instructions":           instructions,
                "required":               inp.has_attr("required") or
                                          inp.get("aria-required") == "true",
                "group_label":            group_label,
                "nearest_heading":        get_nearest_heading(inp),
                "landmark_region":        get_landmark_region(inp),
            })

        # ── Fieldset / legend groups ───────────────────────────────────────────
        for fieldset in form.find_all("fieldset"):
            legend    = fieldset.find("legend")
            leg_text  = clean(legend.get_text()) if legend else None
            inp_types = [
                inp.get("type", "text")
                for inp in fieldset.find_all(["input", "select", "textarea"])
                if inp.get("type", "text").lower() not in SKIP_INPUT_TYPES
            ]
            form_data["groups"].append({
                "legend":      leg_text,
                "input_types": inp_types,
            })

        if form_data["fields"] or form_data["groups"]:
            forms.append(form_data)

    payload["forms"] = forms

    # ── STANDALONE LABELS (outside any <form>) ────────────────────────────────
    # Catches labels that reference controls outside a form element
    all_form_inputs = {
        inp.get("id")
        for form in soup.find_all("form")
        for inp in form.find_all(["input", "select", "textarea"])
        if inp.get("id")
    }
    orphan_labels = []
    for label in soup.find_all("label"):
        for_id = label.get("for")
        if for_id and for_id not in all_form_inputs:
            target = soup.find(id=for_id)
            orphan_labels.append({
                "label_text": clean(label.get_text()),
                "for":        for_id,
                "target_tag": target.name if target else None,
            })
    payload["orphan_labels"] = orphan_labels

    return payload


if __name__ == "__main__":
    import sys
    print(json.dumps(extract(sys.argv[1]), indent=2))
