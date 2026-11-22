"""Build a self-contained offline geometry tutoring demo."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "src"
    data = ROOT / "data" / "visual_tutor"
    payload = {
        "challenge": json.loads((data / "challenge.json").read_text(encoding="utf-8")),
        "geodial": json.loads((data / "geodial_results.json").read_text(encoding="utf-8")),
    }
    replacements = {
        "__PROJECT_DATA__": json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c"),
        "__CORE_SCRIPT__": (source / "visual_tutor_core.js").read_text(encoding="utf-8"),
        "__APP_SCRIPT__": (source / "visual_tutor_app.js").read_text(encoding="utf-8"),
    }
    page = (source / "visual_tutor_template.html").read_text(encoding="utf-8")
    for marker, content in replacements.items():
        if page.count(marker) != 1:
            raise ValueError(f"Expected one {marker} in template.")
        page = page.replace(marker, content)
    (ROOT / "START_HERE.html").write_text(page, encoding="utf-8")
    print("Built START_HERE.html")


if __name__ == "__main__":
    main()
