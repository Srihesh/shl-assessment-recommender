"""Prepare the SHL catalog."""

import json
import pathlib

SRC = pathlib.Path(__file__).parent / "SHL_catalogue.json"
DST = pathlib.Path(__file__).parent / "catalog.json"

# Map categories to codes
KEY_TO_TYPE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Personality & Behavior": "P",
    "Biodata & Situational Judgment": "B",
    "Knowledge & Skills": "K",
    "Simulations": "S",
    "Assessment Exercises": "E",
    "Competencies": "C",
    "Development & 360": "D",
}

# Priority for multiple keys
TYPE_PRIORITY = ["A", "P", "B", "S", "E", "C", "D", "K"]


def pick_type(keys: list[str]) -> str:
    """Return the highest-priority test_type code for a product."""
    codes = [KEY_TO_TYPE.get(k, "K") for k in keys]
    for pref in TYPE_PRIORITY:
        if pref in codes:
            return pref
    return codes[0] if codes else "K"


def normalise(raw: dict) -> dict:
    keys = raw.get("keys", [])
    return {
        "name": raw["name"].strip(),
        "url": raw["link"].strip(),
        "test_type": pick_type(keys),
        "test_type_full": ", ".join(keys) if keys else "Knowledge & Skills",
        "description": raw.get("description", "").strip(),
        "competencies": raw.get("job_levels", []),
        "duration": raw.get("duration", ""),
        "remote_testing": raw.get("remote", "").lower() == "yes",
        "adaptive": raw.get("adaptive", "").lower() == "yes",
        "languages": raw.get("languages", []),
        "entity_id": raw.get("entity_id", ""),
    }


def main() -> None:
    print(f"Reading {SRC} …")
    with open(SRC, encoding="utf-8") as f:
        raw_items = json.load(f)

    normalised = [normalise(item) for item in raw_items]

    # Remove URL duplicates
    seen_urls: set[str] = set()
    deduped: list[dict] = []
    for item in normalised:
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            deduped.append(item)

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    print(f"✓ catalog.json written — {len(deduped)} products")

    # Print breakdown by type
    from collections import Counter
    counts = Counter(item["test_type"] for item in deduped)
    for code, n in sorted(counts.items()):
        print(f"  {code}: {n}")


if __name__ == "__main__":
    main()
