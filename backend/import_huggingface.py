"""Stage the jyesr metadata for review; never infer training permission or subgrades.

Run with --metadata path/to/metadata.csv --output dataset/review.
Images must be obtained separately after resolving source licensing. Outputs are
review candidates, deliberately not accepted by backend.dataset until reviewed.
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = "https://huggingface.co/datasets/jyesr/pokemon-tcg-grading"
REVISION = "02e6b335b59b67e2d2df678ce8fcff4b1a0b1d8a"


def parse_grade(text):
    """Accept unqualified numeric labels only; retain source text elsewhere."""
    match = re.fullmatch(r"[A-Z][A-Z +\-]*\s+(10|[1-9](?:\.5)?)", text.strip())
    return float(match[1]) if match else None


def stage(rows):
    candidates, rejected, seen = [], [], set()
    groups = defaultdict(list)
    for row in rows:
        filename, cert = row.get("filename", ""), row.get("cert_id", "")
        match = re.fullmatch(r"([0-9]+)_(front|back)\.jpg", filename)
        if not match or match[1] != cert or filename in seen:
            raise ValueError(f"Invalid, mismatched, or duplicate image identity: {filename!r}")
        seen.add(filename)
        grade = parse_grade(row.get("grade", ""))
        if grade is None:
            rejected.append({"filename": filename, "grade": row.get("grade"),
                             "reason": "missing, nonnumeric, or qualified grade"})
            continue
        groups[cert].append((row, match[2], grade))
    for cert, views in sorted(groups.items()):
        if len(views) != 2 or {x[1] for x in views} != {"front", "back"} or len({x[2] for x in views}) != 1:
            rejected.extend({"filename": r["filename"], "reason": "incomplete or inconsistent pair"} for r, _, _ in views)
            continue
        for row, side, grade in views:
            candidates.append({
                "card_id": "hf-jyesr-" + cert, "image": "images/" + row["filename"],
                "source": SOURCE, "source_revision": REVISION,
                "source_metadata": row, "training_allowed": False,
                "rights_reference": "", "input_reviewed": False,
                "grade_system": "dataset-reported-psa-unverified",
                "side": side, "design": "unknown", "capture": "scan",
                "annotation_coverage": "unknown", "labels": {"grade": grade},
                "defects": [],
                "review_needed": ["license", "grade provenance", "image availability",
                                  "card design", "grade-label and overlay removal"]})
    audit = {"source": SOURCE, "revision": REVISION,
             "metadata_records": len(rows), "candidate_images": len(candidates),
             "candidate_physical_cards": len(candidates)//2, "excluded_images": len(rejected),
             "physical_card_grade_counts": dict(Counter(str(r["labels"]["grade"]) for r in candidates if r["side"] == "front")),
             "training_ready": False,
             "limitations": ["No license declared at inspected revision",
                             "Expected versus actual grade provenance unresolved",
                             "No localized defect labels in metadata",
                             "Images have not been downloaded or visually audited"]}
    return candidates, rejected, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.metadata, newline="", encoding="utf-8-sig") as stream:
        candidates, rejected, audit = stage(list(csv.DictReader(stream)))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("candidates", candidates), ("excluded", rejected)):
        (out / (name + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
