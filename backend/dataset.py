"""Validate local, permissioned image manifests before any model training.

JSONL rows describe one image, not an inferred annotation. No network downloads.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps

TARGETS = ["grade", "centering", "edges", "corners", "surface"]

def validate(rows, root, *, allow_automated_screening=False):
    root = Path(root).resolve()
    if not rows:
        raise ValueError("Dataset is empty")
    hashes, grades, domains = {}, {}, set()
    for i, row in enumerate(rows, 1):
        def require(ok, message):
            if not ok:
                raise ValueError(f"Row {i}: {message}")
        for field in ("card_id", "image", "source", "rights_reference", "grade_system"):
            require(isinstance(row.get(field), str) and bool(row[field].strip()), f"missing {field}")
        require(row.get("training_allowed") is True, "training permission not recorded")
        require(row.get("side") in ("front", "back"), "side must be front or back")
        require(row.get("design") in ("standard", "full_art", "silver_border", "unknown"), "invalid design")
        require(row.get("capture") in ("phone", "scan", "specialized"), "unknown capture type")
        require(row.get("annotation_coverage") in ("unknown", "partial", "complete"), "missing annotation coverage")
        reviewed = row.get("input_reviewed") is True
        screened = (allow_automated_screening and row.get("screening", {}).get("status") == "passed"
                    and row.get("screening", {}).get("method") == "5k-v1-automated")
        require(reviewed or screened, "confirm image contains no grade label or report overlays; automated screening requires explicit experimental opt-in")
        domains.add(row["grade_system"])
        labels = row.get("labels", {})
        require(isinstance(labels, dict), "labels must be an object")
        require(any(labels.get(k) is not None for k in TARGETS), "no numeric training target")
        for key, value in labels.items():
            require(key in TARGETS, f"unsupported target {key}; keep raw scores in source metadata")
            require(value is None or (type(value) in (int, float) and math.isfinite(value) and 1 <= value <= 10), f"{key} must be null or a 1–10 score")
        grade = labels.get("grade")
        if grade is not None:
            require(row["card_id"] not in grades or grades[row["card_id"]] == grade, "conflicting grades for the same physical card")
            grades[row["card_id"]] = grade
        path = (root / row["image"]).resolve()
        require(path.is_relative_to(root), "image escapes dataset root")
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            require(min(im.size) >= 150, "image too small")
            digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        require(digest not in hashes or hashes[digest] == row["card_id"], "duplicate image assigned to different physical cards")
        hashes[digest] = row["card_id"]
        for defect in row.get("defects", []):
            box = defect.get("bbox", [])
            require(isinstance(defect.get("type"), str) and bool(defect["type"]), "defect needs type")
            require(len(box) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in box), "invalid defect box")
            require(0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1, "defect box must be normalized xyxy")
    if len(domains) != 1:
        raise ValueError("Do not mix grading systems in one training dataset")
    return rows

def split_rows(rows, seed=42):
    ids = sorted({r["card_id"] for r in rows}, key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
    if len(ids) < 3:
        raise ValueError("Need at least three physical cards to create disjoint splits; this is not a training sufficiency threshold")
    n = max(1, len(ids)//10)
    assignment = {card: "test" if i < n else "val" if i < 2*n else "train" for i, card in enumerate(ids)}
    return {split: [r for r in rows if assignment[r["card_id"]] == split] for split in ("train", "val", "test")}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = validate([json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()], args.root)
    splits = split_rows(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    audit = {}
    for name, items in splits.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in items))
        audit[name] = {"images": len(items), "physical_cards": len({r["card_id"] for r in items}),
                       "designs": dict(Counter(r["design"] for r in items)),
                       "captures": dict(Counter(r["capture"] for r in items)),
                       "labels": {k: sum(r["labels"].get(k) is not None for r in items) for k in TARGETS}}
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
