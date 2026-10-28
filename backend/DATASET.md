# GemLens training data

The current website still uses the heuristic engine. The updated regression baseline is not a detector and is not production-ready. Full-art and silver-border images are accepted by the manifest, but support is not proven without training and evaluation.

## One JSONL record per image

Required fields: card_id (physical card, shared across views), image (local relative path), source, rights_reference, training_allowed (true only with documented basis), input_reviewed (no visible grade labels or report overlays), grade_system, side (front/back), design (standard/full_art/silver_border), capture (phone/scan/specialized), annotation_coverage (unknown/partial/complete), labels, defects.

Labels: grade, centering, edges, corners, surface. Each is 1–10 or null/absent. At least one label is required. Do not copy an overall grade into missing subgrades. Preserve source 1000-point scores in separate metadata; do not divide them by 100 to invent equivalent subgrades. All rows must use one grading system.

Defects are optional objects with type and bbox [left,top,right,bottom], normalized to the EXIF-oriented image. An empty defect list with partial or unknown coverage is NOT a clean card. Current regression training stores but does not learn these defect boxes. Dataset validation checks supplied claims, not legal entitlement or expert accuracy.

## Audit and split

```sh
python -m backend.dataset --manifest dataset/manifest.jsonl --root dataset --output dataset/splits
```

This validates image files, rejects identical decoded images assigned different physical IDs, and creates deterministic physical-card splits plus coverage counts in audit.json. Exact-duplicate checks do not detect alternate crops of the same card; manual identity checks remain necessary. Small splits are workflow checks, not proof of sufficient coverage. Review design, grade and capture distributions before training.

## Baseline training

Install torch and torchvision for your platform separately. No weights download implicitly.

```sh
python -m backend.train --root dataset --train dataset/splits/train.jsonl --val dataset/splits/val.jsonl --output model.pt
```

The five-output ResNet18 uses a masked loss: missing labels contribute no loss or evaluation error. Without --weights it starts randomly. --weights accepts a local unmodified torchvision ResNet18 state dictionary for initialization, not a saved GemLens training checkpoint. The script does not implement resume. Checkpoints record unmeasured metrics as null and production_ready as false.

The model still downsamples individual images; this is an overall-grade baseline, not the planned multi-view high-resolution detector. Never expose untrained output heads as condition scores. Reserve test.jsonl until final evaluation. Next stages: paired phone/scanner evaluation data, high-resolution localization with partial-annotation handling, multi-view fusion, per-design metrics, uncertainty calibration and a production inference adapter.

## Larger paired-image experiment

`backend.train_pairs` trains an overall-grade head on the concatenated front/back
features of a frozen ImageNet ResNet18. It requires one front and one back per
physical ID, consistent grades, and disjoint train/validation/test identities.
Feature normalization uses the training split only. Checkpoint selection uses
validation MAE; the test split is evaluated once after selection. Unknown design
is allowed in manifests and must not be reported as verified full-art/silver
coverage. This experiment does not train subgrades or issue localization.

```sh
python -m backend.train_pairs --root work/hf-expanded --train work/hf-expanded/train.jsonl --val work/hf-expanded/val.jsonl --test work/hf-expanded/test.jsonl --weights work/hf-audit/resnet18.pth --output outputs/gemlens-expanded.pt --epochs 50
```

Results are written beside the checkpoint as JSON, including predictions,
per-grade errors, a training-median baseline, validation history, and limitations.
The expanded sample is a bounded subset from each shard, not a random sample of
the complete dataset. Conservative red-line filtering can also reject natural
red artwork and disproportionately affect particular designs. Exclusions need
review before scaling. Tests without low-grade examples cannot establish
performance on severe damage. Keep future phone and unseen-design evaluations
separate. A checkpoint with `production_ready: false` must not replace the live
heuristic engine automatically.

## Detail-crop comparison

Pass `--detail-patches` to add eight original-resolution squares per side:
four corners and four edge midpoints. Crop width is one quarter of the shorter
image dimension; crops resize to 224×224 before encoding. The grade head receives
each side's whole-card features plus mean/max pooled detail features (3072 total
inputs for the pair). This requires an already tightly cropped card image. It is
not a detector, and pooled patch features do not supply defect locations.

`--validation-only` skips loading/evaluating the test manifest, useful for
iteration without repeatedly inspecting final test results. The `--test` argument
is still required by the CLI but its file is ignored in this mode.

The low-grade experiment keeps old training/validation roles and excludes the
old test cards. It adds 43 lower-grade training cards and 12 validation cards;
a fresh 12-card low-grade-only set compares the two prespecified architectures.
It is a diagnostic set, not a representative production benchmark. Validation
chooses each model's checkpoint. Do not tune repeatedly to this diagnostic set.
