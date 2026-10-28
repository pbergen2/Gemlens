# GemLens

A functioning Pokémon card screening prototype with a Python analysis engine, a FastAPI backend, and a responsive scan-and-report website.

## Run the Python backend

Use Python 3.12 (recommended for the pinned PyTorch dependencies):

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-training.txt
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. The website automatically uses the server when `/api/health` responds. Photos are processed in memory and not stored. Interactive API documentation is at `/docs`. `/api/analyze` accepts the raw bytes of a JPG, PNG or WebP with `side=front|back`, optional normalized `crop=[left,top,right,bottom]`, and optional `original_height`. `/api/summarize` accepts up to two engine reports.

The private hosted static site uses Pyodide 0.27.7 in a Web Worker to execute the exact same `dist/engine.py` on-device. It loads the Python runtime from jsDelivr on first inspection. It does **not** host a FastAPI or PyTorch server. Deploy the supplied backend to a Python-capable host for server processing; serving the UI from FastAPI requires no API URL or CORS configuration.

## Current behavior

- Front and back photos required for trained grading; mobile photo picker and drag-and-drop.
- Straight-on rectangular card boundary detection with adjustable bounds.
- Opposing border measurements when transitions are sufficiently consistent.
- Possible whitening regions on colored borders, separated into edges/corners.
- Blur, exposure and resolution warnings; surface explicitly unscored.
- A learned decimal grade estimate from the balanced PyTorch model, with separate heuristic inspection markers.
- Downloadable JSON report, including measurements and limitations.

No external grading standard is claimed. The Python server uses an **experimental trained PyTorch overall-grade model**. Browser-only mode uses an uncalibrated heuristic. Silver borders, artwork, glare, full-art cards, rotation, perspective, and sleeves can defeat the algorithms. It does not identify card authenticity or distinguish all scratches/dents from artwork. Low-resolution analysis can miss small wear. The app does not claim that any score predicts PSA, BGS, or CGC grades. A photo of another rectangular object can pass the boundary check; automatic photo identification is not implemented; an optional TCGdex picker lets users select a reference manually.

The camera workflow uses the device's native file/photo picker; a continuous live-camera detector is not included. Place the card upright, with all corners visible, on a dark uniform background. Front/back detail and angled light remain necessary for manual surface review.

## PyTorch training status

PyTorch has been used to train experimental models. `backend/train_pairs.py` trains a
front/back grade regressor on frozen pretrained ResNet18 features, optionally including
corner and edge detail crops. `backend/train.py` provides the earlier per-image baseline.

The larger run saved `outputs/gemlens-5k.pt`: 5,119 physical front/back card pairs,
500 validation cards, and 500 separately reserved test cards. It uses whole-card
features and selected epoch 13 of 50 using validation error. Validation MAE was
0.789 grade points, with 71.2% within one point. These are same-source scans with
automatically screened images and dataset-reported grades, not independently
verified labels or a phone-photo benchmark. Frozen evaluation on the 500 reserved
test cards produced MAE 0.886, 68.6% within one point, and 17.6% overgraded by more
than one point. The proposed accuracy targets remain unmet. Full results are in
`outputs/gemlens-5k-test.json` and `outputs/training-5k-report.md`.
`backend/evaluate_pairs.py` evaluates a saved checkpoint without training, using
its saved normalization and checking physical-card and exact-image split leakage.

To reproduce the frozen checkpoint evaluation with the local dataset present:

```sh
python -m backend.evaluate_pairs --root work/hf-5k \
  --train work/hf-5k/train.jsonl --val work/hf-5k/val.jsonl \
  --test work/hf-5k/test.jsonl --checkpoint outputs/gemlens-5k.pt \
  --output outputs/gemlens-5k-test.json --allow-automated-screening
```

Keep this test set out of further model selection; a new independent holdout is
needed if its results guide subsequent development.

The earlier `outputs/gemlens-detail.pt` is the detail-crop checkpoint; its metrics
are in `outputs/gemlens-detail.json` and `outputs/lowgrade-detail-report.md`.
It trained on 282 physical cards, with 56 validation cards and 12 low-grade diagnostic
test cards. Diagnostic average absolute error was 1.47 grade points; 5 of 12 cards were
overgraded by more than one point. This small test does not establish production accuracy.

**The Python website now uses `outputs/gemlens-balanced-ranges.pt`.** Upload both original front/back photos; the server validates and crops the card boundaries and runs the paired detail model. It returns a decimal estimate, not a calibrated confidence interval. `dist/engine.py` still supplies separate rule-based criterion checks and suspected whitening markers; these do not explain the learned grade. The checkpoint remains `production_ready=false` and has no trained defect-localization output. See `backend/DATASET.md` for data requirements.
Training data and the Python environment are not included in the project ZIP.

## Verification

```sh
pip install -r backend/requirements-dev.txt
python -m unittest discover -s tests -v
```

Tests cover centered and off-center synthetic fixtures, added whitening, ambiguous neutral borders, invalid captures, malformed input, and grade abstention. They verify algorithm behavior, not real-world grading accuracy. Real card validation and expert calibration remain necessary.

## Deployment notes

For a public Python deployment, use an HTTPS reverse proxy with a 15 MB request body limit and rate limiting, bounded worker concurrency, and request timeouts. This repository is a prototype, not a production abuse-resistant service. No API keys are required. Dependencies use bounded compatible ranges; pin a lockfile in a deployment environment after validation.

Implementation references: [Pyodide usage](https://pyodide.org/en/stable/usage/index.html), [FastAPI documentation](https://fastapi.tiangolo.com/), [PyTorch transfer-learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

## TCGdex references

Run the Python server above, then open **Find your card · optional** beneath the upload area.
Search by name and language, browse results, and select the matching printing. The selected
reference includes its set, number, and catalog image, and is included in downloaded reports.
Replacing/removing the front photo clears the reference. No API key or additional dependency is needed.

- `GET /api/cards?q=Alakazam&language=en&page=1`: 20 results per page.
- `GET /api/cards/base1-1?language=en`: one reference card.
- Requests have a 10-second upstream timeout, a 2 MB response limit, and a bounded one-hour memory cache.
- Errors leave photo inspection available. Static/Pyodide-only hosting has no catalog backend;
  run/deploy FastAPI to enable lookup.
- Search terms reach TCGdex. Reference images load from its asset service; uploaded photos are not sent there.
- References are **not grade labels**, and selecting one does not change grading. This integration does
  not yet perform automatic recognition, image alignment, or reference-based defect detection.
  No catalog images are added to training. Pocket digital cards are excluded.

API documentation: https://tcgdex.dev/rest/cards and https://tcgdex.dev/assets

## Windows quick start

Extract the ZIP, open a terminal in the GemLens folder, and run:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend/requirements-training.txt
.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 in your browser. Python 3.12 is recommended.
Install `backend/requirements-training.txt` to run trained grading, and keep `outputs/gemlens-balanced-ranges.pt` alongside the project. Browser-only inspection does not run this model. Models and training reports are included separately
in `outputs/`; training requires supplying the datasets and split manifests described in the docs.

## Trained grading endpoint

`POST /api/grade?front_bytes=N` accepts the original front image bytes followed by the original back image bytes. Each is limited to 15 MB and 24 megapixels. Optional `front_crop` and `back_crop` are JSON normalized rectangles. Boundaries and capture quality are checked server-side; card crops must be at least 500 pixels on the shorter side. Inference is serialized and lazily loaded, with uploads kept in memory only. Missing model/dependencies return 503 rather than silently substituting a heuristic grade. The model has not been independently evaluated on everyday phone photos.

## Downloading from GitHub

Clone this repository or choose **Code → Download ZIP**, then extract it and follow the setup instructions above (or Windows quick start). Run commands from the project folder. The first dependency install needs internet access; PyTorch packages can be large. No API key or training dataset is needed for grading. TCGdex lookup needs internet access.

The repository must include `outputs/gemlens-balanced-ranges.pt` (about 44 MB). The `.gitignore` explicitly allows this deployed checkpoint while excluding other experiment outputs, the local Python environment, and `work/` datasets. A clone with this checkpoint and installed training requirements can run the existing model without retraining. Check `/api/health` for `trained_model_available: true`; a real grading request confirms successful model loading.

GitHub stores the project; it does not run this Python backend. Each user runs the server locally and opens http://127.0.0.1:8000. Sharing that localhost address does not share the app with another computer. Public hosting is a separate deployment step.

This is an experimental release with source-reported scan labels, not a certified grading service. Before a public release, choose a code license and document the redistribution permissions for bundled artwork and model weights. Dataset-training permission is recorded separately and is not a blanket license for every bundled asset. No license covering third-party artwork is asserted here.
