"""
Curate TRACER threat reports into structured public/gt JSON pairs
for the AAM cybersecurity dataset extension.

Reads PDF paths + labels from cti_bm25_dataset_cybercriminal.csv,
runs the TRACER feature extractor on each PDF, and saves outputs to
data/cyber_extracted/{G-code}/.

Uses a local cache of enterprise-attack.json to avoid 234 network calls.
"""

import csv
import json
import sys
import types
import unittest.mock
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
TRACER_DIR   = Path("/Users/susan/Documents/PhD/HCI - Uncertainty/Project/TRACER")
CSV_PATH     = TRACER_DIR / "cti_datasets" / "cti_bm25_dataset_cybercriminal.csv"
EXTRACTOR    = TRACER_DIR / "LLM and Deterministic Model" / "threat_report_extractor.py"
MITRE_CACHE  = Path(__file__).parent / "data" / "enterprise-attack.json"
OUTPUT_DIR   = Path(__file__).parent / "data" / "cyber_extracted"

TARGET_ACTORS = {"G0007", "G0032", "G0010", "G0034", "G0016", "G0045", "G0046", "G0049"}

# ── load MITRE cache once ────────────────────────────────────────────────────
print(f"Loading MITRE cache from {MITRE_CACHE}...")
with open(MITRE_CACHE) as f:
    _MITRE_DATA = f.read()

class _FakeResponse:
    def __init__(self, text):
        self._text = text
        self.status_code = 200
    def raise_for_status(self):
        pass
    def json(self):
        return json.loads(self._text)

MITRE_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"

def _patched_get(url, **kwargs):
    if MITRE_URL in url:
        return _FakeResponse(_MITRE_DATA)
    import requests as _real_requests
    return _real_requests.Session().get(url, **kwargs)

# ── load extractor module ────────────────────────────────────────────────────
import importlib.util, requests as _requests_mod

spec = importlib.util.spec_from_file_location("threat_report_extractor", EXTRACTOR)
extractor = importlib.util.module_from_spec(spec)
sys.modules["threat_report_extractor"] = extractor

with unittest.mock.patch.object(_requests_mod, "get", _patched_get):
    spec.loader.exec_module(extractor)

import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    sys.exit("spaCy model missing. Run: python -m spacy download en_core_web_sm")

# ── read CSV ─────────────────────────────────────────────────────────────────
csv.field_size_limit(10**7)
rows = []
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        if r["label"] in TARGET_ACTORS:
            rows.append(r)

print(f"Found {len(rows)} PDFs across {len(TARGET_ACTORS)} actors.\n")

# ── process ──────────────────────────────────────────────────────────────────
ok = skipped = errors = 0

for i, row in enumerate(rows, 1):
    pdf_path = Path(row["url"])
    label    = row["label"]
    doc_id   = row["doc_id"]

    out_dir = OUTPUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    pub_path = out_dir / f"{doc_id}_public.json"
    gt_path  = out_dir / f"{doc_id}_gt.json"

    if pub_path.exists() and gt_path.exists():
        skipped += 1
        continue

    print(f"[{i}/{len(rows)}] {label} | {pdf_path.name[:60]}")

    try:
        with unittest.mock.patch.object(_requests_mod, "get", _patched_get):
            pub, gt = extractor.process(
                pdf_path, nlp,
                allow_actor_fallback=False,
                allow_malware_fallback=False,
                fetch_map=True,
                use_ocr=False,
                loose_domains=False,
            )
        # Attach source metadata
        pub["_meta"] = {"label": label, "doc_id": doc_id, "source_pdf": str(pdf_path)}
        gt["_meta"]  = {"label": label, "doc_id": doc_id, "source_pdf": str(pdf_path)}

        with open(pub_path, "w") as f:
            json.dump(pub, f, indent=2, ensure_ascii=False)
        with open(gt_path, "w") as f:
            json.dump(gt, f, indent=2, ensure_ascii=False)
        ok += 1
    except Exception as e:
        print(f"  [!] ERROR: {e}")
        errors += 1

print(f"\nDone. ok={ok}  skipped={skipped}  errors={errors}")
print(f"Output: {OUTPUT_DIR}")
