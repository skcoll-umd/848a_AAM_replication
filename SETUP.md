# Setup & Pipeline Documentation

Full technical reference for running the AAM replication and cybersecurity extension experiments.

---

## Repository Structure

```
data/
  prepared.csv              Full replication set (200 rows — 50 SIQA + 50 CQA, gold + distractor)
  prepared_test.csv         10-question test subset (20 rows, for fast iteration)
  cyber_pairs.csv           Cybersecurity extension — 219 APT reports × 2 choices (438 rows)
  cyber_extracted/          Structured JSON per report (public + gt), organized by G-code
pipeline/                   Ollama client, rationale generation, plausibility rating
prompts/                    Exact prompt templates from Palta et al. Appendix A
analysis/                   Analysis script (Tables 2/3/4, violin plots)
results/                    Output CSVs and figures (gitignored)
prepare_data.py             Reshapes AAM source data into pipeline-ready format
curate_cyber_dataset.py     Runs TRACER extractor on 234 PDFs → data/cyber_extracted/
build_cyber_pairs.py        Builds cyber_pairs.csv from extracted JSON files
run_pipeline.py             Pipeline entrypoint
```

---

## Requirements

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Requires [Ollama](https://ollama.ai) running locally at `http://localhost:11434`. Pull models with:

```bash
ollama pull qwen3.6:35b
ollama pull gemma4:31b
ollama pull deepseek-r1:32b
```

---

## Running Experiments

### Quick test (fast, ~5 min)

```bash
python run_pipeline.py --model qwen3:8b --input data/prepared_test.csv --skip-rationale-gen
```

### Part 1: AAM Replication

```bash
python run_pipeline.py --model qwen3.6:35b gemma4:31b deepseek-r1:32b \
    --input data/prepared.csv --skip-rationale-gen
```

Uses pre-generated GPT-4o rationales from the paper. `--skip-rationale-gen` skips Step 1.

### Part 2: Cybersecurity Extension

```bash
python run_pipeline.py --model qwen3.6:35b gemma4:31b deepseek-r1:32b \
    --input data/cyber_pairs.csv
```

Rationale generation is required here (no pre-generated rationales).

### Analysis

```bash
python analysis/analyze.py
```

Produces Tables 2/3/4 equivalents and violin plots. Output written to `results/figures/`.

### List available models

```bash
python run_pipeline.py --list-models
```

---

## Pipeline Steps

1. **Rationale generation** (temp=1.0) — for each `(question, choice)` pair, generate a PRO rationale (arguing for plausibility) and a CON rationale (arguing against). Prompts A.1–A.2 from paper Appendix A.5.

2. **Plausibility rating** (temp=0.1) — rate each pair under 4 conditions: NO rationale, PRO only, CON only, PRO+CON. 1–5 Likert scale (Impossible → Very Likely). Prompts A.3–A.6. `max_tokens=50` to suppress verbose output.

3. **Analysis** — Δ from NO baseline, chi-squared tests of homogeneity on rating distributions, OLS regression (Δ ~ rationale_type + NO_baseline_rating).

Output schema: `question, choice, condition, rating, rating_raw, model, answer_label, dataset`

---

## Experiment Scale

| Dataset | Questions | Choices/question | Rows | Conditions | Ratings/model |
|---------|-----------|-----------------|------|------------|---------------|
| SIQA    | 50        | 2 (gold + distractor) | 100  | 4          | 400           |
| CQA     | 50        | 2 (gold + distractor) | 100  | 4          | 400           |
| Cyber   | 219       | 2 (gold + distractor) | 438  | 4          | 1,752         |
| **Total** | **319** | | **638** | | **2,552/model** |

Three models × 2,552 ratings = **7,656 total ratings**. Part 2 also requires rationale generation (PRO + CON per row = 876 rationale calls per model).

---

## Part 1 Dataset

**Source:** AAM paper repository (https://github.com/shramay-palta/AAM). Run `prepare_data.py` to regenerate from source:

```bash
python prepare_data.py --aam-dir "/path/to/AAM" --output data/prepared.csv
```

**Key findings to replicate:**
- PRO rationales raise distractor ratings but *drop* gold label ratings
- CON rationales drop both, but drop gold more sharply (>1 Likert point avg)
- PRO+CON ≈ average of PRO and CON effects
- Higher initial NO rating → smaller subsequent change (anchoring)

---

## Part 2 Dataset: Cybersecurity Extension

### Design rationale

Each question is a de-identified threat report summary. The rater is asked: *"How plausible is it that **[actor]** conducted this campaign?"* The actor is either the true attribution (gold) or a confusable alternative (distractor).

Distractors are chosen by **nation-state nexus** where possible — real attribution ambiguity occurs within nexus groups (e.g., distinguishing APT28 from Turla is a genuine analytic challenge). For actors with no same-nexus partner in the dataset, the highest TTP-Jaccard wrong actor is used.

### Actors

| G-code | Actor | Reports | Nexus |
|--------|-------|---------|-------|
| G0007  | APT28 | 56 | Russia |
| G0010  | Turla | 36 | Russia |
| G0016  | APT29 | 17 | Russia |
| G0032  | Lazarus Group | 44 | North Korea |
| G0034  | Sandworm | 22 | Russia |
| G0045  | menuPass | 21 | China |
| G0046  | FIN7 | 7 | Criminal |
| G0049  | APT34 | 16 | Iran |

### Question text format

Each question is a structured block assembled from the extracted JSON fields:

```
<narrative prose (actor/malware names redacted)>
ATT&CK techniques observed: T1012, T1016, ... Tactic categories: Collection, Discovery, ...
Targeted sectors: Defense, Energy, Government.
Delivery method(s): email, phishing.
Tools/frameworks observed: powershell.
```

Fields are only included if present — not every report has delivery vectors or observed tools. The narrative is always required (quality filter enforces this). ATT&CK technique IDs give the rater behavioral signal independent of actor-linked terminology; tactic categories give higher-level framing; sectors and delivery vectors add context a human analyst would use for attribution.

**Scrubbing applied:**
- Hashes, IPs, hex values → `[HASH]`, `[IP]`, `[HEX]`
- Defanged domains / URLs → `[DOMAIN]` / `[URL]`
- Actor-linked malware names (e.g. Zebrocy→APT28, BlackEnergy→Sandworm) → `[REDACTED_MALWARE]`
- Quality filter: α-character ratio ≥ 55%, ≤ 3 remaining IOC tokens, narrative ≥ 80 chars

### Rebuilding from source

The `data/cyber_extracted/` directory is pre-built. To regenerate (requires TRACER repo):

```bash
# Extract structured JSON from PDFs
python curate_cyber_dataset.py

# Build question-choice pairs
python build_cyber_pairs.py
```

Requires TRACER at `/Users/susan/Documents/PhD/HCI - Uncertainty/Project/TRACER` and `pdfplumber`, `spacy`, `en_core_web_sm`.

---

## Models

| Model | Size | Notes |
|-------|------|-------|
| `qwen3.6:35b` | 23 GB | Primary Qwen |
| `gemma4:31b` | 19 GB | Secondary |
| `deepseek-r1:32b` | 19 GB | Reasoning model — tests whether CoT resists rationale sway |

Temperature: rationale generation = 1.0 · plausibility rating = 0.1 (per paper Appendix A.3 for non-OpenAI models)
