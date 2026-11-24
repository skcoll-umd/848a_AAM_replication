# AAM Replication — Arguments that Alter Minds

**Palta, Rankel, Wiegreffe & Rudinger (ACL 2026)**

**Course:** CMSC 848A

**Team:** Susan Collins and Mike Ledford

---

## Claim

Palta et al. show that LLM-generated rationales systematically shift how LLMs rate the plausibility of answer choices — PRO rationales inflate ratings, CON rationales suppress them, and the effect is asymmetric by answer type (gold vs. distractor). We test two things: (1) whether this effect replicates on open-weight models run locally, and (2) whether it extends to a more nuanced "experts" setting that uses a cybersecurity task (APT threat actor attribution)

---

## What We Did

**Replication (Part 1).** We ran the paper's LLM-rater pipeline on 50 SIQA and 50 CQA questions using three open-weight models: `qwen3.6:35b`, `gemma4:31b`, and `deepseek-r1:32b`, served locally via Ollama. Each (question, choice) pair was rated under four conditions — NO rationale, PRO only, CON only, PRO+CON — using the exact prompt templates from the paper (Appendix A.5). Rationales were taken from the paper's pre-generated GPT-4o outputs rather than re-generated. We target replication of Tables 2/3/4 and violin plots (Figures 2/3).

**Extension (Part 2).** In order to test whether LLMs can be swayed in other "expert" level tasks, we constructed a cybersecurity attribution dataset from TRACER (https://github.com/skcoll-umd/TRACER), a labeled Cyber Threat Intelligence (CTI) corpus of 957 Advanced Persistent Threats (APT) reports across 8 nation-state actors (APT28, Turla, APT29, Sandworm, Lazarus Group, menuPass, FIN7, APT34). Each question is a de-identified threat report (actor and malware names redacted, IOCs scrubbed); each choice is an APT group name. Distractors are drawn from the same nation-state nexus where possible (e.g., APT28 vs. Turla vs. APT29 within Russia) to match the genuine confusability of real attribution decisions. The final dataset is 219 reports × 2 choices = 438 rows.

---

## Results

*[To be filled in after experiments complete.]*

Expected structure:
- Mean Δ rating by condition (PRO/CON/PRO+CON vs. NO baseline), split by gold vs. distractor and by dataset (SIQA / CQA / cyber)
- Chi-squared tests of homogeneity on rating distributions
- OLS regression: Δ ~ rationale_type + NO_baseline_rating
- Comparison across models: does the reasoning model (deepseek-r1) resist sway more than non-reasoning models?

---

## Conclusions

*[To be filled in after experiments complete.]*

---

## What Didn't Work / Limitations

*[To be filled in after experiments complete — be honest here, this is explicitly graded.]*

Known limitations going in:
- **Rationale source mismatch:** replication uses GPT-4o-generated rationales rather than model-generated ones, which may underestimate sway for models whose own rationales are more persuasive to themselves.
- **Cyber dataset imbalance:** FIN7 has only 7 usable reports vs. 56 for APT28; effects may be noisy for underrepresented actors.
- **Distractor validity:** for single-nexus actors (Lazarus, menuPass, APT34, FIN7) distractors are chosen by TTP overlap rather than nation-state nexus — these may be less genuinely confusable than the Russian-actor pairs.
- **No human baseline:** the paper tests both human and LLM raters; we test LLMs only.

---

## Repository

See [SETUP.md](SETUP.md) for full pipeline documentation, dataset details, and instructions for running experiments.

**Paper:** https://github.com/shramay-palta/AAM
**Cyber dataset source:** TRACER (see SETUP.md)
