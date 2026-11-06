"""
Prepare input data for the AAM replication pipeline (Step 2 only).

Reads the paper's pro_con CSVs (which include GPT-4o-generated rationales),
reshapes to long format (2 rows per question: gold + distractor), and writes
a clean CSV ready to feed directly into run_pipeline.py --skip-rationale-gen.

Usage:
    python prepare_data.py --aam-dir "/Users/susan/Documents/AI Projects/AAM" \
                           --output data/prepared.csv
    python prepare_data.py --aam-dir "/path/to/AAM" --dataset siqa --output data/siqa.csv
"""

import argparse
import pandas as pd


AAM_DIR_DEFAULT = "/Users/susan/Documents/AI Projects/AAM"
OUTPUT_DEFAULT = "data/prepared.csv"


def load_dataset(aam_dir: str, dataset: str) -> pd.DataFrame:
    """Load a pro_con CSV and reshape to long format.

    Returns columns: question, choice, answer_label, rationale_pro, rationale_con, dataset
    """
    path = f"{aam_dir}/data/pro_con/{dataset}_human_pro_con.csv"
    df = pd.read_csv(path)

    rows = []
    for _, r in df.iterrows():
        # SIQA has context + question; CQA has question only
        if "context" in df.columns and pd.notna(r.get("context", "")):
            q = f"{r['context']} {r['question']}"
        else:
            q = r["question"]

        rows.append({
            "question": q,
            "choice": r["gold_label"],
            "answer_label": "gold",
            "rationale_pro": r["gold_label_pro_rationale"],
            "rationale_con": r["gold_label_con_rationale"],
            "dataset": dataset,
        })
        rows.append({
            "question": q,
            "choice": r["distractor"],
            "answer_label": "distractor",
            "rationale_pro": r["distractor_pro_rationale"],
            "rationale_con": r["distractor_con_rationale"],
            "dataset": dataset,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare AAM data for LLM rating")
    parser.add_argument("--aam-dir", default=AAM_DIR_DEFAULT,
                        help="Path to the cloned AAM repo")
    parser.add_argument("--dataset", choices=["siqa", "cqa", "both"], default="both",
                        help="Which dataset(s) to include (default: both)")
    parser.add_argument("--output", default=OUTPUT_DEFAULT,
                        help="Output CSV path")
    args = parser.parse_args()

    datasets = ["siqa", "cqa"] if args.dataset == "both" else [args.dataset]

    frames = [load_dataset(args.aam_dir, d) for d in datasets]
    out = pd.concat(frames, ignore_index=True)

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows ({len(out)//2} questions) to {args.output}")
    for d in datasets:
        n = (out["dataset"] == d).sum()
        print(f"  {d}: {n} rows ({n//2} questions)")


if __name__ == "__main__":
    main()
