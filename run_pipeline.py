"""
AAM Replication Pipeline — entrypoint.

Usage:
    # Full pipeline — generate rationales then rate (original flow)
    python run_pipeline.py --model qwen3:32b --input data/siqa_sample.csv

    # Skip rationale generation — use pre-generated rationales from prepared CSV
    python run_pipeline.py --model qwen3:32b --input data/prepared.csv --skip-rationale-gen

    # Multiple models
    python run_pipeline.py --model qwen3:32b gemma4:31b --input data/prepared.csv --skip-rationale-gen
"""

import argparse
import os
import pandas as pd

from pipeline.rationale_gen import generate_rationales
from pipeline.plausibility_rate import rate_plausibility
from pipeline.ollama_client import list_models


RESULTS_DIR = "results"
RATIONALE_COLS = {"rationale_pro", "rationale_con"}


def load_input(path: str, skip_rationale_gen: bool) -> pd.DataFrame:
    """Load and validate the input CSV."""
    df = pd.read_csv(path)
    required = {"question", "choice"}
    if skip_rationale_gen:
        required |= RATIONALE_COLS
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")
    return df


def run(model: str, df: pd.DataFrame, output_dir: str, skip_rationale_gen: bool) -> str:
    os.makedirs(output_dir, exist_ok=True)

    if skip_rationale_gen:
        df_with_rationales = df
    else:
        # Step 1 — generate rationales (temp=1)
        df_with_rationales = generate_rationales(df, model=model, temperature=1.0)

    # Step 2 — rate plausibility under 4 conditions (temp=0.1 for non-OpenAI)
    df_ratings = rate_plausibility(df_with_rationales, model=model, temperature=0.1)

    # Carry through extra columns (answer_label, dataset) if present
    extra_cols = [c for c in df_with_rationales.columns
                  if c not in {"question", "choice", "rationale_pro", "rationale_con"}]
    if extra_cols:
        df_ratings = df_ratings.merge(
            df_with_rationales[["question", "choice"] + extra_cols].drop_duplicates(),
            on=["question", "choice"],
            how="left",
        )

    out_path = os.path.join(output_dir, f"ratings_{model.replace(':', '-')}.csv")
    df_ratings.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="AAM replication pipeline")
    parser.add_argument(
        "--model",
        nargs="+",
        default=["qwen3:32b"],
        help="Ollama model name(s) to use",
    )
    parser.add_argument(
        "--input",
        default="data/siqa_sample.csv",
        help="Path to input CSV",
    )
    parser.add_argument(
        "--output",
        default=RESULTS_DIR,
        help="Directory for output CSVs",
    )
    parser.add_argument(
        "--skip-rationale-gen",
        action="store_true",
        help="Skip Step 1; input CSV must already have rationale_pro and rationale_con columns",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print locally available Ollama models and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        for m in list_models():
            print(m)
        return

    df = load_input(args.input, args.skip_rationale_gen)
    print(f"Loaded {len(df)} rows from {args.input}")
    if args.skip_rationale_gen:
        print("Skipping rationale generation (pre-generated rationales found in input)")

    all_results = []
    for model in args.model:
        print(f"\n=== Model: {model} ===")
        out_path = run(model, df, args.output, args.skip_rationale_gen)
        all_results.append(out_path)

    # Merge all model outputs into one combined CSV
    if len(all_results) > 1:
        combined = pd.concat([pd.read_csv(p) for p in all_results], ignore_index=True)
        combined_path = os.path.join(args.output, "ratings_all_models.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined → {combined_path}")


if __name__ == "__main__":
    main()
