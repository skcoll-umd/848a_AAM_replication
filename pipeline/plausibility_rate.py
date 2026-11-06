"""Step 2 — Rate plausibility under 4 conditions and return long-format records."""

import re
import pandas as pd
from tqdm import tqdm

from pipeline.ollama_client import generate
from prompts.templates import (
    rating_no_rationale,
    rating_pro_only,
    rating_con_only,
    rating_pro_con,
)

CONDITIONS = ["NO", "PRO", "CON", "PRO+CON"]


def _parse_rating(response: str) -> int | None:
    """Extract the first integer 1-5 from the model response."""
    match = re.search(r"\b([1-5])\b", response)
    return int(match.group(1)) if match else None


def rate_plausibility(
    df: pd.DataFrame,
    model: str,
    temperature: float = 0.0,
) -> pd.DataFrame:
    """Rate each (question, choice) under all 4 conditions.

    Expected input columns: question, choice, rationale_pro, rationale_con
    Returns a long-format DataFrame with columns:
        question, choice, condition, rating, rating_raw, model
    """
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Ratings ({model})"):
        q, c = row["question"], row["choice"]
        pro, con = row["rationale_pro"], row["rationale_con"]

        condition_prompts = {
            "NO": rating_no_rationale(q, c),
            "PRO": rating_pro_only(q, c, pro),
            "CON": rating_con_only(q, c, con),
            "PRO+CON": rating_pro_con(q, c, pro, con),
        }

        for condition, prompt in condition_prompts.items():
            raw = generate(model, prompt, temperature=temperature)
            rating = _parse_rating(raw)
            records.append(
                {
                    "question": q,
                    "choice": c,
                    "condition": condition,
                    "rating": rating,
                    "rating_raw": raw,
                    "model": model,
                }
            )

    return pd.DataFrame(records)
