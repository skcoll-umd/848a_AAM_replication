"""Step 1 — Generate PRO and CON rationales for each (question, choice) pair."""

import pandas as pd
from tqdm import tqdm

from pipeline.ollama_client import generate
from prompts.templates import rationale_pro, rationale_con


def generate_rationales(
    df: pd.DataFrame,
    model: str,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """Add `rationale_pro` and `rationale_con` columns to `df`.

    Expected input columns: question, choice
    Returns a copy of `df` with two new columns appended.
    """
    pros, cons = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Rationales ({model})"):
        pro_prompt = rationale_pro(row["question"], row["choice"])
        con_prompt = rationale_con(row["question"], row["choice"])

        pro_text = generate(model, pro_prompt, temperature=temperature)
        con_text = generate(model, con_prompt, temperature=temperature)

        pros.append(pro_text)
        cons.append(con_text)

    result = df.copy()
    result["rationale_pro"] = pros
    result["rationale_con"] = cons
    return result
