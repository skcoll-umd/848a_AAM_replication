"""Analysis script — replicates Tables 2, 3, 4 and Figures 2/3 from Palta et al. (2026).

Overview
--------
The pipeline produces one ratings CSV per model run. This script loads all of them,
computes per-item deltas relative to the NO-rationale baseline, and runs three levels
of analysis that mirror the paper:

  Table 3  — mean raw ratings + mean Δ, broken out by condition × answer_label × model × dataset
  Table 2  — % of items that went up / down / stayed the same under each condition
  Chi-sq   — test of homogeneity: does the rating *distribution* shift vs NO? (1-5 Likert bins)
  Table 4  — OLS regression: Δ ~ condition + NO_baseline_rating
               • condition coefficients = average sway attributable to PRO/CON/PRO+CON
               • no_rating coefficient = anchoring effect (high initial rating → smaller shift)
  Figures  — violin plots matching paper Figures 2/3: rating distributions by condition,
               one plot per dataset, gold vs. distractor rows, models as columns

All outputs are split by `dataset` (siqa / cqa / cyber) so replication and extension
results are always kept separate.
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Color scheme from the paper (Figure 2/3 legend)
CONDITION_COLORS = {
    "NO":      "#FFD700",  # yellow  — no rationale, the baseline
    "PRO":     "#9B59B6",  # purple  — rationale argues the choice is plausible
    "CON":     "#E74C3C",  # red     — rationale argues the choice is implausible
    "PRO+CON": "#27AE60",  # green   — both rationales shown together
}
CONDITION_ORDER = ["NO", "PRO", "CON", "PRO+CON"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> pd.DataFrame:
    """Load all ratings CSVs from results_dir and return a combined DataFrame.

    Each CSV is one model's output from run_pipeline.py. Expected columns:
        question, choice, condition, rating, model, answer_label, dataset

    Multiple CSVs are simply stacked — the model column distinguishes them.
    """
    paths = glob.glob(os.path.join(results_dir, "*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {results_dir!r}")
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-item rating deltas vs the NO-rationale baseline.

    For each (question, choice, model) triplet the pipeline produces 4 ratings —
    one per condition. This function pivots to wide format so each row is one item,
    then computes:

        delta_PRO     = rating_PRO     - rating_NO
        delta_CON     = rating_CON     - rating_NO
        delta_PRO+CON = rating_PRO+CON - rating_NO

    A positive delta means the rationale raised the plausibility rating vs baseline;
    negative means it suppressed it.  Items missing a NO rating are dropped (can't
    compute a delta without a baseline).

    Returns wide-format DataFrame with columns:
        question, choice, model, answer_label, dataset,
        no_rating, delta_PRO, delta_CON, delta_PRO+CON
    """
    index_cols = ["question", "choice", "model", "answer_label", "dataset"]
    pivot = df.pivot_table(
        index=index_cols,
        columns="condition",
        values="rating",
        aggfunc="mean",  # handles any duplicate rows from reruns
    ).reset_index()

    pivot = pivot.rename(columns={"NO": "no_rating"})
    pivot = pivot.dropna(subset=["no_rating"])

    for cond in ["PRO", "CON", "PRO+CON"]:
        if cond in pivot.columns:
            pivot[f"delta_{cond}"] = pivot[cond] - pivot["no_rating"]

    # Ensure delta columns exist even if a condition is absent in a partial run
    for cond in ["PRO", "CON", "PRO+CON"]:
        if f"delta_{cond}" not in pivot.columns:
            pivot[f"delta_{cond}"] = np.nan

    keep = index_cols + ["no_rating", "delta_PRO", "delta_CON", "delta_PRO+CON"]
    return pivot[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 3 — mean ratings and deltas
# ---------------------------------------------------------------------------

def table_mean_ratings(df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
    """Return Table 3 equivalent: mean rating + mean Δ by condition × answer_label × model × dataset.

    Table 3 in the paper shows two things side by side:
      - The absolute mean rating under each condition (1–5 Likert scale)
      - The mean shift (Δ) relative to the NO baseline

    Splitting by answer_label (gold vs. distractor) is what reveals the asymmetry:
    the paper finds PRO rationales *raise* distractor ratings but can *lower* gold
    label ratings — the model is paradoxically less confident in the right answer
    after seeing an argument in its favor.
    """
    # Mean raw ratings per condition
    raw = (
        df.groupby(["model", "dataset", "answer_label", "condition"])["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "mean_rating"})
    )

    # Mean deltas — melt delta_df from wide to long so we can group by condition
    delta_long = delta_df.melt(
        id_vars=["model", "dataset", "answer_label"],
        value_vars=["delta_PRO", "delta_CON", "delta_PRO+CON"],
        var_name="condition",
        value_name="delta",
    )
    delta_long["condition"] = delta_long["condition"].str.replace("delta_", "", regex=False)
    delta_means = (
        delta_long.groupby(["model", "dataset", "answer_label", "condition"])["delta"]
        .mean()
        .reset_index()
        .rename(columns={"delta": "mean_delta"})
    )

    result = raw.merge(delta_means, on=["model", "dataset", "answer_label", "condition"], how="left")
    result["mean_delta"] = result["mean_delta"].fillna(0.0)  # NO condition Δ is defined as 0
    return result


def print_table3(t3: pd.DataFrame) -> None:
    """Print Table 3 equivalent to stdout."""
    print("\n=== Table 3: Mean Plausibility Ratings and Δ from NO Baseline ===")
    display = t3.copy()
    display["mean_rating"] = display["mean_rating"].map("{:.2f}".format)
    display["mean_delta"] = display["mean_delta"].map("{:+.2f}".format)
    print(display.to_string(index=False))


# ---------------------------------------------------------------------------
# Table 2 — % increase / decrease / no change
# ---------------------------------------------------------------------------

def table_pct_change(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Return Table 2 equivalent: % of items whose rating increased / decreased / stayed the same.

    Rather than averaging deltas (which can mask opposing shifts), Table 2 asks:
    for what fraction of individual (question, choice) pairs did the rationale
    move the rating at all?  This is a distribution-free view of sway magnitude.

    The paper uses this to show that PRO rationales increase ratings for the
    majority of distractor items while CON rationales decrease ratings for most
    gold items — the effect is consistent across items, not just an average artifact.
    """
    delta_long = delta_df.melt(
        id_vars=["model", "dataset", "answer_label"],
        value_vars=["delta_PRO", "delta_CON", "delta_PRO+CON"],
        var_name="condition",
        value_name="delta",
    ).dropna(subset=["delta"])
    delta_long["condition"] = delta_long["condition"].str.replace("delta_", "", regex=False)

    rows = []
    for keys, group in delta_long.groupby(["model", "dataset", "answer_label", "condition"]):
        n = len(group)
        rows.append({
            "model": keys[0],
            "dataset": keys[1],
            "answer_label": keys[2],
            "condition": keys[3],
            "pct_increase":   round(100 * (group["delta"] > 0).sum() / n, 1),
            "pct_decrease":   round(100 * (group["delta"] < 0).sum() / n, 1),
            "pct_no_change":  round(100 * (group["delta"] == 0).sum() / n, 1),
        })

    result = pd.DataFrame(rows)
    # Recompute no_change from the other two to avoid floating point rounding errors
    # making rows sum to 99.9% or 100.1%
    result["pct_no_change"] = (
        100 - result["pct_increase"] - result["pct_decrease"]
    ).round(1)
    return result


def print_table2(t2: pd.DataFrame) -> None:
    """Print Table 2 equivalent to stdout."""
    print("\n=== Table 2: % Increase / Decrease / No Change vs NO Baseline ===")
    display = t2.copy()
    for col in ["pct_increase", "pct_decrease", "pct_no_change"]:
        display[col] = display[col].map("{:.0f}%".format)
    print(display.to_string(index=False))


# ---------------------------------------------------------------------------
# Statistical tests: chi-squared homogeneity
# ---------------------------------------------------------------------------

def chi_squared_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-squared test of homogeneity between the NO baseline and each condition.

    The question: does adding a rationale shift the *shape* of the rating
    distribution, not just the mean?  A significant result means the rationale
    changed how ratings are spread across the 1–5 Likert scale — e.g., moving
    mass from 3s to 1s, rather than just nudging everyone down by 0.3 points.

    Method: for each (model, dataset, answer_label) group, we build a 2×5
    contingency table (NO ratings vs. condition ratings, binned 1–5) and run
    scipy's chi2_contingency.  Empty bins present in both rows are dropped
    before the test to avoid inflating degrees of freedom.

    Low p-values (< 0.05) indicate the rationale meaningfully redistributed
    ratings, not just shifted the mean.
    """
    rows = []
    for (model, dataset, answer_label), group in df.groupby(["model", "dataset", "answer_label"]):
        no_counts = group[group["condition"] == "NO"]["rating"].dropna()
        for cond in ["PRO", "CON", "PRO+CON"]:
            cond_counts = group[group["condition"] == cond]["rating"].dropna()
            if len(cond_counts) < 2:
                continue
            # Count how many ratings fell on each point of the 1–5 scale
            bins = [1, 2, 3, 4, 5]
            no_hist = pd.Series(no_counts.round().astype(int)).value_counts().reindex(bins, fill_value=0)
            cond_hist = pd.Series(cond_counts.round().astype(int)).value_counts().reindex(bins, fill_value=0)
            contingency = np.array([no_hist.values, cond_hist.values])
            # Drop Likert bins where both conditions have zero observations —
            # these contribute nothing to the test and can cause zero-division
            mask = contingency.sum(axis=0) > 0
            if mask.sum() < 2:
                continue
            chi2, p, _, _ = stats.chi2_contingency(contingency[:, mask])
            rows.append({
                "model": model, "dataset": dataset, "answer_label": answer_label,
                "condition": cond, "chi2": chi2, "p_value": p,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OLS regression (Table 4)
# ---------------------------------------------------------------------------

def ols_regression(delta_df: pd.DataFrame) -> pd.DataFrame:
    """OLS regression: Δ ~ C(condition) + no_rating, fit separately per model × dataset × answer_label.

    Two questions this answers:

    1. Condition coefficients (PRO, CON, PRO+CON vs implicit NO baseline):
       How much does each rationale type shift the rating on average, holding
       everything else constant?  A large positive PRO coefficient means PRO
       rationales reliably inflate ratings; a large negative CON coefficient
       means CON rationales reliably suppress them.

    2. no_rating coefficient (anchoring):
       The paper finds that items with a high NO baseline rating change less
       under rationale conditions — the model is already confident and harder
       to move.  A negative no_rating coefficient confirms this anchoring effect.
       We expect this to be negative: the higher the starting point, the smaller
       the subsequent shift.

    Fit separately per (model, dataset, answer_label) so we can compare:
      - Does deepseek-r1 show smaller condition coefficients than qwen/gemma? (reasoning resistance)
      - Is the cyber dataset more or less susceptible than SIQA/CQA? (domain transfer)
      - Is the gold vs distractor asymmetry in the coefficients consistent with the paper?
    """
    delta_long = delta_df.melt(
        id_vars=["model", "dataset", "answer_label", "no_rating"],
        value_vars=["delta_PRO", "delta_CON", "delta_PRO+CON"],
        var_name="condition",
        value_name="delta",
    ).dropna(subset=["delta"])
    delta_long["condition"] = delta_long["condition"].str.replace("delta_", "", regex=False)

    rows = []
    for (model, dataset, answer_label), group in delta_long.groupby(["model", "dataset", "answer_label"]):
        if group["condition"].nunique() < 2:
            continue  # need variation in condition to fit
        try:
            fit = smf.ols("delta ~ C(condition) + no_rating", data=group).fit()
            for feature, coef in fit.params.items():
                pval = fit.pvalues[feature]
                rows.append({
                    "model": model, "dataset": dataset, "answer_label": answer_label,
                    "feature": feature, "coefficient": coef,
                    "p_value": pval, "significant": pval < 0.05,
                })
        except Exception:
            continue
    return pd.DataFrame(rows)


def print_table4(ols: pd.DataFrame) -> None:
    """Print Table 4 equivalent to stdout."""
    print("\n=== Table 4: OLS Regression Coefficients (Δ ~ condition + NO baseline) ===")
    display = ols.copy()
    display["coefficient"] = display["coefficient"].map("{:.4f}".format)
    display["p_value"] = display["p_value"].map("{:.4f}".format)
    display["significant"] = display["significant"].map(lambda x: "**" if x else "")
    print(display.to_string(index=False))


# ---------------------------------------------------------------------------
# Violin plots (Figures 2/3)
# ---------------------------------------------------------------------------

def plot_ratings(df: pd.DataFrame, output_dir: str) -> None:
    """Generate violin plots matching paper Figures 2/3. One PNG per dataset.

    Layout: 2 rows (gold / distractor) × N columns (one per model).
    Each cell shows the distribution of raw ratings under all 4 conditions.

    Violin plots are more informative than bar charts here because we care
    about the full shape of the distribution — not just the mean. The paper's
    key asymmetry shows up as the PRO violin being right-skewed for distractors
    but not for gold labels, and CON being left-skewed for gold labels.

    One PNG is saved per dataset (siqa, cqa, cyber) to results/figures/.
    """
    os.makedirs(output_dir, exist_ok=True)

    for dataset, dataset_df in df.groupby("dataset"):
        models = sorted(dataset_df["model"].unique())
        n_models = len(models)
        fig, axes = plt.subplots(
            2, n_models,
            figsize=(4 * n_models, 8),
            sharey=True,   # shared y-axis so rating scales are directly comparable across models
            squeeze=False,
        )
        fig.suptitle(
            f"{dataset.upper()} Ratings by Model for Different Rationale Conditions",
            fontsize=13, y=1.01,
        )

        label_rows = ["Gold Label Ratings", "Distractor Ratings"]
        answer_labels = ["gold", "distractor"]

        for col, model in enumerate(models):
            model_df = dataset_df[dataset_df["model"] == model]
            for row, (answer_label, row_title) in enumerate(zip(answer_labels, label_rows)):
                ax = axes[row][col]
                plot_df = model_df[model_df["answer_label"] == answer_label]

                sns.violinplot(
                    data=plot_df,
                    x="condition", y="rating",
                    hue="condition",
                    order=CONDITION_ORDER,
                    palette=CONDITION_COLORS,
                    inner="box",    # show median + IQR box inside the violin
                    legend=False,
                    ax=ax,
                )
                ax.set_ylim(1, 5)
                ax.set_yticks([1, 2, 3, 4, 5])
                ax.set_xlabel("")
                ax.set_ylabel("Mean Rating" if col == 0 else "")
                ax.set_title(model if row == 0 else "")

            axes[0][col].set_title(model, fontsize=10)

        # Row labels on the left margin (gold / distractor)
        for row, row_title in enumerate(label_rows):
            axes[row][0].annotate(
                row_title, xy=(0, 0.5), xytext=(-axes[row][0].yaxis.labelpad - 10, 0),
                xycoords=axes[row][0].yaxis.label, textcoords="offset points",
                ha="right", va="center", fontsize=11, fontweight="bold",
                rotation=90,
            )

        # Shared legend at the top
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=CONDITION_COLORS[c], label=f"{c} Rationale") for c in CONDITION_ORDER]
        fig.legend(handles=legend_elements, loc="upper center", ncol=4,
                   bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"violin_{dataset}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Load all model outputs from results/ — one CSV per model run
    df = load_results("results/")

    # Compute per-item deltas vs NO baseline (wide format, one row per item)
    delta_df = compute_deltas(df)

    # Table 3: mean ratings + mean Δ by condition × answer_label × model × dataset
    t3 = table_mean_ratings(df, delta_df)
    print_table3(t3)

    # Table 2: fraction of items that went up / down / no change under each condition
    t2 = table_pct_change(delta_df)
    print_table2(t2)

    # Chi-squared: does the rating distribution shift vs NO (not just the mean)?
    chi = chi_squared_tests(df)
    print("\n=== Chi-Squared Tests of Homogeneity vs NO Baseline ===")
    print(chi.to_string(index=False))

    # Table 4: OLS — isolates condition effect and anchoring effect (no_rating coef)
    ols = ols_regression(delta_df)
    print_table4(ols)

    # Figures 2/3: violin plots, one PNG per dataset
    plot_ratings(df, "results/figures/")
    print("\nPlots saved to results/figures/")

    # Save Table 3 as CSV for easy reference / sharing
    t3.to_csv("results/analysis_summary.csv", index=False)
    print("Summary saved to results/analysis_summary.csv")


if __name__ == "__main__":
    main()
