"""Analysis script — replicates Tables 2, 3, 4 and Figures 2/3 from Palta et al. (2026)."""

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

# Paper colour scheme
CONDITION_COLORS = {
    "NO":      "#FFD700",
    "PRO":     "#9B59B6",
    "CON":     "#E74C3C",
    "PRO+CON": "#27AE60",
}
CONDITION_ORDER = ["NO", "PRO", "CON", "PRO+CON"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> pd.DataFrame:
    """Load all ratings CSVs from results_dir and return a combined DataFrame."""
    paths = glob.glob(os.path.join(results_dir, "*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {results_dir!r}")
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-item rating deltas vs NO baseline.

    Returns wide-format DataFrame with columns:
        question, choice, model, answer_label, dataset,
        no_rating, delta_PRO, delta_CON, delta_PRO+CON
    """
    index_cols = ["question", "choice", "model", "answer_label", "dataset"]
    pivot = df.pivot_table(
        index=index_cols,
        columns="condition",
        values="rating",
        aggfunc="mean",
    ).reset_index()

    pivot = pivot.rename(columns={"NO": "no_rating"})
    pivot = pivot.dropna(subset=["no_rating"])

    for cond in ["PRO", "CON", "PRO+CON"]:
        if cond in pivot.columns:
            pivot[f"delta_{cond}"] = pivot[cond] - pivot["no_rating"]

    # Ensure delta columns exist even if condition absent in data
    for cond in ["PRO", "CON", "PRO+CON"]:
        if f"delta_{cond}" not in pivot.columns:
            pivot[f"delta_{cond}"] = np.nan

    keep = index_cols + ["no_rating", "delta_PRO", "delta_CON", "delta_PRO+CON"]
    return pivot[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 3 — mean ratings and deltas
# ---------------------------------------------------------------------------

def table_mean_ratings(df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
    """Return Table 3 equivalent: mean rating + Δ by condition × answer_label × model × dataset."""
    # Mean raw ratings
    raw = (
        df.groupby(["model", "dataset", "answer_label", "condition"])["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "mean_rating"})
    )

    # Mean deltas — melt delta_df to long
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
    result["mean_delta"] = result["mean_delta"].fillna(0.0)  # NO condition Δ = 0
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
    """Return Table 2 equivalent: % increase/decrease/no change by condition × model × dataset × answer_label."""
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
    # Fix floating point rounding so rows sum to exactly 100
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
    """Chi-squared test of homogeneity between NO and each other condition."""
    rows = []
    for (model, dataset, answer_label), group in df.groupby(["model", "dataset", "answer_label"]):
        no_counts = group[group["condition"] == "NO"]["rating"].dropna()
        for cond in ["PRO", "CON", "PRO+CON"]:
            cond_counts = group[group["condition"] == cond]["rating"].dropna()
            if len(cond_counts) < 2:
                continue
            # Build contingency table on integer bins 1-5
            bins = [1, 2, 3, 4, 5]
            no_hist = pd.Series(no_counts.round().astype(int)).value_counts().reindex(bins, fill_value=0)
            cond_hist = pd.Series(cond_counts.round().astype(int)).value_counts().reindex(bins, fill_value=0)
            contingency = np.array([no_hist.values, cond_hist.values])
            # Drop columns where both are zero
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
    """OLS: Δ ~ C(condition) + no_rating, by answer_label × model × dataset."""
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
            continue
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
    """Generate violin plots matching paper Figures 2/3. One PNG per dataset."""
    os.makedirs(output_dir, exist_ok=True)

    for dataset, dataset_df in df.groupby("dataset"):
        models = sorted(dataset_df["model"].unique())
        n_models = len(models)
        fig, axes = plt.subplots(
            2, n_models,
            figsize=(4 * n_models, 8),
            sharey=True,
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
                    inner="box",
                    legend=False,
                    ax=ax,
                )
                ax.set_ylim(1, 5)
                ax.set_yticks([1, 2, 3, 4, 5])
                ax.set_xlabel("")
                ax.set_ylabel("Mean Rating" if col == 0 else "")
                ax.set_title(model if row == 0 else "")

            axes[0][col].set_title(model, fontsize=10)

        # Row labels
        for row, row_title in enumerate(label_rows):
            axes[row][0].annotate(
                row_title, xy=(0, 0.5), xytext=(-axes[row][0].yaxis.labelpad - 10, 0),
                xycoords=axes[row][0].yaxis.label, textcoords="offset points",
                ha="right", va="center", fontsize=11, fontweight="bold",
                rotation=90,
            )

        # Legend
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
    df = load_results("results/")
    delta_df = compute_deltas(df)

    t3 = table_mean_ratings(df, delta_df)
    print_table3(t3)

    t2 = table_pct_change(delta_df)
    print_table2(t2)

    chi = chi_squared_tests(df)
    print("\n=== Chi-Squared Tests of Homogeneity vs NO Baseline ===")
    print(chi.to_string(index=False))

    ols = ols_regression(delta_df)
    print_table4(ols)

    plot_ratings(df, "results/figures/")
    print("\nPlots saved to results/figures/")

    t3.to_csv("results/analysis_summary.csv", index=False)
    print("Summary saved to results/analysis_summary.csv")


if __name__ == "__main__":
    main()
