"""Tests for analysis/analyze.py — replication of Palta et al. (2026)."""

import os
import pandas as pd
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Task 1: load_results
# ---------------------------------------------------------------------------

def test_load_results_columns():
    from analysis.analyze import load_results
    df = load_results("results/")
    expected = {"question", "choice", "condition", "rating", "model", "answer_label", "dataset"}
    assert expected.issubset(df.columns)


def test_load_results_conditions():
    from analysis.analyze import load_results
    df = load_results("results/")
    assert set(df["condition"].unique()) == {"NO", "PRO", "CON", "PRO+CON"}


def test_load_results_answer_labels():
    from analysis.analyze import load_results
    df = load_results("results/")
    assert set(df["answer_label"].unique()) == {"gold", "distractor"}


# ---------------------------------------------------------------------------
# Task 2: compute_deltas
# ---------------------------------------------------------------------------

def test_compute_deltas_columns():
    from analysis.analyze import load_results, compute_deltas
    df = load_results("results/")
    deltas = compute_deltas(df)
    for col in ["no_rating", "delta_PRO", "delta_CON", "delta_PRO+CON"]:
        assert col in deltas.columns


def test_compute_deltas_drops_missing_no():
    from analysis.analyze import load_results, compute_deltas
    df = load_results("results/")
    deltas = compute_deltas(df)
    assert deltas["no_rating"].isna().sum() == 0


def test_compute_deltas_values():
    from analysis.analyze import compute_deltas
    rows = [
        {"question": "Q", "choice": "A", "condition": "NO",      "rating": 3.0, "model": "m", "answer_label": "gold", "dataset": "siqa"},
        {"question": "Q", "choice": "A", "condition": "PRO",     "rating": 4.0, "model": "m", "answer_label": "gold", "dataset": "siqa"},
        {"question": "Q", "choice": "A", "condition": "CON",     "rating": 2.0, "model": "m", "answer_label": "gold", "dataset": "siqa"},
        {"question": "Q", "choice": "A", "condition": "PRO+CON", "rating": 3.0, "model": "m", "answer_label": "gold", "dataset": "siqa"},
    ]
    df = pd.DataFrame(rows)
    deltas = compute_deltas(df)
    assert deltas.iloc[0]["delta_PRO"] == 1.0
    assert deltas.iloc[0]["delta_CON"] == -1.0
    assert deltas.iloc[0]["delta_PRO+CON"] == 0.0


# ---------------------------------------------------------------------------
# Task 3: table_mean_ratings
# ---------------------------------------------------------------------------

def test_table_mean_ratings_shape():
    from analysis.analyze import load_results, compute_deltas, table_mean_ratings
    df = load_results("results/")
    deltas = compute_deltas(df)
    t3 = table_mean_ratings(df, deltas)
    assert len(t3) >= 8
    assert "mean_rating" in t3.columns
    assert "mean_delta" in t3.columns


def test_table_mean_ratings_no_delta_is_zero():
    from analysis.analyze import load_results, compute_deltas, table_mean_ratings
    df = load_results("results/")
    deltas = compute_deltas(df)
    t3 = table_mean_ratings(df, deltas)
    no_rows = t3[t3["condition"] == "NO"]
    assert (no_rows["mean_delta"] == 0.0).all()


# ---------------------------------------------------------------------------
# Task 4: table_pct_change
# ---------------------------------------------------------------------------

def test_table_pct_change_sums_to_100():
    from analysis.analyze import load_results, compute_deltas, table_pct_change
    df = load_results("results/")
    deltas = compute_deltas(df)
    t2 = table_pct_change(deltas)
    total = t2["pct_increase"] + t2["pct_decrease"] + t2["pct_no_change"]
    assert (total.round(1) == 100.0).all()


def test_table_pct_change_conditions():
    from analysis.analyze import load_results, compute_deltas, table_pct_change
    df = load_results("results/")
    deltas = compute_deltas(df)
    t2 = table_pct_change(deltas)
    assert set(t2["condition"].unique()) == {"PRO", "CON", "PRO+CON"}


# ---------------------------------------------------------------------------
# Task 5: chi_squared_tests and ols_regression
# ---------------------------------------------------------------------------

def test_chi_squared_p_values_in_range():
    from analysis.analyze import load_results, chi_squared_tests
    df = load_results("results/")
    chi = chi_squared_tests(df)
    assert chi["p_value"].between(0, 1).all()
    assert set(chi["condition"].unique()) == {"PRO", "CON", "PRO+CON"}


def test_ols_regression_columns():
    from analysis.analyze import load_results, compute_deltas, ols_regression
    df = load_results("results/")
    deltas = compute_deltas(df)
    ols = ols_regression(deltas)
    for col in ["answer_label", "feature", "coefficient", "p_value", "significant"]:
        assert col in ols.columns


def test_ols_no_rating_negative_coefficient():
    """Paper finding: NO baseline rating has a negative coefficient (anchoring effect)."""
    from analysis.analyze import load_results, compute_deltas, ols_regression
    df = load_results("results/")
    deltas = compute_deltas(df)
    ols = ols_regression(deltas)
    no_rating_rows = ols[ols["feature"] == "no_rating"]
    assert (no_rating_rows["coefficient"] < 0).all()


# ---------------------------------------------------------------------------
# Task 6: plot_ratings
# ---------------------------------------------------------------------------

def test_plot_ratings_creates_file(tmp_path):
    from analysis.analyze import load_results, plot_ratings
    df = load_results("results/")
    plot_ratings(df, str(tmp_path))
    pngs = list(tmp_path.glob("violin_*.png"))
    assert len(pngs) >= 1
    for f in pngs:
        assert f.stat().st_size > 0


# ---------------------------------------------------------------------------
# Task 7: main()
# ---------------------------------------------------------------------------

def test_main_runs_without_error():
    from analysis.analyze import main
    main()


def test_main_creates_summary_csv():
    from analysis.analyze import main
    main()
    assert os.path.exists("results/analysis_summary.csv")
