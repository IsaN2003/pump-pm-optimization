"""Unit tests for pm_optimization. Run with: python -m pytest"""

import numpy as np
import pandas as pd
import pytest

import pm_optimization as pm


def make_orders(rows):
    """Build a tiny work-order table in the same format as the CMMS export."""
    df = pd.DataFrame(rows, columns=["Bas. start date", "Order Type", "Equipment"])
    df["date"] = pd.to_datetime(df["Bas. start date"], format="%m/%d/%Y")
    return df


def test_build_failure_times_rules():
    """Check each rule in build_failure_times on a small hand-made example.

    The window runs from 1 Jan to 31 Jan 2020 (30 days), so every expected
    time below is just a day count times 24 hours.
    """
    orders = make_orders([
        ("1/1/2020", "PMRT", 1),   # window starts
        ("1/11/2020", "CMRT", 1),  # pump 1 fails on day 10
        ("1/21/2020", "CMTA", 1),  # a later failure: must be ignored
        ("1/1/2020", "CMRT", 2),   # failure at t = 0: must be skipped...
        ("1/6/2020", "CMRT", 2),   # ...so pump 2 fails on day 5
        ("1/5/2020", "PMRT", 3),   # pump 3 never fails
        ("1/31/2020", "PMTA", 3),  # PMRT + PMTA: must NOT drop pump 3; window ends
        ("1/15/2020", "GSRT", 4),  # other order types: must be ignored
    ])

    result = pm.build_failure_times(orders).set_index("unit_id")

    # Pump 4 only has an ignored order type, so it shouldn't appear at all
    assert set(result.index) == {1, 2, 3}
    # Pumps 1 and 2 fail; pump 3 is censored at the full window length
    assert result.loc[1, "hours"] == 10 * 24 and result.loc[1, "failed"] == 1
    assert result.loc[2, "hours"] == 5 * 24 and result.loc[2, "failed"] == 1
    assert result.loc[3, "hours"] == 30 * 24 and result.loc[3, "failed"] == 0


def test_anonymize_units():
    """Pumps are relabelled P01, P02, ... in time order, whatever order the rows arrive in."""
    table = pd.DataFrame({"unit_id": [30, 10, 20],
                          "hours": [720.0, 240.0, 120.0], "failed": [0, 1, 1]})
    result = pm.anonymize_units(table)
    assert list(result["unit_id"]) == ["P01", "P02", "P03"]
    assert list(result["hours"]) == [120.0, 240.0, 720.0]
    # Shuffling the input rows gives the same labels
    shuffled = pm.anonymize_units(table.sample(frac=1, random_state=0))
    pd.testing.assert_frame_equal(result, shuffled)


# Weibull analysis
def test_adjusted_rank_equals_rank_without_suspensions():
    """With no suspensions, Johnson's adjusted rank is just the plain rank."""
    table = pm.median_rank_table([100, 200, 300, 400], [1, 1, 1, 1])
    assert list(table["adj_rank"]) == [1, 2, 3, 4]


def test_suspension_pushes_later_failures_up():
    """A suspension before a failure raises that failure's adjusted rank."""
    # failure, suspension, failure: the 2nd failure's adjusted rank must jump past 2
    table = pm.median_rank_table([100, 150, 200], [1, 0, 1])
    assert table["adj_rank"].iloc[0] == pytest.approx(1.0)
    assert table["adj_rank"].iloc[1] == pytest.approx(1 + (4 - 1) / (1 + 1))  # = 2.5


@pytest.mark.parametrize("beta, eta", [(0.8, 1_000), (1.5, 50_000), (3.0, 10)])
def test_fit_recovers_known_parameters(beta, eta):
    """Fitting a large simulated sample gives back the beta and eta it came from."""
    # Fixed seed so the test is repeatable; 3000 failures keeps the error within 5%
    t = eta * np.random.default_rng(0).weibull(beta, 3000)
    fit = pm.fit_weibull_mrr(t, np.ones(len(t), dtype=int))
    assert fit["beta"] == pytest.approx(beta, rel=0.05)
    assert fit["eta_hours"] == pytest.approx(eta, rel=0.05)


def test_reliability_basics():
    """R(t) and F(t) always add up to 1, and R(eta) = e^-1 for any beta."""
    t = np.linspace(0, 100, 11)
    np.testing.assert_allclose(pm.reliability(t, 1.2, 40) + pm.failure_prob(t, 1.2, 40), 1)
    assert pm.reliability(40, 1.2, 40) == pytest.approx(np.exp(-1))  # R(η) = 36.8%


def test_save_and_load_params(tmp_path):
    """Parameters survive a save/load round trip, with eta_months added on save."""
    path = tmp_path / "params.json"
    pm.save_params({"beta": 1.2, "eta_hours": 730.08}, path)
    loaded = pm.load_params(path)
    assert loaded["beta"] == 1.2
    assert loaded["eta_months"] == pytest.approx(730.08 / pm.HOURS_PER_MONTH)
