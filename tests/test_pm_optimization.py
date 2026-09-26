"""Unit tests for pm_optimization. Run with: python -m pytest"""

import pandas as pd

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
