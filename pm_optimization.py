"""Shared code for the pump preventive-maintenance optimization project.

The notebooks import this module so that paths, work-order rules and the
data-preparation logic live in one place and can be unit tested.
"""
from pathlib import Path

import pandas as pd

# Paths
# Everything is resolved relative to this file, so the notebooks work no
# matter which directory Jupyter was started from.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FIG_DIR = ROOT / "figures"

RAW_WORK_ORDERS = RAW_DIR / "work_orders.csv"
FAILURE_TIMES = DATA_DIR / "failure_times.csv"

# Work-order classification
# Other order types in the export (GSRT, EPRT, RWRT, ...) say nothing about
# whether the pump failed, so they are left out of the analysis.
FAILURE_ORDER_TYPES = ("CMRT", "CMTA")  # corrective maintenance -> failure
PM_ORDER_TYPES = ("PMRT", "PMTA")       # preventive maintenance -> pump in service


# Data preparation
def load_work_orders(path=RAW_WORK_ORDERS):
    """Load the CMMS export and parse the start date."""
    df = pd.read_csv(path)
    # The export uses US-style dates. Giving the format explicitly avoids
    # pandas guessing and swapping day and month on ambiguous dates.
    df["date"] = pd.to_datetime(df["Bas. start date"], format="%m/%d/%Y")
    return df


def build_failure_times(work_orders):
    """Reduce work orders to one time-to-failure record per pump.

    Rules:
    - Only corrective (CMRT, CMTA) and preventive (PMRT, PMTA) orders are used.
    - Time is measured in hours from the first order in the window.
    - A pump's first corrective order (after t = 0) is its failure (failed = 1).
    - A pump with PM orders but no failure is censored at the window length (failed = 0).

    Returns a DataFrame with columns unit_id, hours and failed, ready for
    Weibull / survival fitting.
    """
    wo = work_orders[work_orders["Order Type"].isin(FAILURE_ORDER_TYPES + PM_ORDER_TYPES)].copy()

    # The observation window is shared by all pumps: it runs from the first
    # to the last relevant order in the whole export.
    start, end = wo["date"].min(), wo["date"].max()
    window_hours = (end - start).total_seconds() / 3600
    wo["hours"] = (wo["date"] - start).dt.total_seconds() / 3600

    # Failures: earliest corrective order per pump.
    # Orders at t = 0 are skipped because we can't tell how long the pump had
    # been running before the data starts, so a time of zero would be wrong.
    cm = wo[wo["Order Type"].isin(FAILURE_ORDER_TYPES) & (wo["hours"] > 0)]
    failures = (
        cm.sort_values("hours")
        .groupby("Equipment", as_index=False)
        .first()[["Equipment", "hours"]]
    )
    failures["failed"] = 1

    # Suspensions: pumps with PM orders that never failed.
    # They were still running at the end of the window, so they are
    # right-censored at the full window length.
    pm_pumps = set(wo.loc[wo["Order Type"].isin(PM_ORDER_TYPES), "Equipment"])
    survived = pm_pumps - set(failures["Equipment"])
    suspensions = pd.DataFrame({"Equipment": sorted(survived), "hours": window_hours, "failed": 0})

    out = pd.concat([failures, suspensions], ignore_index=True)
    out = out.rename(columns={"Equipment": "unit_id"})
    # Sort by time, and put failures ahead of suspensions when times tie
    # (the usual convention for survival tables).
    return out.sort_values(["hours", "failed", "unit_id"], ascending=[True, False, True]).reset_index(drop=True)


def anonymize_units(failure_times):
    """Replace the equipment IDs with P01, P02, ... so no real IDs are stored.

    Pumps are numbered in time order (the order of build_failure_times, with
    ties broken by equipment ID), so the same raw export always gives the
    same labels. The mapping itself is not kept.
    """
    out = failure_times.sort_values(["hours", "failed", "unit_id"], ascending=[True, False, True])
    out = out.reset_index(drop=True)
    width = max(2, len(str(len(out))))
    out["unit_id"] = [f"P{i:0{width}d}" for i in range(1, len(out) + 1)]
    return out
