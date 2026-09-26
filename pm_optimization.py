"""Shared code for the pump preventive-maintenance optimization project.

The notebooks import this module so that paths, model constants, the
data-preparation logic, the Weibull fitting and the PM-interval optimization
model live in one place and can be unit tested.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
WEIBULL_PARAMS = DATA_DIR / "weibull_params.json"

# Model constants
HOURS_PER_MONTH = 30.42 * 24  # average month length (30.42 days) in hours
# Costs are in relative units: one PM job = 1, and a corrective job costs
# about 126 times as much. Reported annual costs are an index (current plan = 100).
PM_COST = 1.0                 # cost of one PM job
CM_COST = 125.6611491829204   # cost of one corrective job, relative to a PM job
CURRENT_PM_INTERVAL = 6.0     # months, the current plan
BOUNDS = (1.0, 12.0)          # allowed PM intervals (months)

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


# Weibull analysis
def median_rank_table(hours, failed):
    """Build the median-rank table: rank, reversed rank, Johnson's
    adjusted rank, Bernard's median rank, ln(t) and ln(ln(1/(1-F))). Returns failures only.
    """
    table = pd.DataFrame({"hours": np.asarray(hours, float), "failed": np.asarray(failed, int)})
    # Rank all pumps by time. On ties, failures come before suspensions, the
    # same convention as build_failure_times.
    table = table.sort_values(["hours", "failed"], ascending=[True, False]).reset_index(drop=True)
    n = len(table)
    table["rank"] = np.arange(1, n + 1)
    table["reversed_rank"] = n - table["rank"] + 1

    # Johnson's adjusted rank: each suspension spreads its share of the
    # remaining ranks over the later failures, so a failure's rank is bumped up
    # by the suspensions before it. Suspensions get no rank of their own.
    adj_ranks, prev = [], 0.0
    for rr, f in zip(table["reversed_rank"], table["failed"]):
        if f:
            prev = prev + (n + 1 - prev) / (1 + rr)
            adj_ranks.append(prev)
        else:
            adj_ranks.append(np.nan)
    table["adj_rank"] = adj_ranks

    # Only failures are plotted: estimate F with Bernard's approximation, then
    # take the Weibull plot coordinates x = ln(t), y = ln(ln(1/(1-F))).
    fails = table[table["failed"] == 1].copy()
    fails["median_rank"] = (fails["adj_rank"] - 0.3) / (n + 0.4)
    fails["ln_t"] = np.log(fails["hours"])
    fails["ln_ln"] = np.log(-np.log(1 - fails["median_rank"]))
    return fails.reset_index(drop=True)


def fit_weibull_mrr(hours, failed):
    """Fit a 2-parameter Weibull by median-rank regression (regression of y on x)."""
    table = median_rank_table(hours, failed)
    # On the Weibull plot the line is y = beta * x - beta * ln(eta), so the
    # slope is beta and eta comes back from the intercept.
    beta, intercept = np.polyfit(table["ln_t"], table["ln_ln"], 1)
    r2 = np.corrcoef(table["ln_t"], table["ln_ln"])[0, 1] ** 2
    return {
        "beta": float(beta),
        "eta_hours": float(np.exp(-intercept / beta)),
        "intercept": float(intercept),
        "r2": float(r2),
        "n_units": int(len(hours)),
        "n_failures": int(np.sum(failed)),
    }


def reliability(t, beta, eta):
    """R(t) = exp(-(t/eta)^beta). t and eta must be in the same unit."""
    return np.exp(-((np.asarray(t, float) / eta) ** beta))


def failure_prob(t, beta, eta):
    """F(t) = 1 - R(t): probability of failing by time t."""
    return 1 - reliability(t, beta, eta)


def save_params(params, path=WEIBULL_PARAMS):
    """Save the fitted Weibull parameters (adds eta in months) for later notebooks."""
    params = dict(params)
    params["eta_months"] = params["eta_hours"] / HOURS_PER_MONTH
    path.write_text(json.dumps(params, indent=2))


def load_params(path=WEIBULL_PARAMS):
    """Load the Weibull parameters saved by notebook 02."""
    return json.loads(path.read_text())


# Optimization model
@dataclass(frozen=True)
class PMModel:
    """The PM-interval model. Times are in months, costs in relative units.

    Objective 1:          C(x) = (12/x) * C_PM * n  +  F(x) * C_CM * n
                          (reported as an index, current plan = 100)
    Objective 2:          F(x) = 1 - exp(-(x/eta)^beta)
    """

    beta: float
    eta_months: float
    n_pumps: int
    pm_cost: float = PM_COST
    cm_cost: float = CM_COST

    def failure_prob(self, x):
        """Probability that a pump fails within a PM interval of x months."""
        return failure_prob(x, self.beta, self.eta_months)

    def reliability(self, x):
        """Probability that a pump survives a PM interval of x months."""
        return reliability(x, self.beta, self.eta_months)

    def annual_cost(self, x):
        """Annual PM cost plus expected corrective cost, in relative units (one PM job = 1)."""
        x = np.asarray(x, float)
        # 12 / x PM rounds a year, each one servicing every pump
        preventive = (12 / x) * self.pm_cost * self.n_pumps
        # Expected number of failures per pump in one interval, times the repair cost
        corrective = self.failure_prob(x) * self.cm_cost * self.n_pumps
        return preventive + corrective

    def cost_index(self, x, base=CURRENT_PM_INTERVAL):
        """Annual cost as an index: the plan with interval `base` (the current plan) = 100."""
        return self.annual_cost(x) / self.annual_cost(base) * 100

    def objectives(self, x):
        """Both objectives for each interval in x: columns [annual cost index, failure probability]."""
        return np.column_stack([self.cost_index(x), self.failure_prob(x)])

    @classmethod
    def from_params(cls, params):
        """Build the model from the parameters saved by notebook 02.

        The pump count is the number of pumps in the fitted data (n_units).
        """
        return cls(beta=params["beta"], eta_months=params["eta_months"],
                   n_pumps=params["n_units"])


def pareto_mask(F):
    """True for each row of F that no other row dominates (all objectives minimized)."""
    F = np.asarray(F)
    mask = np.ones(len(F), dtype=bool)
    # Row i dominates row j if it is no worse in every objective and strictly
    # better in at least one. Rows already known to be dominated are skipped,
    # since anything they dominate is also dominated by whatever beat them.
    for i in range(len(F)):
        if mask[i]:
            dominated = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
            mask[dominated] = False
    return mask


def brute_force_front(model, n_points=20_000, bounds=BOUNDS):
    """Exact Pareto front by evaluating a dense grid of intervals (possible with one variable)."""
    x = np.linspace(bounds[0], bounds[1], n_points)
    F = model.objectives(x)
    keep = pareto_mask(F)
    return x[keep], F[keep]


def interval_label(months):
    """Format a PM interval for display: 3.5 -> '3M 2W', 6 -> '6M'."""
    whole = int(months)
    return f"{whole}M" + (" 2W" if months - whole >= 0.5 else "")


def practical_plans(model, step=0.5, bounds=BOUNDS):
    """Evaluate schedulable intervals (half-month steps) and flag the Pareto-optimal ones."""
    # The small epsilon makes the upper bound (e.g. 12 months) part of the range
    x = np.arange(bounds[0], bounds[1] + 1e-9, step)
    F = model.objectives(x)
    return pd.DataFrame({
        "interval_months": x,
        "label": [interval_label(v) for v in x],
        "annual_cost_index": F[:, 0],
        "failure_prob_pct": F[:, 1] * 100,
        "reliability_pct": (1 - F[:, 1]) * 100,
        "pareto_optimal": pareto_mask(F),
    })


def compare_with_current(model, current_x=CURRENT_PM_INTERVAL):
    """Current plan vs the recommended plan (lowest-cost Pareto-optimal half-month plan)."""
    plans = practical_plans(model)
    cur = plans.set_index("interval_months").loc[current_x]
    sel = plans.loc[plans["annual_cost_index"].idxmin()]
    table = pd.DataFrame({
        f"Current ({cur['label']})": [cur["annual_cost_index"], cur["reliability_pct"]],
        f"Recommended ({sel['label']})": [sel["annual_cost_index"], sel["reliability_pct"]],
    }, index=["Annual cost (index, current plan = 100)", "Reliability over interval (%)"])
    table["Change"] = table.iloc[:, 1] - table.iloc[:, 0]
    return table
