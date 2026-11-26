"""RFM segmentation with action playbooks (TR-AN-01).

Scores are percentile ranks mapped to 1..5. Tied customers share the lowest score of
their tie block (``rank(method="min")``). For example, every one-time buyer gets F=1
whatever the share of one-time buyers, which raw ``qcut`` cannot guarantee. Segments come
from a complete partition of the R x F grid. Monetary value breaks ties in prioritization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bizpulse.analytics.data import as_of as data_as_of

SEGMENTS: tuple[str, ...] = (
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "New Customers",
    "At Risk",
    "Hibernating/Lost",
)


@dataclass(frozen=True)
class Playbook:
    priority: int
    goal: str
    actions: tuple[str, ...]
    success_metric: str


PLAYBOOKS: dict[str, Playbook] = {
    "Champions": Playbook(
        1,
        "Protect and amplify your best customers",
        (
            "Invite to a VIP / early-access program; never discount by default",
            "Ask for reviews, referrals and case studies",
            "Offer premium bundles, upgrades or higher tiers",
        ),
        "Retention rate and referral-sourced revenue",
    ),
    "Loyal Customers": Playbook(
        2,
        "Grow share of wallet",
        (
            "Cross-sell adjacent categories based on purchase history",
            "Points / tiered loyalty rewards that unlock at the next spend level",
            "Personalized replenishment or renewal reminders",
        ),
        "Revenue per customer and categories per customer",
    ),
    "Potential Loyalists": Playbook(
        3,
        "Convert recent buyers into habitual buyers",
        (
            "Second-purchase incentive with a time limit",
            "Onboarding / education sequence showing product value",
            "Recommend best-sellers that drive repeat purchases",
        ),
        "Second-order conversion rate within 60 days",
    ),
    "New Customers": Playbook(
        4,
        "Deliver a great first experience",
        (
            "Welcome series with usage tips and expectations",
            "Follow-up on delivery / activation; fix friction fast",
            "Small, targeted incentive for the next purchase",
        ),
        "30-day repeat rate and first-order satisfaction",
    ),
    "At Risk": Playbook(
        5,
        "Win back previously valuable customers before they churn",
        (
            "Personal outreach or account review for high-monetary customers",
            "Reactivation offer tied to their favourite category",
            "Survey: why did you stop buying?",
        ),
        "Reactivation rate and recovered revenue",
    ),
    "Hibernating/Lost": Playbook(
        6,
        "Re-engage cheaply or let go",
        (
            "Low-cost automated reactivation campaign",
            "Suppress from paid retargeting to save budget",
            "Sunset after no response; keep for seasonal campaigns",
        ),
        "Cost per reactivated customer",
    ),
}


def segment_for(r_score: int, f_score: int) -> str:
    """Map an (R, F) score pair to exactly one segment (complete partition of 5x5)."""
    if r_score >= 4 and f_score >= 4:
        return "Champions"
    if r_score >= 3 and f_score >= 3:
        return "Loyal Customers"
    if r_score >= 4 and f_score == 1:
        return "New Customers"
    if r_score >= 3:
        return "Potential Loyalists"
    if f_score >= 3:
        return "At Risk"
    return "Hibernating/Lost"


def percentile_score(values: pd.Series[float], *, higher_is_better: bool = True) -> pd.Series[int]:
    """1..5 score from the tie-aware percentile rank of *values*."""
    ranked = values.rank(method="min", ascending=higher_is_better)
    raw = np.floor(5 * (ranked.to_numpy(dtype="float64") - 1) / len(values)).astype("int64") + 1
    return pd.Series(np.clip(raw, 1, 5), index=values.index, dtype="int64")


def rfm_table(tx: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per identified customer with R/F/M values, scores and segment."""
    reference = as_of if as_of is not None else data_as_of(tx)
    rows = tx.loc[tx["status"].ne("cancelled") & ~tx["is_anonymous"]]
    if rows.empty:
        return pd.DataFrame(
            columns=["customer_id", "recency_days", "frequency", "monetary", "segment"]
        )
    completed_net = rows["net"].where(rows["status"].eq("completed"), 0.0)
    grouped = rows.assign(completed_net=completed_net).groupby("customer_id")
    table = pd.DataFrame(
        {
            "first_order": grouped["order_ts"].min(),
            "last_order": grouped["order_ts"].max(),
            "frequency": grouped["order_id"].nunique().astype("int64"),
            "monetary": grouped["completed_net"].sum().astype("float64"),
        }
    ).sort_index()
    table["recency_days"] = (reference - table["last_order"].dt.normalize()).dt.days.astype("int64")
    table["r_score"] = percentile_score(
        table["recency_days"].astype("float64"), higher_is_better=False
    )
    table["f_score"] = percentile_score(table["frequency"].astype("float64"))
    table["m_score"] = percentile_score(table["monetary"])
    table["rfm"] = (
        table["r_score"].astype(str) + table["f_score"].astype(str) + table["m_score"].astype(str)
    )
    table["segment"] = [
        segment_for(int(r), int(f)) for r, f in zip(table["r_score"], table["f_score"], strict=True)
    ]
    return table.reset_index()


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Per-segment size, value and playbook, ordered by playbook priority."""
    total_customers = len(rfm)
    total_value = float(rfm["monetary"].sum()) if total_customers else 0.0
    rows = []
    for name in SEGMENTS:
        part = rfm.loc[rfm["segment"] == name]
        playbook = PLAYBOOKS[name]
        value = float(part["monetary"].sum())
        rows.append(
            {
                "segment": name,
                "customers": len(part),
                "share_customers": len(part) / total_customers if total_customers else 0.0,
                "net_revenue": value,
                "share_revenue": value / total_value if total_value else 0.0,
                "avg_recency_days": float(part["recency_days"].mean()) if len(part) else None,
                "avg_frequency": float(part["frequency"].mean()) if len(part) else None,
                "avg_monetary": float(part["monetary"].mean()) if len(part) else None,
                "priority": playbook.priority,
                "goal": playbook.goal,
                "actions": " | ".join(playbook.actions),
                "success_metric": playbook.success_metric,
            }
        )
    return pd.DataFrame(rows)
