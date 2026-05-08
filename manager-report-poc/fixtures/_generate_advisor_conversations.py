"""Deterministic generator for advisor_conversations.csv.

Produces ~120 days of conversation-grain rows so the Platform view can
window by period (7/14/30/90d) and compute trend (current vs prior window).

Run:
    uv run python fixtures/_generate_advisor_conversations.py
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
NOW = datetime(2026, 5, 8, 17, 0, 0)
HISTORY_DAYS = 120

ISSUES = [
    "Repair delays",
    "Pricing",
    "Parts backlog",
    "Communication",
    "Scheduling",
    "Service quality",
]


# Per-advisor profile.
# vol_per_30d: target conversation count in any 30d window
# neg_rate_30d: long-run negative share
# resp_mean_s / resp_sd_s: response-time gaussian params (seconds)
# top_neg_issue: which issue dominates among their negatives
# recent_shift: "worse" -> recent 30d has higher neg rate, "better" -> lower, "stable" -> flat
ADVISORS = [
    {
        "name": "Sarah M",
        "vol_per_30d": 52,
        "neg_rate_30d": 0.19,
        "resp_mean_s": 1080,  # ~18m
        "resp_sd_s": 360,
        "top_neg_issue": "Repair delays",
        "recent_shift": "worse",
    },
    {
        "name": "John D",
        "vol_per_30d": 45,
        "neg_rate_30d": 0.05,
        "resp_mean_s": 360,  # ~6m
        "resp_sd_s": 120,
        "top_neg_issue": "Pricing",
        "recent_shift": "stable",
    },
    {
        "name": "Alex P",
        "vol_per_30d": 60,
        "neg_rate_30d": 0.11,
        "resp_mean_s": 660,  # ~11m
        "resp_sd_s": 240,
        "top_neg_issue": "Parts backlog",
        "recent_shift": "better",
    },
    {
        "name": "Riley N",
        "vol_per_30d": 34,
        "neg_rate_30d": 0.07,
        "resp_mean_s": 540,
        "resp_sd_s": 180,
        "top_neg_issue": "Communication",
        "recent_shift": "stable",
    },
    {
        "name": "Morgan L",
        "vol_per_30d": 42,
        "neg_rate_30d": 0.09,
        "resp_mean_s": 720,
        "resp_sd_s": 240,
        "top_neg_issue": "Scheduling",
        "recent_shift": "stable",
    },
]
# How much more likely the advisor's "top" issue is vs each other issue
# among their negative conversations. Higher = mode is more reliable.
TOP_ISSUE_WEIGHT = 6.0


def neg_rate_for(advisor: dict, days_ago: float) -> float:
    base = advisor["neg_rate_30d"]
    shift = advisor["recent_shift"]
    if shift == "worse":
        # last 30d significantly higher; older 30d slightly lower so avg ≈ base
        return base * 1.6 if days_ago < 30 else base * 0.7
    if shift == "better":
        return base * 0.55 if days_ago < 30 else base * 1.4
    return base


def main() -> None:
    rng = random.Random(SEED)
    rows: list[dict] = []

    for advisor in ADVISORS:
        n_total = round(advisor["vol_per_30d"] / 30 * HISTORY_DAYS)
        for _ in range(n_total):
            seconds_ago = rng.randint(0, HISTORY_DAYS * 86400)
            started_at = NOW - timedelta(seconds=seconds_ago)
            days_ago = seconds_ago / 86400

            is_negative = rng.random() < neg_rate_for(advisor, days_ago)
            if is_negative:
                sentiment = "negative"
            else:
                sentiment = rng.choices(["positive", "neutral"], weights=[0.65, 0.35])[0]

            # negatives more often unresolved; non-negatives almost always resolved
            if is_negative:
                resolved = rng.random() > 0.45
            else:
                resolved = rng.random() > 0.05

            response_time = max(
                30, int(rng.gauss(advisor["resp_mean_s"], advisor["resp_sd_s"]))
            )

            if is_negative:
                weights = [
                    TOP_ISSUE_WEIGHT if iss == advisor["top_neg_issue"] else 1.0
                    for iss in ISSUES
                ]
                issue = rng.choices(ISSUES, weights=weights)[0]
            else:
                issue = rng.choice(ISSUES)

            rows.append(
                {
                    "conversation_id": f"c_{len(rows):05d}",
                    "advisor_name": advisor["name"],
                    "started_at": started_at.isoformat(timespec="seconds"),
                    "sentiment": sentiment,
                    "resolved": "true" if resolved else "false",
                    "response_time_seconds": response_time,
                    "issue_category": issue,
                }
            )

    rows.sort(key=lambda r: r["started_at"], reverse=True)

    out = Path(__file__).parent / "advisor_conversations.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
