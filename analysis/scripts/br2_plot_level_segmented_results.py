#!/usr/bin/env python3
"""
Create level-segmented visualizations from BR2 active form results.

Input:
- weekly_meetings/Quadrimester2/br2/res/derived/by_level/all_active_combined.csv

Outputs (PNG + PDF):
- weekly_meetings/Quadrimester2/br2/res/derived/by_level/plots/
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
IN_CSV = (
    ROOT
    / "weekly_meetings"
    / "Quadrimester2"
    / "br2"
    / "res"
    / "derived"
    / "by_level"
    / "all_active_combined.csv"
)
OUT_DIR = IN_CSV.parent / "plots"


LEVEL_ORDER = [
    "bachelor_1",
    "bachelor_2",
    "bachelor_3",
    "master",
    "secondary",
    "other_unknown",
]


def parse_float(v: str) -> float | None:
    v = (v or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def save_fig(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with IN_CSV.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    level_counts = Counter((r.get("level_group") or "").strip() for r in rows)
    lang_by_level = defaultdict(Counter)
    score_by_level = defaultdict(list)
    difficulty_by_level = defaultdict(list)
    correct_by_level = defaultdict(list)
    other_by_level = defaultdict(list)

    for r in rows:
        lvl = (r.get("level_group") or "").strip()
        lang = (r.get("source_language") or "").strip() or "NA"
        lang_by_level[lvl][lang] += 1

        s = parse_float(r.get("d_score_pct", ""))
        if s is not None:
            score_by_level[lvl].append(s)

        d = parse_float(r.get("d_avg_difficulty", ""))
        if d is not None:
            difficulty_by_level[lvl].append(d)

        c = parse_float(r.get("d_correct_count", ""))
        if c is not None:
            correct_by_level[lvl].append(c)

        o = parse_float(r.get("grid_other_count", ""))
        if o is not None:
            other_by_level[lvl].append(o)

    levels = [l for l in LEVEL_ORDER if level_counts.get(l, 0) > 0]

    # 1) Pie chart: level distribution
    fig, ax = plt.subplots(figsize=(7, 7))
    counts = [level_counts[l] for l in levels]
    ax.pie(counts, labels=levels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Active responses by student level")
    save_fig(fig, "pie_active_responses_by_level")

    # 2) Stacked bar: FR/EN by level
    fig, ax = plt.subplots(figsize=(9, 5))
    fr_counts = [lang_by_level[l].get("FR", 0) for l in levels]
    en_counts = [lang_by_level[l].get("EN", 0) for l in levels]
    x = range(len(levels))
    ax.bar(x, fr_counts, label="FR")
    ax.bar(x, en_counts, bottom=fr_counts, label="EN")
    ax.set_xticks(list(x))
    ax.set_xticklabels(levels, rotation=20, ha="right")
    ax.set_ylabel("Number of active responses")
    ax.set_title("Language composition by student level")
    ax.legend()
    save_fig(fig, "stacked_language_by_level")

    # 3) Histogram of Section D score distribution by level (overlay)
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = list(range(0, 110, 10))
    for lvl in levels:
        vals = score_by_level.get(lvl, [])
        if vals:
            ax.hist(vals, bins=bins, alpha=0.45, label=lvl)
    ax.set_xlabel("Section D score (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Section D score distribution by level")
    ax.legend(fontsize=8)
    save_fig(fig, "hist_score_pct_by_level")

    # 4) Boxplot: difficulty by level
    fig, ax = plt.subplots(figsize=(9, 5))
    box_data = [difficulty_by_level[l] for l in levels if difficulty_by_level.get(l)]
    box_labels = [l for l in levels if difficulty_by_level.get(l)]
    ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    ax.set_ylabel("Average perceived difficulty")
    ax.set_title("Perceived difficulty by level")
    plt.xticks(rotation=20, ha="right")
    save_fig(fig, "box_difficulty_by_level")

    # 5) Mean indicators by level (score, correct_count, other_count)
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_score = []
    mean_correct = []
    mean_other = []
    for l in levels:
        s = score_by_level.get(l, [])
        c = correct_by_level.get(l, [])
        o = other_by_level.get(l, [])
        mean_score.append(sum(s) / len(s) if s else 0.0)
        mean_correct.append(sum(c) / len(c) if c else 0.0)
        mean_other.append(sum(o) / len(o) if o else 0.0)
    x = range(len(levels))
    w = 0.25
    ax.bar([i - w for i in x], mean_score, width=w, label="Mean score %")
    ax.bar(x, mean_correct, width=w, label="Mean correct count (0-6)")
    ax.bar([i + w for i in x], mean_other, width=w, label="Mean Other count (0-6)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(levels, rotation=20, ha="right")
    ax.set_title("Level-wise mean indicators")
    ax.legend(fontsize=8)
    save_fig(fig, "bar_mean_indicators_by_level")

    print("Generated plot files in:", OUT_DIR)


if __name__ == "__main__":
    main()

