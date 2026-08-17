"""16 - Concrete artifacts for the supervisor meeting.

(A) fig_linkage_dendrograms_compare : the four linkages side by side (ward/complete/
    average/single) on the SAME exam behaviour data (569 linked students, 12 behaviour
    features, log+z-scaled) — makes the "why Ward" argument VISIBLE (single/average/
    complete peel off singletons; only Ward gives a balanced tree).
    Granularity: one point per STUDENT; whole exam (behaviour aggregated per student).

(B) fig_exam_solve_per_question : granular per-QUESTION full-solve rate, shown vs true
    score, q1..q6. Granularity: per exam question. Formula: among students who attempted
    question q, share whose FINAL submission scores 100 (true) vs 100 (shown INGInious).
    (Granular view to present BEFORE the whole-exam aggregate.)

Outputs -> data/v2/res_python/plots/  (versioned PDF+PNG)
"""
import csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
FEAT = REPO / "data/v2/res_python/features"
LINK = REPO / "data/v2/res_python/linkage"
TRUE = REPO / "data/v2/res_python/true_scores"
FIGS = REPO / "data/v2/res_python/plots"

BEHAV = ["questions_attempted", "median_attempts", "median_fast_retry_ratio",
         "median_long_pause_ratio", "median_mean_delta_sec", "median_improving_ratio",
         "median_edit_size", "churn_ratio", "breakthrough_ratio",
         "median_nloc", "median_comment_ratio", "median_n_concepts"]
LOG = {"median_attempts", "median_mean_delta_sec", "median_edit_size", "median_nloc"}
GREY, BLUE, GREEN = "#8C8C8C", "#0072B2", "#009E73"


def latest(pat): return sorted(glob.glob(str(pat)))[-1]


def savefig(fig, stem):
    FIGS.mkdir(parents=True, exist_ok=True)
    n = 1
    while (FIGS / f"{stem}_v{n}.pdf").exists() or (FIGS / f"{stem}_v{n}.png").exists():
        n += 1
    for e in ("pdf", "png"):
        fig.savefig(FIGS / f"{stem}_v{n}.{e}", bbox_inches="tight")
    plt.close(fig); print(f"  saved {stem}_v{n}.pdf/.png")


def linkage_compare():
    linked = {r["hash"] for r in csv.DictReader(open(latest(LINK / "linked_students_v*.csv")))}
    feats = {r["hash"]: r for r in csv.DictReader(open(latest(FEAT / "exam_features_all_v*.csv")))}
    hashes = [h for h in feats if h in linked]

    def val(h, c):
        try: return float(feats[h][c])
        except: return np.nan
    X = np.nan_to_num(np.array([[val(h, c) for c in BEHAV] for h in hashes], float), nan=0.0)
    for j, c in enumerate(BEHAV):
        if c in LOG: X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    Xs = StandardScaler().fit_transform(X)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, lk in zip(axes.ravel(), ["ward", "complete", "average", "single"]):
        Z = linkage(Xs, method=lk)
        dendrogram(Z, no_labels=True, color_threshold=0, ax=ax)
        for coll in ax.collections:
            coll.set_linewidth(0.4); coll.set_color("#333333")
        ax.set_title(f"{lk} linkage", fontsize=11)
        ax.set_yticks([])
    fig.suptitle(f"Exam behaviour — hierarchical tree by linkage (n={len(hashes)} students)\n"
                 "single/average/complete peel off individual points; only Ward splits the cohort into balanced groups",
                 fontsize=12, y=1.02)
    savefig(fig, "fig_linkage_dendrograms_compare")


def _finals(rows):
    last = defaultdict(int)
    for r in rows:
        last[(r["hash"], r["qname"])] = max(last[(r["hash"], r["qname"])], int(r["n_submission"]))
    return {(r["hash"], r["qname"]): r for r in rows
            if int(r["n_submission"]) == last[(r["hash"], r["qname"])]}


def _true(r):
    return float(r["true_score"]) if r["status"] == "OK" and r["true_score"] != "" else 0.0


def solve_per_question():
    """Granular TRUE difficulty per question (finals): pass (>=50) vs full (==100)."""
    rows = list(csv.DictReader(open(latest(TRUE / "true_scores_all_v*.csv"))))
    fin = _finals(rows)
    natt = defaultdict(int); npass = defaultdict(int); nfull = defaultdict(int)
    for (h, q), r in fin.items():
        t = _true(r); natt[q] += 1
        npass[q] += (t >= 50); nfull[q] += (t == 100)
    qs = sorted(natt)
    passr = [npass[q] / natt[q] for q in qs]; fullr = [nfull[q] / natt[q] for q in qs]
    x = np.arange(len(qs)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.4))
    b1 = ax.bar(x - w / 2, passr, w, color=BLUE, label="passed (true ≥ 50)")
    b2 = ax.bar(x + w / 2, fullr, w, color=GREEN, label="fully solved (true = 100)")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.0%}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels([q + f"\n(n={natt[q]})" for q in qs])
    ax.set_ylim(0, 1); ax.set_ylabel("share of students (final submission, true score)")
    ax.set_title("True difficulty per exam question (granular; finals)", fontsize=12)
    ax.legend(frameon=False)
    savefig(fig, "fig_exam_solve_per_question")


def inflation_per_question():
    """Granular per-question finding of the true-score pipeline: among INTERMEDIATE
    submissions, share whose shown INGInious score OVERSTATED the true score."""
    rows = list(csv.DictReader(open(latest(TRUE / "true_scores_all_v*.csv"))))
    last = defaultdict(int)
    for r in rows:
        last[(r["hash"], r["qname"])] = max(last[(r["hash"], r["qname"])], int(r["n_submission"]))
    infl = defaultdict(int); n = defaultdict(int)
    for r in rows:
        if int(r["n_submission"]) == last[(r["hash"], r["qname"])]:
            continue  # intermediates only
        q = r["qname"]; n[q] += 1
        if float(r["filename_score"]) - _true(r) > 1e-6:
            infl[q] += 1
    qs = sorted(n)
    rate = [infl[q] / n[q] for q in qs]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(range(len(qs)), rate, color="#E69F00")
    for b in bars:
        ax.annotate(f"{b.get_height():.0%}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(qs))); ax.set_xticklabels([q + f"\n(n={n[q]})" for q in qs])
    ax.set_ylim(0, 1); ax.set_ylabel("share of intermediate submissions (per question)")
    ax.set_title("Per question: intermediate submissions whose shown score overstated the true score", fontsize=12)
    savefig(fig, "fig_exam_inflation_per_question")


if __name__ == "__main__":
    print("meeting figures ->", FIGS)
    solve_per_question()      # granular true difficulty per question
    inflation_per_question()  # granular per-question inflation (true-score finding)
    print("done. (linkage_compare already produced v1)")
