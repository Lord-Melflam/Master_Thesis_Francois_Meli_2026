"""12 - Per-student exam grade -> balanced score categories (Kim's request).

Exam grade per student = mean of the FINAL true score over the 6 questions
(unattempted question = 0). We compare Kim's fixed bins with balanced (quartile) bins,
and report the 'exactly 100' special-class size. Score categories are a DESCRIPTOR to
cross-tab against behaviour clusters (we cluster WITHOUT score).

Outputs -> data/v2/res_python/clustering/exam_score_categories_vN.csv
Figure  -> plots/fig_exam_grade_distribution_vN.{pdf,png}
"""
import csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import versioned_path

REPO = Path(__file__).resolve().parents[3]
# official re-grade: exam_clean_true (from 07) — its `score` column is the true score
TRUE_GLOB = str(REPO / "data/v2/res_python/true_scores/exam_clean_true_v*.csv")
OUT = REPO / "data/v2/res_python/clustering"
FIGS = REPO / "data/v2/res_python/plots"
N_Q = 6
FIXED = [(0, 24, "very_bad"), (25, 49, "bad"), (50, 74, "good"), (75, 99, "very_good"), (100, 100, "perfect")]


def latest(pat): return sorted(glob.glob(pat))[-1]


def student_grades():
    rows = list(csv.DictReader(open(latest(TRUE_GLOB))))
    last = defaultdict(int)
    for r in rows:
        last[(r["hash"], r["qname"])] = max(last[(r["hash"], r["qname"])], int(r["n_submission"]))
    final_true = {}
    for r in rows:
        if int(r["n_submission"]) == last[(r["hash"], r["qname"])]:
            final_true[(r["hash"], r["qname"])] = float(r["score"])   # score = official true score
    by_student = defaultdict(float)
    for (h, q), t in final_true.items():
        by_student[h] += t
    return {h: s / N_Q for h, s in by_student.items()}   # /6, unattempted=0


def main():
    g = student_grades()
    hashes = list(g); vals = np.array([g[h] for h in hashes])
    print(f"students: {len(vals)}  mean={vals.mean():.1f}  median={np.median(vals):.1f}")
    print("quantiles (0,25,50,75,100):", np.percentile(vals, [0, 25, 50, 75, 100]).round(1).tolist())
    n100 = int((vals >= 99.99).sum())
    print(f"exactly 100 (all six perfect): {n100} ({100*n100/len(vals):.1f}%)")

    print("\nKim's FIXED bins:")
    for lo, hi, name in FIXED:
        n = int(((vals >= lo) & (vals <= hi)).sum())
        print(f"  {name:10s} [{lo}-{hi}]: {n} ({100*n/len(vals):.0f}%)")

    # balanced (quartile) bins -> 4 ~equal groups
    q = np.percentile(vals, [25, 50, 75])
    print(f"\nBALANCED quartile boundaries: {q.round(1).tolist()}")
    bal_names = ["Q1_lowest", "Q2", "Q3", "Q4_highest"]
    def bal_cat(v):
        return bal_names[int(np.searchsorted(q, v, side="right"))]
    from collections import Counter
    bc = Counter(bal_cat(v) for v in vals)
    for nm in bal_names:
        print(f"  {nm:11s}: {bc[nm]} ({100*bc[nm]/len(vals):.0f}%)")

    # write assignment (both schemes)
    OUT.mkdir(parents=True, exist_ok=True)
    p = versioned_path(OUT, "exam_score_categories", "csv")
    def fixed_cat(v):
        for lo, hi, name in FIXED:
            if lo <= v <= hi: return name
        return "?"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["hash", "exam_grade", "fixed_category", "balanced_category"])
        for h in hashes:
            w.writerow([h, round(g[h], 2), fixed_cat(g[h]), bal_cat(g[h])])
    print(f"wrote {p}")

    # figure: grade distribution with both boundary schemes
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(vals, bins=np.arange(0, 102, 4), color="#0072B2", alpha=0.85)
    for b in [25, 50, 75]:
        ax.axvline(b, color="#D55E00", ls="--", lw=1)
    for b in q:
        ax.axvline(b, color="#009E73", ls=":", lw=1.5)
    ax.set_xlabel("exam grade per student (mean final true score over 6 questions)")
    ax.set_ylabel("students")
    ax.set_title("Exam grade distribution — fixed bins (orange --) vs balanced quartiles (green :)", fontsize=11)
    n = 1
    while (FIGS / f"fig_exam_grade_distribution_v{n}.pdf").exists() or (FIGS / f"fig_exam_grade_distribution_v{n}.png").exists():
        n += 1
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_exam_grade_distribution_v{n}.{ext}", bbox_inches="tight")
    plt.close(fig); print(f"fig fig_exam_grade_distribution_v{n}.pdf/.png")


if __name__ == "__main__":
    main()
