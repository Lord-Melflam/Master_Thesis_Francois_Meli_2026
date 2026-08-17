"""15 - Significance of the year-engagement -> exam-outcome link.

(1) Mann-Whitney U on EXAM grade between the two year-behaviour groups (from 14),
    with effect size (Cliff's delta) — does the ~13-pt gap hold up?
(2) Robustness (less circular than testing by cluster): Spearman correlation between a
    DIRECT year-engagement measure (active weeks, questions attempted) and exam grade.

Prints results + writes a summary (versioned) -> data/v2/res_python/clustering/
"""
import csv, glob
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from common import versioned_path

REPO = Path(__file__).resolve().parents[3]
CLU = REPO / "data/v2/res_python/clustering"
FEAT = REPO / "data/v2/res_python/features"


def latest(stem, base): return sorted(glob.glob(str(base / f"{stem}_v*.csv")))[-1]


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((a[:, None] > b[None, :]).sum(1))
    lt = sum((a[:, None] < b[None, :]).sum(1))
    return (gt - lt) / (len(a) * len(b))


def main():
    asg = list(csv.DictReader(open(latest("missions_cluster_assignments", CLU))))
    groups = {}
    for r in asg:
        groups.setdefault(r["group"], []).append(float(r["exam_grade"]))
    # 2 groups expected; label by mean grade
    gk = sorted(groups, key=lambda g: -np.mean(groups[g]))
    hi, lo = groups[gk[0]], groups[gk[1]]   # hi = higher-mean-grade (more-engaged) group
    U, p = mannwhitneyu(hi, lo, alternative="two-sided")
    d = cliffs_delta(hi, lo)
    L = ["(1) Mann-Whitney U — EXAM grade, more-engaged vs less-engaged year-behaviour group",
         f"    more-engaged group:  n={len(hi)}  median={np.median(hi):.1f}  mean={np.mean(hi):.1f}",
         f"    less-engaged group:  n={len(lo)}  median={np.median(lo):.1f}  mean={np.mean(lo):.1f}",
         f"    U={U:.0f}  p={p:.2e}  Cliff's delta={d:.2f}  ({'small' if abs(d)<0.33 else 'medium' if abs(d)<0.47 else 'large'} effect)"]

    # (2) direct engagement -> exam grade (Spearman), less circular
    feats = {r["hash"]: r for r in csv.DictReader(open(latest("missions_features_all", FEAT)))}
    hashes = [r["hash"] for r in asg if r["hash"] in feats]
    grade = np.array([float(next(x for x in asg if x["hash"] == h)["exam_grade"]) for h in hashes])
    L.append("\n(2) Spearman correlation — direct year-engagement measure vs EXAM grade (n=%d)" % len(hashes))
    for feat in ["active_weeks", "active_days", "questions_attempted", "total_attempts"]:
        try:
            x = np.array([float(feats[h][feat]) for h in hashes])
        except Exception:
            continue
        rho, pp = spearmanr(x, grade)
        L.append(f"    {feat:20s}: rho={rho:+.2f}  p={pp:.2e}")

    L.append("\nInterpretation: p tells us the association is unlikely to be chance; Cliff's delta / rho"
             "\ntell us it is real but MODEST — a tendency, not a predictor (large individual overlap).")
    out = "\n".join(L)
    print(out)
    p_ = versioned_path(CLU, "significance_engagement_vs_exam", "txt")
    Path(p_).write_text(out + "\n"); print(f"\nwrote {p_.name}")


if __name__ == "__main__":
    main()
