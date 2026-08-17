"""10 - Merge per-student feature tables into one table per dataset.

Joins behaviour (04) + incremental-diff (05) + code-quality (09) on `hash`, using the
behaviour table as the base (all students). Produces one wide table per dataset.

Column roles (used by the clustering step 11):
  BEHAVIOUR (cluster ON): effort, rhythm, iteration-dynamics, code-quality.
  OUTCOME  (held ASIDE) : best/final score, solved rates, score gain, never-solved.

Outputs (versioned) -> data/v2/res_python/features/{exam,missions}_features_all_vN.csv
"""
import csv
import glob
from pathlib import Path
from common import versioned_path

REPO = Path(__file__).resolve().parents[3]
FEAT = REPO / "data/v2/res_python/features"

OUTCOME = ["median_best_score", "median_final_score", "median_score_gain",
           "solved_pass_count", "solved_pass_rate", "solved_full_count", "solved_full_rate",
           "never_solved_count"]


def latest(stem):
    return sorted(glob.glob(str(FEAT / f"{stem}_v*.csv")))[-1]


def load(stem):
    return {r["hash"]: r for r in csv.DictReader(open(latest(stem)))}


def merge(prefix):
    base = load(f"{prefix}_behaviour_features")      # all students
    diff = load(f"{prefix}_diff_features")
    cq = load(f"{prefix}_codequality_features")
    rows = []
    # column order: hash, behaviour(base minus outcome), diff, codequality, then outcome
    beh_cols = [c for c in next(iter(base.values())).keys() if c not in ("hash",) + tuple(OUTCOME)]
    diff_cols = [c for c in next(iter(diff.values())).keys() if c != "hash"] if diff else []
    cq_cols = [c for c in next(iter(cq.values())).keys() if c != "hash"] if cq else []
    cols = ["hash"] + beh_cols + diff_cols + cq_cols + OUTCOME
    for h, b in base.items():
        r = {"hash": h}
        for c in beh_cols:
            r[c] = b.get(c, "")
        for c in diff_cols:
            r[c] = diff.get(h, {}).get(c, "")
        for c in cq_cols:
            r[c] = cq.get(h, {}).get(c, "")
        for c in OUTCOME:
            r[c] = b.get(c, "")
        rows.append(r)
    p = versioned_path(FEAT, f"{prefix}_features_all", "csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    missing_diff = sum(1 for h in base if h not in diff)
    print(f"{prefix}: {len(rows)} students, {len(cols)-1} features -> {p.name}  ({missing_diff} without diff features)")
    return cols


for prefix in ("exam", "missions"):
    cols = merge(prefix)
print("\nOUTCOME columns held aside for clustering:", ", ".join(OUTCOME))
