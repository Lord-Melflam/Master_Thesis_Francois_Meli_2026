"""05 - Tier 2: incremental DIFF / edit-size features (reads the .py submissions).

For each (student, question), submissions are ordered and consecutive pairs are diffed
(difflib) to measure how the code CHANGES from one attempt to the next, combined with
the score change. This is the 'incremental' signal the thesis is named for, and the
one that was built in v1 but dropped from the final features.

Per consecutive pair: edit_size = # changed lines; dscore = score change. Classified as
  tiny fix  (edit <= 2), big rewrite (edit >= 20),
  churn     (edit >= 5 AND dscore <= 0)  -> effort without progress,
  breakthrough (edit <= 3 AND dscore >= 50) -> the '0->100 in one line' case.
Aggregated per question, then per student by median (counts summed).

Usage: python3 05_build_diff_features.py [exam] [missions]   (default: both)

Outputs (versioned) -> data/v2/res_python/features/
  exam_diff_features_vN.csv / missions_diff_features_vN.csv / diff_features_summary_vN.txt
"""
import csv
import glob
import sys
import difflib
from pathlib import Path
from collections import defaultdict
import numpy as np

from common import versioned_path

REPO = Path(__file__).resolve().parents[3]
EXAM_CLEAN_GLOB = str(REPO / "data/v2/res_python/exam_audit/exam_clean_v*.csv")
EXAM_TRUE_GLOB = str(REPO / "data/v2/res_python/true_scores/exam_clean_true_v*.csv")  # true scores (07)
EXAM_FLAG_GLOB = str(REPO / "data/v2/res_python/exam_audit/exam_flagged_hashes_v*.csv")
EXAM_BASE = REPO / "data/v2/last_archive/2026.exam/2026.01_comment"
MISS_GLOB = str(REPO / "data/v2/last_archive/2025.Q1/mission_*_comment/*/data.csv")
OUT = REPO / "data/v2/res_python/features"

# Edit-size cutoffs are DATA-DERIVED from each dataset's edit-size distribution (not hardcoded):
#   small edit = <= median edit size ; large edit = >= 90th percentile.
# The score-gain cutoff is semantic and cited: a "big gain" is >= half the marks.
# See feedback_tfe/PARAMETERS.md.
SMALL_EDIT_PCTL, LARGE_EDIT_PCTL = 50, 90
BREAK_DSCORE = 50.0          # semantic: gain of >= half of the 0-100 marks
DIFF_COLS = ["median_edit_size", "tiny_fix_ratio", "big_rewrite_ratio",
             "churn_ratio", "breakthrough_ratio", "breakthrough_count", "n_edits"]


def latest(pat):
    fs = sorted(glob.glob(pat))
    if not fs:
        raise FileNotFoundError(pat)
    return fs[-1]


def read_lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.readlines()
    except FileNotFoundError:
        return None


def edit_size(a, b):
    """# changed lines between two line-lists (added + removed via opcodes)."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")


def per_question_diffs(recs):
    """recs: list of (n_submission, score, filepath) for one (student, question)."""
    recs = sorted(recs, key=lambda x: x[0])
    edits, dscores = [], []
    prev_lines, prev_score = None, None
    for _, score, path in recs:
        lines = read_lines(path)
        if lines is None:
            prev_lines, prev_score = None, None
            continue
        if prev_lines is not None:
            edits.append(edit_size(prev_lines, lines))
            dscores.append(score - prev_score)
        prev_lines, prev_score = lines, score
    return edits, dscores


def build(records, is_mission):
    by_sq = defaultdict(list)
    for h, q, n, sc, path in records:
        by_sq[(h, q)].append((n, sc, path))

    # pass 1: edits/dscores per (student, question) + collect all edit sizes for the dataset
    perq = {}
    all_edits = []
    for (h, q), recs in by_sq.items():
        edits, dscores = per_question_diffs(recs)
        if not edits:
            continue
        e = np.array(edits, dtype=float)
        perq[(h, q)] = (e, np.array(dscores, dtype=float))
        all_edits.append(e)
    all_e = np.concatenate(all_edits) if all_edits else np.array([0.0])
    small = float(np.percentile(all_e, SMALL_EDIT_PCTL))   # "small edit" cutoff (median)
    large = float(np.percentile(all_e, LARGE_EDIT_PCTL))   # "large edit" cutoff (p90)

    # pass 2: classify with the data-derived cutoffs, then aggregate per student by median
    per_student = defaultdict(list)
    for (h, q), (e, d) in perq.items():
        n = len(e)
        breaks = int(np.sum((e <= small) & (d >= BREAK_DSCORE)))
        per_student[h].append({
            "median_edit": float(np.median(e)),
            "tiny": float(np.mean(e <= small)),
            "big": float(np.mean(e >= large)),
            "churn": float(np.mean((e >= large) & (d <= 0))),
            "break_ratio": breaks / n,
            "breaks": breaks,
            "n": n,
        })

    out = []
    for h, qs in per_student.items():
        row = {"hash": h}
        row["median_edit_size"] = round(float(np.median([q["median_edit"] for q in qs])), 2)
        row["tiny_fix_ratio"] = round(float(np.median([q["tiny"] for q in qs])), 4)
        row["big_rewrite_ratio"] = round(float(np.median([q["big"] for q in qs])), 4)
        row["churn_ratio"] = round(float(np.median([q["churn"] for q in qs])), 4)
        row["breakthrough_ratio"] = round(float(np.median([q["break_ratio"] for q in qs])), 4)
        row["breakthrough_count"] = sum(q["breaks"] for q in qs)
        row["n_edits"] = sum(q["n"] for q in qs)
        out.append(row)
    return out, (small, large)


def exam_records():
    _t = sorted(glob.glob(EXAM_TRUE_GLOB))
    rows = list(csv.DictReader(open(_t[-1] if _t else latest(EXAM_CLEAN_GLOB))))
    recs = []
    for r in rows:
        path = EXAM_BASE / r["qname"] / "code" / r["file"]
        recs.append((r["hash"], r["qname"], int(r["n_submission"]), float(r["score"]), path))
    return recs


def mission_records():
    staff = {r["hash"] for r in csv.DictReader(open(latest(EXAM_FLAG_GLOB)))}
    recs = []
    for p in sorted(glob.glob(MISS_GLOB)):
        qdir = Path(p).parent
        mission = qdir.parent.name
        for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
            if r["hash"] in staff:
                continue
            path = qdir / "code" / r["file"]
            recs.append((r["hash"], mission + "/" + r["qname"], int(r["n_submission"]), float(r["score"]), path))
    return recs


def write_table(path, rows):
    cols = ["hash"] + DIFF_COLS
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    which = [a.lower() for a in sys.argv[1:]] or ["exam", "missions"]
    OUT.mkdir(parents=True, exist_ok=True)
    L = [f"DIFF FEATURES (Tier 2). Edit cutoffs DATA-DERIVED (small=p{SMALL_EDIT_PCTL}, "
         f"large=p{LARGE_EDIT_PCTL} of edit sizes); churn = large edit & no gain; "
         f"breakthrough = small edit & gain>= {BREAK_DSCORE:.0f} (half the marks)."]
    paths = []
    if "exam" in which:
        feats, thr = build(exam_records(), is_mission=False)
        p = versioned_path(OUT, "exam_diff_features", "csv")
        write_table(p, feats)
        paths.append(p)
        L.append(f"exam: {len(feats)} students -> {p.name}  [small<= {thr[0]:.0f}, large>= {thr[1]:.0f} lines]")
        L.append("  exam means: " + ", ".join(
            f"{c}={round(np.mean([r[c] for r in feats]),3)}" for c in
            ["median_edit_size", "tiny_fix_ratio", "churn_ratio", "breakthrough_ratio"]))
    if "missions" in which:
        feats, thr = build(mission_records(), is_mission=True)
        p = versioned_path(OUT, "missions_diff_features", "csv")
        write_table(p, feats)
        paths.append(p)
        L.append(f"missions: {len(feats)} students -> {p.name}  [small<= {thr[0]:.0f}, large>= {thr[1]:.0f} lines]")
        L.append("  missions means: " + ", ".join(
            f"{c}={round(np.mean([r[c] for r in feats]),3)}" for c in
            ["median_edit_size", "tiny_fix_ratio", "churn_ratio", "breakthrough_ratio"]))
    p_sum = versioned_path(OUT, "diff_features_summary", "txt")
    with open(p_sum, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote:", *[str(p) for p in paths], str(p_sum), sep="\n  ")


if __name__ == "__main__":
    main()
