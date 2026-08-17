"""07 - Certify the exam clean table as TRUE-scored.

As of 2026-08, the exam raw data (data/v2/last_archive/2026.exam/2026.01_comment) is a
FULL OFFICIAL INGInious re-grade of every submission: the `score` column is already the
true per-submission score (the old shown/cached scores are preserved under
2026.01_comment_v0). So we no longer recompute scores in a sandbox (script 06 is retired,
kept only for provenance — it validated the official data at 99.6% exact, q5 100%, q6 97%).

This script therefore just CERTIFIES the clean table as the true table: it copies
exam_clean -> exam_clean_true unchanged (score already true), stamping true_status so
downstream scripts (04/05/09/12/23/25/26/32), which read exam_clean_true, keep working
with no changes. `filename_score` is kept equal to `score` for schema compatibility.

Output (versioned) -> data/v2/res_python/true_scores/exam_clean_true_vN.csv
"""
import csv
import glob
from pathlib import Path
from common import versioned_path

REPO = Path(__file__).resolve().parents[3]
EXAM_CLEAN_GLOB = str(REPO / "data/v2/res_python/exam_audit/exam_clean_v*.csv")
OUT = REPO / "data/v2/res_python/true_scores"


def latest(pat):
    return sorted(glob.glob(pat))[-1]


def main():
    clean = list(csv.DictReader(open(latest(EXAM_CLEAN_GLOB))))
    cols = list(clean[0].keys())
    out_cols = cols + ["filename_score", "true_status"]

    p = versioned_path(OUT, "exam_clean_true", "csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols)
        w.writeheader()
        for r in clean:
            r["filename_score"] = r["score"]      # already the true (official) score
            r["true_status"] = "OFFICIAL_REGRADE"
            w.writerow(r)
    print(f"wrote {p}  ({len(clean)} rows; scores are the official INGInious re-grade)")


if __name__ == "__main__":
    main()
