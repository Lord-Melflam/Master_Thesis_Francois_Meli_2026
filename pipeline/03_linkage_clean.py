"""03 - Longitudinal linkage on CLEAN data.

Which students appear in BOTH the Q1 missions (learning process) and the Jan-2026 exam
(outcome), after removing the 7 staff/test hashes. This defines the cohort for any
process -> outcome analysis.

Reads:
  data/v2/res_python/exam_audit/exam_clean_v*.csv        (staff already removed)
  data/v2/last_archive/2025.Q1/mission_*_comment/*/data.csv
  data/v2/res_python/exam_audit/exam_flagged_hashes_v*.csv  (the 7 staff hashes)
Writes -> data/v2/res_python/linkage/
  linked_students_vN.csv     hashes present in both (the process->outcome cohort)
  linkage_summary_vN.txt
"""
import csv
import glob
from pathlib import Path

from common import versioned_path

REPO = Path(__file__).resolve().parents[3]
EXAM_CLEAN_GLOB = str(REPO / "data/v2/res_python/exam_audit/exam_clean_v*.csv")
EXAM_FLAG_GLOB = str(REPO / "data/v2/res_python/exam_audit/exam_flagged_hashes_v*.csv")
MISS_GLOB = str(REPO / "data/v2/last_archive/2025.Q1/mission_*_comment/*/data.csv")
OUT = REPO / "data/v2/res_python/linkage"


def latest(pat):
    fs = sorted(glob.glob(pat))
    if not fs:
        raise FileNotFoundError(pat)
    return fs[-1]


def hashes_from_csv(path, col="hash"):
    with open(path, newline="", encoding="utf-8") as fh:
        return {r[col] for r in csv.DictReader(fh)}


def main():
    staff = hashes_from_csv(latest(EXAM_FLAG_GLOB))
    exam = hashes_from_csv(latest(EXAM_CLEAN_GLOB))   # already staff-free

    missions = set()
    for p in sorted(glob.glob(MISS_GLOB)):
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                missions.add(r["hash"])
    missions -= staff   # apply the same staff removal to missions

    both = exam & missions
    exam_only = exam - missions
    miss_only = missions - exam

    OUT.mkdir(parents=True, exist_ok=True)
    p_link = versioned_path(OUT, "linked_students", "csv")
    with open(p_link, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["hash"])
        for h in sorted(both):
            w.writerow([h])

    L = [
        "LONGITUDINAL LINKAGE (clean; 7 staff hashes removed from both)",
        f"exam students (outcome):      {len(exam)}",
        f"mission students (process):   {len(missions)}",
        f"in BOTH (process->outcome):   {len(both)}",
        f"  = {100*len(both)/len(exam):.1f}% of exam students have year data",
        f"  = {100*len(both)/len(missions):.1f}% of year students sat the exam",
        f"exam-only (no tracked year work): {len(exam_only)}",
        f"mission-only (didn't sit exam):   {len(miss_only)}",
    ]
    p_sum = versioned_path(OUT, "linkage_summary", "txt")
    with open(p_sum, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    print("\n".join(L))
    print(f"\nwrote:\n  {p_link}\n  {p_sum}")


if __name__ == "__main__":
    main()
