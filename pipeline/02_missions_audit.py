"""02 - Q1 2025-2026 weekly-missions data-quality audit (read-only on source).

Missions are formative weekly INGInious assignments done at home, with NO time window,
and students KEEP access after the semester -> so neither submission volume nor late
dates are valid outlier signals (both are legitimate for open practice).

Therefore the ONLY reliable non-student signal here is the cross-reference with the
staff/test hashes already identified in the exam audit (01). Mission-only staff (who
never touched the exam) are not reliably detectable and are reported as a known limit.

Everything else (dates, volumes, attrition) is reported as DESCRIPTIVE context only.

Outputs (versioned) -> data/v2/res_python/missions_audit/
  missions_flagged_hashes_vN.csv   exam-staff hashes found in the missions
  missions_audit_summary_vN.txt    reconciliation, coherence, attrition, dates (context)
"""
import csv
import glob
from pathlib import Path
from collections import defaultdict, Counter

from common import versioned_path, parse_ts, parse_fname

REPO = Path(__file__).resolve().parents[3]
MISS_GLOB = str(REPO / "data/v2/last_archive/2025.Q1/mission_*_comment/*/data.csv")
EXAM_FLAG_GLOB = str(REPO / "data/v2/res_python/exam_audit/exam_flagged_hashes_v*.csv")
OUT = REPO / "data/v2/res_python/missions_audit"


def latest(glob_pat):
    fs = sorted(glob.glob(glob_pat))
    return fs[-1] if fs else None


def main():
    paths = sorted(glob.glob(MISS_GLOB))

    rows = []
    for p in paths:
        mission = Path(p).parent.parent.name  # mission_XX_comment
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_mission"] = mission
                rows.append(r)
    n = len(rows)

    # staff hashes identified in the exam audit (the ONLY flag source for missions)
    exam_staff = set()
    ef = latest(EXAM_FLAG_GLOB)
    if ef:
        with open(ef, newline="", encoding="utf-8") as fh:
            exam_staff = {r["hash"] for r in csv.DictReader(fh)}

    rec = Counter()
    dates_month = Counter()
    statscore = Counter()
    bad_ts = 0
    date_min = date_max = None
    per_mission_students = defaultdict(set)
    per_mission_subs = Counter()
    per_hash_subs = Counter()
    dup_keys = Counter()

    for r in rows:
        f = parse_fname(r["file"])
        if f is None:
            rec["fname_unparsed"] += 1
        else:
            rec["score_match"] += (abs(f["grade"] - float(r["score"])) < 1e-6)
            rec["submission_match"] += (f["submission"] == int(r["n_submission"]))
            rec["hash_match"] += (f["hash"] == r["hash"])
        sc = float(r["score"])
        statscore[(r["status"], "100" if sc == 100 else ("0" if sc == 0 else "mid"))] += 1
        dt = parse_ts(r["timestamp"])
        if dt is None:
            bad_ts += 1
        else:
            d = dt.date().isoformat()
            dates_month[d[:7]] += 1
            date_min = d if date_min is None else min(date_min, d)
            date_max = d if date_max is None else max(date_max, d)
        per_mission_students[r["_mission"]].add(r["hash"])
        per_mission_subs[r["_mission"]] += 1
        per_hash_subs[r["hash"]] += 1
        dup_keys[(r["hash"], r["_mission"], r["qname"], r["n_submission"])] += 1

    # FLAG: only exam-staff hashes that also appear in the missions
    flagged = {h: "exam_staff_hash" for h in exam_staff if h in per_hash_subs}
    dups = {k: c for k, c in dup_keys.items() if c > 1}

    OUT.mkdir(parents=True, exist_ok=True)
    p_flag = versioned_path(OUT, "missions_flagged_hashes", "csv")
    with open(p_flag, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["hash", "total_submissions", "reason"])
        for h in sorted(flagged, key=lambda x: -per_hash_subs[x]):
            w.writerow([h, per_hash_subs[h], flagged[h]])

    p_sum = versioned_path(OUT, "missions_audit_summary", "txt")
    L = []
    L.append(f"MISSIONS AUDIT (2025.Q1)  -  {n} submissions, {len(per_hash_subs)} distinct hashes, {len(paths)} questions")
    L.append(f"date range: {date_min} .. {date_max}   (unparseable ts: {bad_ts})")
    L.append("(note: post-semester submissions are LEGITIMATE - students keep INGInious access.)")
    L.append("")
    parsed = n - rec["fname_unparsed"]
    L.append("CSV vs filename reconciliation:")
    L.append(f"  filenames unparsed: {rec['fname_unparsed']}")
    L.append(f"  score match:      {rec['score_match']}/{parsed}")
    L.append(f"  submission match: {rec['submission_match']}/{parsed}")
    L.append(f"  hash match:       {rec['hash_match']}/{parsed}")
    L.append("")
    L.append("status vs score coherence:")
    for k, c in sorted(statscore.items()):
        L.append(f"  status={k[0]:8s} score={k[1]:4s}: {c}")
    L.append("")
    L.append("submissions per month (DESCRIPTIVE - late months are legitimate practice):")
    for m, c in sorted(dates_month.items()):
        L.append(f"  {m}: {c}")
    L.append("")
    L.append("per-mission attrition (distinct students | submissions):")
    for m in sorted(per_mission_students):
        L.append(f"  {m}: {len(per_mission_students[m])} students | {per_mission_subs[m]} subs")
    L.append("")
    L.append(f"duplicate (hash,mission,qname,n_submission) keys: {len(dups)}")
    L.append("")
    present = sum(1 for h in exam_staff if h in per_hash_subs)
    L.append(f"FLAGGED = exam-staff hashes also present in missions: {present}/{len(exam_staff)}")
    for h in sorted(flagged, key=lambda x: -per_hash_subs[x]):
        L.append(f"  {h[:12]}: {per_hash_subs[h]} mission submissions")
    L.append("")
    L.append("KNOWN LIMIT: mission-only staff (never touched the exam) are not reliably")
    L.append("detectable from volume or dates; not flagged.")
    with open(p_sum, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    print("\n".join(L))
    print(f"\nwrote:\n  {p_flag}\n  {p_sum}")


if __name__ == "__main__":
    main()
