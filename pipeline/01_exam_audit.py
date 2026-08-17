"""01 - January-2026 exam data-quality audit (read-only on source).

Confirms the exam structure (single day, two slots), reconciles the CSV against the
filenames, and detects non-student (staff/test) hashes behaviourally.

The 588 -> 581 cleaning (matches Chapter 3): the delivered official re-grade
(2026.01_comment) already contains only the 581 genuine exam students. The 7
non-genuine accounts (hashes active on days other than 22 Jan 2026) were flagged by
the TIMING rules below on the raw 588-account data (2026.01_comment_v0, which still
carries the old shown scores; the flagged record is in res_python_v0/exam_audit) and
were excluded before re-grading. Flagging is timestamp-based and score-independent, so
it yields the same 7 whether run on the raw (_v0) or would-be raw official set; run on
the delivered 581 it flags 0 and confirms all are genuine. IMPORTANT: this script reads
2026.01_comment (the official-regrade scores). Do NOT repoint it at _v0 to "reproduce"
the 588 count: that would pull the OLD shown scores into exam_clean and corrupt every
downstream score. Only the timing rules ever remove accounts; the score/submission/hash
reconciliation and the status-vs-score coherence are integrity checks that remove nothing.

Exam facts (LINFO1101/LEPL1401, 22 Jan 2026):
  slot 1 = 09:00-12:00 (PEPS +1h -> 13:00), slot 2 = 12:15-15:15 (PEPS +1h -> 16:15),
  +15 min launch buffer. Legitimate window: 22 Jan, ~09:00-16:30; max span ~4h15 (PEPS).

Outputs (versioned) -> data/v2/res_python/exam_audit/
  exam_flagged_hashes_vN.csv   candidate non-students + reasons
  exam_clean_vN.csv            submissions with flagged hashes removed (+ slot, relative time)
  exam_audit_summary_vN.txt    human-readable summary
"""
import csv
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

from common import versioned_path, parse_ts, parse_fname, load_rows

REPO = Path(__file__).resolve().parents[3]
EXAM_GLOB = str(REPO / "data/v2/last_archive/2026.exam/2026.01_comment/q*/data.csv")
OUT = REPO / "data/v2/res_python/exam_audit"

EXAM_DATE = "2026-01-22"
SLOT1_START = datetime(2026, 1, 22, 9, 0, 0)
SLOT2_START = datetime(2026, 1, 22, 12, 15, 0)
MAX_SPAN_MIN = 255          # 4h15 = 3h normal + 1h PEPS + 15 min buffer
DAY_OPEN, DAY_CLOSE = 9.0, 16.5   # legitimate hours-of-day on exam day


def hour_of(dt):
    return dt.hour + dt.minute / 60.0


def main():
    rows = load_rows(sorted(glob.glob(EXAM_GLOB)))
    n = len(rows)

    # --- reconcile CSV vs filename (score/submission/hash) ---
    rec = Counter()
    for r in rows:
        f = parse_fname(r["file"])
        if f is None:
            rec["fname_unparsed"] += 1
            continue
        rec["score_match"] += (abs(f["grade"] - float(r["score"])) < 1e-6)
        rec["submission_match"] += (f["submission"] == int(r["n_submission"]))
        rec["hash_match"] += (f["hash"] == r["hash"])

    # --- per-hash aggregation ---
    by_hash = defaultdict(list)
    bad_ts = 0
    for r in rows:
        dt = parse_ts(r["timestamp"])
        if dt is None:
            bad_ts += 1
            continue
        by_hash[r["hash"]].append(dt)

    flagged = {}          # hash -> reasons
    slot_of = {}          # hash -> 1/2 (only for kept students)
    dates = Counter()
    hours = Counter()
    statscore = Counter()
    for r in rows:
        dt = parse_ts(r["timestamp"])
        if dt:
            dates[dt.date().isoformat()] += 1
            hours[dt.hour] += 1
        sc = float(r["score"])
        statscore[(r["status"], "100" if sc == 100 else ("0" if sc == 0 else "mid"))] += 1

    for h, times in by_hash.items():
        times.sort()
        span = (times[-1] - times[0]).total_seconds() / 60.0
        multiday = any(t.date().isoformat() != EXAM_DATE for t in times)
        out_hours = any(
            (hour_of(t) < DAY_OPEN or hour_of(t) > DAY_CLOSE)
            for t in times if t.date().isoformat() == EXAM_DATE
        )
        over = span > MAX_SPAN_MIN
        reasons = [name for name, cond in
                   (("multiday", multiday), ("out_of_hours", out_hours), ("span>4h15", over)) if cond]
        if reasons:
            flagged[h] = (len(times), span, times[0], times[-1], ";".join(reasons))
        else:
            first_h = hour_of(times[0])
            slot_of[h] = 1 if first_h < 12.1 else 2

    # ---------------- write artifacts ----------------
    OUT.mkdir(parents=True, exist_ok=True)

    p_flag = versioned_path(OUT, "exam_flagged_hashes", "csv")
    with open(p_flag, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["hash", "n_submissions", "span_minutes", "first_ts", "last_ts", "reasons"])
        for h, (ns, span, t0, t1, reasons) in sorted(flagged.items(), key=lambda x: -x[1][1]):
            w.writerow([h, ns, round(span, 1), t0.isoformat(), t1.isoformat(), reasons])

    p_clean = versioned_path(OUT, "exam_clean", "csv")
    kept = 0
    with open(p_clean, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["hash", "qname", "n_submission", "score", "status", "timestamp",
                    "slot", "minutes_from_slot_start", "file"])
        for r in rows:
            h = r["hash"]
            if h in flagged:
                continue
            dt = parse_ts(r["timestamp"])
            if dt is None:
                continue
            slot = slot_of.get(h)
            start = SLOT1_START if slot == 1 else SLOT2_START
            rel = (dt - start).total_seconds() / 60.0
            w.writerow([h, r["qname"], r["n_submission"], r["score"], r["status"],
                        r["timestamp"], slot, round(rel, 2), r["file"]])
            kept += 1

    p_sum = versioned_path(OUT, "exam_audit_summary", "txt")
    lines = []
    lines.append(f"EXAM AUDIT (2026.01)  -  {n} submissions, {len(by_hash)} distinct hashes")
    lines.append(f"unparseable timestamps: {bad_ts}")
    lines.append("")
    lines.append("CSV vs filename reconciliation (of parseable filenames):")
    parsed = n - rec["fname_unparsed"]
    lines.append(f"  filenames unparsed (Test*.py etc.): {rec['fname_unparsed']}")
    lines.append(f"  score match:      {rec['score_match']}/{parsed}")
    lines.append(f"  submission match: {rec['submission_match']}/{parsed}")
    lines.append(f"  hash match:       {rec['hash_match']}/{parsed}")
    lines.append("")
    lines.append("status vs score coherence:")
    for k, c in sorted(statscore.items()):
        lines.append(f"  status={k[0]:8s} score={k[1]:4s}: {c}")
    lines.append("")
    lines.append("submissions per date:")
    for d, c in sorted(dates.items()):
        lines.append(f"  {d}: {c}")
    lines.append("")
    lines.append("submissions per hour-of-day:")
    for hh in range(24):
        if hours.get(hh):
            lines.append(f"  {hh:02d}h: {hours[hh]}")
    lines.append("")
    kept_hashes = len(by_hash) - len(flagged)
    s1 = sum(1 for v in slot_of.values() if v == 1)
    s2 = sum(1 for v in slot_of.values() if v == 2)
    lines.append(f"FLAGGED non-student hashes: {len(flagged)}  (all their rows removed)")
    lines.append(f"kept students: {kept_hashes}  (slot1={s1}, slot2={s2})")
    lines.append(f"kept submissions: {kept}")
    with open(p_sum, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote:\n  {p_flag}\n  {p_clean}\n  {p_sum}")


if __name__ == "__main__":
    main()
