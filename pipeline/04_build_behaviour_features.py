"""04 - Per-student BEHAVIOUR feature table (Tier 1: from clean CSVs, no .py parsing).

Builds one row per student for the exam and for the missions. Features are computed
per (student, question) then aggregated per student by MEDIAN (outlier-robust), with
count-based totals. BEHAVIOUR features are what we cluster on; OUTCOME features are
held aside (to characterise/validate, never to define profiles). Score is demoted.

Definitions (documented; thresholds DATA-DERIVED, not hardcoded):
  - PASS threshold = 50 -> solved = best_score >= 50. (institutional: the course pass mark.)
  - rhythm gaps = seconds between consecutive submissions of the SAME question, capped at
    24h (a gap beyond a day is a return across sessions, not within-session rhythm).
  - fast retry / long pause thresholds are COMPUTED from each dataset's own gap
    distribution: fast = 10th percentile, long = 90th percentile (the extreme tenths).
    So there is no hardcoded 60s/600s; the cut adapts to the data (exam vs coursework)
    and is reported in the summary. See feedback_tfe/PARAMETERS.md.

Outputs (versioned) -> data/v2/res_python/features/
  exam_behaviour_features_vN.csv
  missions_behaviour_features_vN.csv
  behaviour_features_summary_vN.txt
"""
import csv
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np

from common import versioned_path, parse_ts

REPO = Path(__file__).resolve().parents[3]
EXAM_CLEAN_GLOB = str(REPO / "data/v2/res_python/exam_audit/exam_clean_v*.csv")
EXAM_TRUE_GLOB = str(REPO / "data/v2/res_python/true_scores/exam_clean_true_v*.csv")  # true scores (07)
EXAM_FLAG_GLOB = str(REPO / "data/v2/res_python/exam_audit/exam_flagged_hashes_v*.csv")
MISS_GLOB = str(REPO / "data/v2/last_archive/2025.Q1/mission_*_comment/*/data.csv")
OUT = REPO / "data/v2/res_python/features"

PASS = 50.0                     # institutional: the course pass mark (cited, not tuned)
FAST_PCTL, LONG_PCTL = 10, 90   # fast/long gap thresholds = these percentiles of the gap dist
GAP_CAP_S = 24 * 3600.0         # ignore gaps > 24h (returns across sessions, not rhythm)

BEHAVIOUR_COLS = ["questions_attempted", "total_attempts", "median_attempts",
                  "median_fast_retry_ratio", "median_long_pause_ratio",
                  "median_mean_delta_sec", "median_span_sec",
                  "median_time_to_success_sec", "median_improving_ratio"]
OUTCOME_COLS = ["median_best_score", "median_final_score", "median_score_gain",
                "solved_pass_count", "solved_pass_rate",      # best >= 50
                "solved_full_count", "solved_full_rate",      # best == 100
                "never_solved_count"]                          # never reached pass


def latest(pat):
    fs = sorted(glob.glob(pat))
    if not fs:
        raise FileNotFoundError(pat)
    return fs[-1]


def consecutive_gaps(rows):
    """Seconds between consecutive submissions of one (student, question), capped at 24h."""
    rows = sorted(rows, key=lambda r: int(r["n_submission"]))
    ts = [parse_ts(r["timestamp"]) for r in rows]
    g = [(ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1) if ts[i] and ts[i + 1]]
    return [x for x in g if 0 <= x <= GAP_CAP_S]


def per_question_stats(rows, fast_s, long_s):
    """rows for one (student, question), each dict with score, n_submission, timestamp.
    fast_s/long_s are the data-derived gap thresholds for this dataset."""
    rows = sorted(rows, key=lambda r: int(r["n_submission"]))
    scores = [float(r["score"]) for r in rows]
    ts = [parse_ts(r["timestamp"]) for r in rows]
    attempts = len(rows)
    best, first, final = max(scores), scores[0], scores[-1]

    # improving ratio (share of consecutive positive score deltas)
    deltas = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
    improving = np.mean([d > 0 for d in deltas]) if deltas else np.nan

    # timing gaps between consecutive submissions of this question (24h-capped)
    gaps = consecutive_gaps(rows)
    fast = np.mean([g <= fast_s for g in gaps]) if gaps else np.nan
    longp = np.mean([g >= long_s for g in gaps]) if gaps else np.nan
    mean_delta = np.mean(gaps) if gaps else np.nan
    span = (ts[-1] - ts[0]).total_seconds() if ts[0] and ts[-1] else np.nan

    # time to first pass (>= PASS)
    tts = np.nan
    for r, sc, t in zip(rows, scores, ts):
        if sc >= PASS and ts[0] and t:
            tts = (t - ts[0]).total_seconds()
            break

    return {"attempts": attempts, "best": best, "first": first, "final": final,
            "improving": improving, "fast": fast, "long": longp,
            "mean_delta": mean_delta, "span": span, "tts": tts,
            "solved_pass": best >= PASS, "solved_full": best >= 100.0,
            "date": ts[0].date() if ts[0] else None}


def build(rows, qkey, is_mission):
    by_sq = defaultdict(list)          # (hash, qkey) -> rows
    for r in rows:
        by_sq[(r["hash"], qkey(r))].append(r)

    # pass 1: derive the fast/long gap thresholds from THIS dataset's gap distribution
    all_gaps = []
    for rr in by_sq.values():
        all_gaps += consecutive_gaps(rr)
    fast_s = float(np.percentile(all_gaps, FAST_PCTL)) if all_gaps else 0.0
    long_s = float(np.percentile(all_gaps, LONG_PCTL)) if all_gaps else 0.0

    per_student = defaultdict(list)    # hash -> list of per-question stat dicts
    for (h, q), rr in by_sq.items():
        per_student[h].append(per_question_stats(rr, fast_s, long_s))

    def med(vals):
        arr = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
        return float(np.median(arr)) if arr.size else ""

    out = []
    for h, qs in per_student.items():
        row = {"hash": h}
        row["questions_attempted"] = len(qs)
        row["total_attempts"] = sum(q["attempts"] for q in qs)
        row["median_attempts"] = med([q["attempts"] for q in qs])
        row["median_fast_retry_ratio"] = med([q["fast"] for q in qs])
        row["median_long_pause_ratio"] = med([q["long"] for q in qs])
        row["median_mean_delta_sec"] = med([q["mean_delta"] for q in qs])
        row["median_span_sec"] = med([q["span"] for q in qs])
        row["median_time_to_success_sec"] = med([q["tts"] for q in qs])
        row["median_improving_ratio"] = med([q["improving"] for q in qs])
        row["median_best_score"] = med([q["best"] for q in qs])
        row["median_final_score"] = med([q["final"] for q in qs])
        row["median_score_gain"] = med([q["best"] - q["first"] for q in qs])
        n_q = len(qs)
        s_pass = sum(1 for q in qs if q["solved_pass"])
        s_full = sum(1 for q in qs if q["solved_full"])
        row["solved_pass_count"] = s_pass
        row["solved_pass_rate"] = round(s_pass / n_q, 4)
        row["solved_full_count"] = s_full
        row["solved_full_rate"] = round(s_full / n_q, 4)
        row["never_solved_count"] = n_q - s_pass
        if is_mission:
            dates = [q["date"] for q in qs if q["date"]]
            row["active_weeks"] = len({d.isocalendar()[:2] for d in dates})
            row["active_days"] = len(set(dates))
        out.append(row)
    return out, (fast_s, long_s)


def write_table(path, rows, extra_behaviour=()):
    cols = ["hash"] + BEHAVIOUR_COLS + list(extra_behaviour) + OUTCOME_COLS
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main():
    staff = {r["hash"] for r in csv.DictReader(open(latest(EXAM_FLAG_GLOB)))}

    # --- EXAM (already staff-free; prefer TRUE-scored table from 07 if present) ---
    _t = sorted(glob.glob(EXAM_TRUE_GLOB))
    exam_src = _t[-1] if _t else latest(EXAM_CLEAN_GLOB)
    print(f"[exam] scores from: {Path(exam_src).name}")
    exam_rows = list(csv.DictReader(open(exam_src)))
    exam_feats, exam_thr = build(exam_rows, qkey=lambda r: r["qname"], is_mission=False)

    # --- MISSIONS (remove staff) ---
    miss_rows = []
    for p in sorted(glob.glob(MISS_GLOB)):
        mission = Path(p).parent.parent.name
        for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
            if r["hash"] in staff:
                continue
            r["_mission"] = mission
            miss_rows.append(r)
    miss_feats, miss_thr = build(miss_rows, qkey=lambda r: r["_mission"] + "/" + r["qname"], is_mission=True)

    OUT.mkdir(parents=True, exist_ok=True)
    p_exam = versioned_path(OUT, "exam_behaviour_features", "csv")
    p_miss = versioned_path(OUT, "missions_behaviour_features", "csv")
    write_table(p_exam, exam_feats)
    write_table(p_miss, miss_feats, extra_behaviour=("active_weeks", "active_days"))

    p_sum = versioned_path(OUT, "behaviour_features_summary", "txt")
    L = []
    L.append(f"BEHAVIOUR FEATURE TABLES (Tier 1, PASS={PASS} [institutional])")
    L.append(f"data-derived gap thresholds (p{FAST_PCTL}/p{LONG_PCTL}, 24h-capped):")
    L.append(f"  exam:     fast<= {exam_thr[0]:.0f}s, long>= {exam_thr[1]:.0f}s")
    L.append(f"  missions: fast<= {miss_thr[0]:.0f}s, long>= {miss_thr[1]:.0f}s")
    L.append(f"exam:     {len(exam_feats)} students  -> {p_exam.name}")
    L.append(f"missions: {len(miss_feats)} students  -> {p_miss.name}")
    L.append("")
    L.append("CLUSTER ON (behaviour): " + ", ".join(BEHAVIOUR_COLS) + ", [missions: active_weeks, active_days]")
    L.append("HELD ASIDE (outcome):   " + ", ".join(OUTCOME_COLS))
    L.append("")
    # quick sanity: exam medians
    def colmean(rows, c):
        v = [float(r[c]) for r in rows if r.get(c) not in ("", None)]
        return round(sum(v) / len(v), 2) if v else float("nan")
    L.append("exam sanity (means over students):")
    for c in ["median_attempts", "median_fast_retry_ratio", "median_long_pause_ratio",
              "median_best_score", "solved_pass_rate", "solved_full_rate"]:
        L.append(f"  {c}: {colmean(exam_feats, c)}")
    with open(p_sum, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote:\n  {p_exam}\n  {p_miss}\n  {p_sum}")


if __name__ == "__main__":
    main()
