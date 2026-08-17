"""32 - Per-question exam analysis on the v2 TRUE scores (re-derivation of the old
Chapter-4 "question difficulty / Q1->Q6 myth / time-vs-score" material, done correctly).

The June version was marked wrong by Olivier ("0.4 is not strong"; the "adjacent questions
are most correlated" claim circled FALSE; "Speculation" on fatigue). Here we recompute on
the true scores and report honestly, with correct effect-size language (Cohen: |r|~0.1 small,
0.3 moderate, 0.5 large) and no asserted causes.

Per (student, question): final true score (last submission), best, active time (sum of
consecutive gaps, 24h-capped), attempts. Then:
  - per-question difficulty: mean final score, pass rate (final >= 50), full-solve rate (=100)
  - question x question correlation of per-student final scores (Pearson, pairwise-complete)
  - time-vs-score: Pearson(active time, final score) per question

Outputs -> clustering/per_question_summary_v*.txt
figures  -> plots/fig_exam_question_difficulty_v* , fig_exam_question_correlation_v*
"""
import csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import parse_ts

REPO = Path(__file__).resolve().parents[3]
CLU = REPO / "data/v2/res_python/clustering"
FIGS = REPO / "data/v2/res_python/plots"
EXAM_TRUE_GLOB = str(REPO / "data/v2/res_python/true_scores/exam_clean_true_v*.csv")
LINKED_GLOB = str(REPO / "data/v2/res_python/linkage/linked_students_v*.csv")
GAP_CAP_S = 24 * 3600.0
PASS = 50.0


def latest(pat): return sorted(glob.glob(str(pat)))[-1]


def versioned(stem, ext, base):
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"{stem}_v{n}.{ext}").exists(): n += 1
    return base / f"{stem}_v{n}.{ext}"


def label(r):
    a = abs(r)
    return "large" if a >= 0.5 else "moderate" if a >= 0.3 else "small" if a >= 0.1 else "negligible"


def main():
    rows = list(csv.DictReader(open(latest(EXAM_TRUE_GLOB))))
    # Stay on the 569 linked set (students present in BOTH coursework and exam), the
    # scope used everywhere else in the report. Do not analyse the full 581 exam-takers.
    linked = {r["hash"] for r in csv.DictReader(open(latest(LINKED_GLOB)))}
    by_sq = defaultdict(list)
    for r in rows:
        if r["hash"] not in linked:
            continue
        by_sq[(r["hash"], r["qname"])].append(r)

    final, active = {}, {}   # (hash,q) -> final true score / active seconds
    for (h, q), rr in by_sq.items():
        rr = sorted(rr, key=lambda r: int(r["n_submission"]))
        final[(h, q)] = float(rr[-1]["score"])
        ts = [parse_ts(r["timestamp"]) for r in rr]
        gaps = [(ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1) if ts[i] and ts[i + 1]]
        active[(h, q)] = sum(g for g in gaps if 0 <= g <= GAP_CAP_S)

    questions = sorted({q for _, q in by_sq})
    students = sorted({h for h, _ in by_sq})

    L = [f"PER-QUESTION EXAM ANALYSIS (v2 true scores). {len(students)} students, {len(questions)} questions.",
         "effect-size labels (Cohen): |r|>=0.5 large, >=0.3 moderate, >=0.1 small.", ""]

    # difficulty
    L.append("=== difficulty (final true score per question) ===")
    L.append(f"  {'q':<10} {'n':>4} {'mean':>6} {'median':>7} {'pass%':>6} {'full%':>6} {'mean_time_min':>13}")
    diff = {}
    for q in questions:
        fs = np.array([final[(h, q)] for h in students if (h, q) in final])
        tm = np.array([active[(h, q)] for h in students if (h, q) in active]) / 60.0
        diff[q] = (len(fs), fs.mean(), np.median(fs), 100 * np.mean(fs >= PASS), 100 * np.mean(fs >= 100), tm.mean())
        L.append(f"  {q:<10} {len(fs):>4} {fs.mean():>6.1f} {np.median(fs):>7.1f} "
                 f"{100*np.mean(fs>=PASS):>6.0f} {100*np.mean(fs>=100):>6.0f} {tm.mean():>13.1f}")

    # question x question correlation (pairwise-complete Pearson on final scores)
    M = np.full((len(questions), len(questions)), np.nan)
    for i, qi in enumerate(questions):
        for j, qj in enumerate(questions):
            xs = [(final[(h, qi)], final[(h, qj)]) for h in students if (h, qi) in final and (h, qj) in final]
            if len(xs) >= 10:
                a, b = np.array(xs).T
                if a.std() > 0 and b.std() > 0:
                    M[i, j] = np.corrcoef(a, b)[0, 1]
    L.append("\n=== question x question correlation of final scores (Pearson) ===")
    L.append("        " + "  ".join(f"{q:>6}" for q in questions))
    for i, qi in enumerate(questions):
        L.append(f"  {qi:<6}" + "  ".join(f"{M[i,j]:6.2f}" if np.isfinite(M[i, j]) else "     ." for j in range(len(questions))))
    off = [(questions[i], questions[j], M[i, j]) for i in range(len(questions)) for j in range(i + 1, len(questions)) if np.isfinite(M[i, j])]
    off.sort(key=lambda x: -abs(x[2]))
    L.append("  strongest pairs: " + "; ".join(f"{a}-{b} r={r:.2f} ({label(r)})" for a, b, r in off[:3]))
    L.append("  weakest pairs:   " + "; ".join(f"{a}-{b} r={r:.2f} ({label(r)})" for a, b, r in off[-3:]))
    adjacent = [r for a, b, r in off if abs(int(a[-1]) - int(b[-1])) == 1] if all(q[-1].isdigit() for q in questions) else []
    if adjacent:
        L.append(f"  adjacent-question pairs: mean r={np.mean(adjacent):.2f} vs all pairs mean r={np.mean([r for *_ ,r in off]):.2f} "
                 f"-> adjacency is {'NOT ' if np.mean(adjacent) <= np.mean([r for *_,r in off]) else ''}the main driver")

    # time vs score per question
    L.append("\n=== time spent vs final score, per question (Pearson) ===")
    for q in questions:
        pairs = [(active[(h, q)], final[(h, q)]) for h in students if (h, q) in final]
        a, b = np.array(pairs).T
        r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else float("nan")
        L.append(f"  {q:<10} r={r:>5.2f} ({label(r)})   pass rate {diff[q][3]:.0f}%")

    ps = versioned("per_question_summary", "txt", CLU); Path(ps).write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"wrote {ps.name}")

    # figure 1: per-question performance (mean AND median final score + pass rate)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(questions)); w = 0.27
    ax.bar(x - w, [diff[q][1] for q in questions], w, color="#0072B2", label="mean final score")
    ax.bar(x,     [diff[q][2] for q in questions], w, color="#56B4E9", label="median final score")
    ax.bar(x + w, [diff[q][3] for q in questions], w, color="#E69F00", label="pass rate (%)")
    ax.set_xticks(x); ax.set_xticklabels(questions); ax.set_ylim(0, 100)
    ax.set_ylabel("score / %"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Per-question performance: mean and median final score, and pass rate", fontsize=10)
    p = versioned("fig_exam_question_difficulty", "pdf", FIGS)
    for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig); print(f"fig {p.stem}")

    # figure 2: correlation heatmap
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(x); ax.set_xticklabels(questions); ax.set_yticks(x); ax.set_yticklabels(questions)
    for i in range(len(questions)):
        for j in range(len(questions)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Question-to-question correlation of final scores", fontsize=11)
    p = versioned("fig_exam_question_correlation", "pdf", FIGS)
    for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig); print(f"fig {p.stem}")


if __name__ == "__main__":
    main()
