"""33 - Re-grade validation & comparison tables (kept for the report).

Two comparisons, saved to disk so the report cites fixed numbers (no ad-hoc recompute):

  A. SHOWN (old, v0) vs TRUE (official re-grade), per question. Shows the inflation we set
     out to correct: how many submissions changed, and in which direction. Split into FINAL
     submissions (INGInious truly grades these -> should mostly agree) vs ALL submissions
     (intermediates were the inflated ones).

  B. OUR sandbox recompute (retired script 06/07, archived in res_python_v0) vs the OFFICIAL
     re-grade, per question. This is the validation of our earlier method (exact-match %).

Inputs:
  v0 shown  : data/v2/last_archive/2026.exam/2026.01_comment_v0/q*/data.csv
  official  : data/v2/last_archive/2026.exam/2026.01_comment/q*/data.csv
  our recompute: data/v2/res_python_v0/true_scores/exam_clean_true_v*.csv

Outputs -> clustering/regrade_validation_summary_v*.txt ; plots/fig_regrade_shown_vs_true_v*
"""
import csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
V0 = REPO / "data/v2/last_archive/2026.exam/2026.01_comment_v0"
OFF = REPO / "data/v2/last_archive/2026.exam/2026.01_comment"
RECOMPUTE_GLOB = str(REPO / "data/v2/res_python_v0/true_scores/exam_clean_true_v*.csv")
CLU = REPO / "data/v2/res_python/clustering"
FIGS = REPO / "data/v2/res_python/plots"
QS = ["q1", "q2", "q3", "q4", "q5", "q6"]


def versioned(stem, ext, base):
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"{stem}_v{n}.{ext}").exists(): n += 1
    return base / f"{stem}_v{n}.{ext}"


def scores(base):
    """(hash,q,n_submission) -> score, and final-submission set."""
    s = {}; last = defaultdict(int)
    for q in QS:
        for r in csv.DictReader(open(base / q / "data.csv")):
            k = (r["hash"], q, int(r["n_submission"]))
            s[k] = float(r["score"])
            last[(r["hash"], q)] = max(last[(r["hash"], q)], int(r["n_submission"]))
    finals = {(h, q, n) for (h, q), n in last.items()}
    return s, finals


def main():
    shown, sf = scores(V0)
    true, tf = scores(OFF)
    L = ["RE-GRADE VALIDATION (official INGInious re-grade).", ""]

    # --- A. shown (v0) vs true (official) ---
    L.append("=== A. shown (old) vs true (official re-grade) — per question ===")
    L.append(f"  {'q':<4} {'common':>7} {'changed%':>9} {'lower%':>7} {'higher%':>8}  | {'FINALS match%':>13}")
    per_q_dir = {}
    for q in QS:
        common = [k for k in true if k[1] == q and k in shown]
        chg = [k for k in common if abs(true[k] - shown[k]) > 1e-6]
        lo = sum(1 for k in chg if true[k] < shown[k]); hi = len(chg) - lo
        fin = [k for k in common if k in tf]
        fin_match = sum(1 for k in fin if abs(true[k] - shown[k]) < 1e-6)
        per_q_dir[q] = (100 * lo / max(len(common), 1), 100 * hi / max(len(common), 1),
                        100 * (len(common) - len(chg)) / max(len(common), 1))
        L.append(f"  {q:<4} {len(common):>7} {100*len(chg)/max(len(common),1):>8.0f}% "
                 f"{100*lo/max(len(common),1):>6.0f}% {100*hi/max(len(common),1):>7.0f}%  | "
                 f"{100*fin_match/max(len(fin),1):>12.0f}%")
    allc = [k for k in true if k in shown]
    chg = sum(1 for k in allc if abs(true[k] - shown[k]) > 1e-6)
    fin = [k for k in allc if k in tf]; fmatch = sum(1 for k in fin if abs(true[k] - shown[k]) < 1e-6)
    L.append(f"  ALL  {len(allc):>7} {100*chg/len(allc):>8.0f}% "
             f"{'':>7}{'':>8}  | {100*fmatch/len(fin):>12.1f}%")
    L.append("  reading: intermediates changed a lot (shown scores were inflated); FINAL submissions")
    L.append("           mostly already matched (INGInious truly grades the final).")

    # --- B. our recompute vs official ---
    L.append("\n=== B. our sandbox recompute (retired 06/07) vs official re-grade ===")
    rec = {}
    rc = sorted(glob.glob(RECOMPUTE_GLOB))
    if rc:
        for r in csv.DictReader(open(rc[-1])):
            rec[(r["hash"], r["qname"], int(r["n_submission"]))] = float(r["score"])
        L.append(f"  {'q':<4} {'common':>7} {'exact-match%':>13} {'mean|Δ|':>8}")
        for q in QS:
            common = [k for k in true if k[1] == q and k in rec]
            ex = sum(1 for k in common if abs(true[k] - rec[k]) < 1e-6)
            md = np.mean([abs(true[k] - rec[k]) for k in common]) if common else 0
            L.append(f"  {q:<4} {len(common):>7} {100*ex/max(len(common),1):>12.1f}% {md:>8.2f}")
        common = [k for k in true if k in rec]
        ex = sum(1 for k in common if abs(true[k] - rec[k]) < 1e-6)
        L.append(f"  ALL  {len(common):>7} {100*ex/max(len(common),1):>12.1f}%")
        L.append("  reading: our earlier recompute was ~99.6% exact vs the official grade (q5 100%,")
        L.append("           q6 ~97% — the reconstructed given-code was the only soft spot).")
    else:
        L.append("  (our recompute archive not found at res_python_v0; skipped)")

    ps = versioned("regrade_validation_summary", "txt", CLU)
    Path(ps).write_text("\n".join(L) + "\n"); print("\n".join(L)); print(f"wrote {ps.name}")

    # figure A: per-question shown-vs-true direction
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(QS)); w = 0.6
    lo = [per_q_dir[q][0] for q in QS]; hi = [per_q_dir[q][1] for q in QS]; sa = [per_q_dir[q][2] for q in QS]
    ax.bar(x, sa, w, label="unchanged", color="#BBBBBB")
    ax.bar(x, lo, w, bottom=sa, label="true < shown (was inflated)", color="#D55E00")
    ax.bar(x, hi, w, bottom=[sa[i]+lo[i] for i in range(len(QS))], label="true > shown", color="#0072B2")
    ax.set_xticks(x); ax.set_xticklabels(QS); ax.set_ylabel("% of submissions"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8); ax.set_title("Official re-grade vs old shown scores, per question", fontsize=11)
    p = versioned("fig_regrade_shown_vs_true", "pdf", FIGS)
    for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig); print(f"fig {p.stem}")


if __name__ == "__main__":
    main()
