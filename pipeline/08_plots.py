"""08 - Figures for the supervisor meetings. Each is born from a question, and is analysed +
interpreted in its title/subtitle. Saved as PDF *and* PNG, versioned (never overwrite).

Outputs -> data/v2/res_python/plots/<name>_vN.{pdf,png}
"""
import csv
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.autolayout": False,
})
# Okabe-Ito colourblind-safe palette
BLUE, ORANGE, GREEN, VERM, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8C8C8C"

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "data/v2/res_python"
OUT = RES / "plots"


def latest(pat):
    return sorted(glob.glob(str(pat)))[-1]


def save_fig(fig, stem, subtitle=""):
    OUT.mkdir(parents=True, exist_ok=True)
    n = 1
    while (OUT / f"{stem}_v{n}.pdf").exists() or (OUT / f"{stem}_v{n}.png").exists():
        n += 1
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}_v{n}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}_v{n}.pdf/.png")
    return n


def col_stats(path, c):
    v = np.array([float(r[c]) for r in csv.DictReader(open(path)) if r.get(c) not in ("", None)])
    return float(v.mean()), float(v.std())


def barlabels(ax, bars, stds, fmt="{:.0%}"):
    for b, s in zip(bars, stds):
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h + s),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")


ERRKW = dict(capsize=3, error_kw=dict(lw=1, ecolor="#333333"))   # std whiskers on bars


# ---------- Figure 1: are intermediate scores trustworthy? ----------
def fig_true_score():
    rows = list(csv.DictReader(open(latest(RES / "true_scores/true_scores_all_v*.csv"))))
    last = defaultdict(int)
    for r in rows:
        last[(r["hash"], r["qname"])] = max(last[(r["hash"], r["qname"])], int(r["n_submission"]))
    fin, inter = [], []
    for r in rows:
        true = float(r["true_score"]) if r["status"] == "OK" and r["true_score"] != "" else 0.0
        d = float(r["filename_score"]) - true         # >0 = INGInious overstated
        (fin if int(r["n_submission"]) == last[(r["hash"], r["qname"])] else inter).append(d)
    fin, inter = np.array(fin), np.array(inter)
    ex_f = 100 * np.mean(np.abs(fin) < 1e-6)
    ex_i = 100 * np.mean(np.abs(inter) < 1e-6)
    infl = 100 * np.mean(inter > 1e-6)
    bins = np.arange(-100, 105, 5)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    a1.hist(fin, bins=bins, color=BLUE)
    a1.set_title(f"Final submissions (n={len(fin):,})\n{ex_f:.1f}% exact match", fontsize=11)
    a2.hist(inter, bins=bins, color=ORANGE)
    a2.set_title(f"Intermediate submissions (n={len(inter):,})\n{ex_i:.0f}% exact, {infl:.0f}% inflated", fontsize=11)
    for a in (a1, a2):
        a.axvline(0, color=GRAY, lw=1, ls="--")
        a.set_xlabel("INGInious score − recomputed true score  (points)")
    a1.set_ylabel("submissions")
    fig.suptitle("Shown INGInious score − recomputed true score, by submission position",
                 fontsize=12, y=1.0)
    save_fig(fig, "fig_true_score_reliability")


# ---------- Figure 2: iteration under exam pressure vs coursework ----------
def fig_iteration_contrast():
    exam = latest(RES / "features/exam_diff_features_v*.csv")
    miss = latest(RES / "features/missions_diff_features_v*.csv")
    metrics = [("churn_ratio", "Churn\n(edits, no score gain)"),
               ("breakthrough_ratio", "Breakthrough\n(tiny fix → +50 pts)")]
    ex = [col_stats(exam, m) for m, _ in metrics]
    co = [col_stats(miss, m) for m, _ in metrics]
    ex_m, ex_s = [a for a, _ in ex], [b for _, b in ex]
    co_m, co_s = [a for a, _ in co], [b for _, b in co]
    x = np.arange(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    b1 = ax.bar(x - w / 2, ex_m, w, yerr=ex_s, color=VERM, label="Exam (Jan 2026)", **ERRKW)
    b2 = ax.bar(x + w / 2, co_m, w, yerr=co_s, color=BLUE, label="Coursework (Q1, over the year)", **ERRKW)
    barlabels(ax, b1, ex_s); barlabels(ax, b2, co_s)
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylabel("share of consecutive edits per student\n(mean ± SD over students)")
    ax.legend(frameon=False)
    ax.set_title("Iteration behaviour: exam vs coursework", fontsize=12)
    save_fig(fig, "fig_iteration_exam_vs_coursework")


# ---------- Figure 3: true exam difficulty ----------
def fig_exam_difficulty():
    behs = sorted(glob.glob(str(RES / "features/exam_behaviour_features_v*.csv")))
    old, new = behs[-2], behs[-1]   # v2 = shown/filename scores, v3 = true scores
    labels = [("solved_pass_rate", "Passed (≥50%)"), ("solved_full_rate", "Fully solved (100%)")]
    ov = [col_stats(old, c) for c, _ in labels]
    nv = [col_stats(new, c) for c, _ in labels]
    ov_m, ov_s = [a for a, _ in ov], [b for _, b in ov]
    nv_m, nv_s = [a for a, _ in nv], [b for _, b in nv]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    b1 = ax.bar(x - w / 2, ov_m, w, yerr=ov_s, color=GRAY, label="Using shown INGInious scores", **ERRKW)
    b2 = ax.bar(x + w / 2, nv_m, w, yerr=nv_s, color=GREEN, label="Using recomputed TRUE scores", **ERRKW)
    barlabels(ax, b1, ov_s); barlabels(ax, b2, nv_s)
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in labels])
    ax.set_ylim(0, 1.0); ax.set_ylabel("share of student·question pairs\n(mean ± SD over students)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Exam success: shown vs recomputed true scores", fontsize=12)
    save_fig(fig, "fig_exam_true_difficulty")


if __name__ == "__main__":
    print("generating figures for the supervisor meetings ->", OUT)
    fig_true_score()
    fig_iteration_contrast()
    fig_exam_difficulty()
    print("done.")
