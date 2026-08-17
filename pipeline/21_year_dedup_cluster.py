"""21 - De-duplicated YEAR clustering. Tests whether the year structure is more than an
"engaged vs not" artifact of collinear features (see 20's correlation matrix).

Collapse the two redundant blocks to one representative each (keep the rest, incl. churn):
  engagement {questions_attempted, active_weeks, active_days} -> keep active_days
  pacing     {gap between tries, long_pauses}                 -> keep gap between tries
=> 11 features (from 14). 569 linked cohort.

Compares FULL(14) vs DEDUP(11) across k (silhouette + subsample-stability ARI), and gives
the k=3 dedup cluster signatures + mean exam grade. Keeps all features for description
(this only changes the clustering INPUT).

Outputs -> clustering/year_dedup_summary_v*.txt ; figures -> plots/fig_year_dedup_*
"""
import csv, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

REPO = Path(__file__).resolve().parents[3]
FEAT = REPO / "data/v2/res_python/features"
LINK = REPO / "data/v2/res_python/linkage"
CLU = REPO / "data/v2/res_python/clustering"
FIGS = REPO / "data/v2/res_python/plots"

DISPLAY = {
    "questions_attempted": "questions attempted", "median_attempts": "attempts/question",
    "median_fast_retry_ratio": "quick resubmissions", "median_long_pause_ratio": "long pauses",
    "median_mean_delta_sec": "gap between tries", "median_improving_ratio": "tries that improved",
    "median_edit_size": "lines changed/edit", "churn_ratio": "edits, no gain",
    "breakthrough_ratio": "small fix, big gain", "median_nloc": "code lines",
    "median_comment_ratio": "comment share", "median_n_concepts": "distinct concepts",
    "active_weeks": "active weeks", "active_days": "active days",
}
YEAR_FULL = list(DISPLAY)
DROP = {"questions_attempted", "active_weeks", "median_long_pause_ratio"}
YEAR_DEDUP = [c for c in DISPLAY if c not in DROP]
LOG = {"median_attempts", "median_mean_delta_sec", "median_edit_size", "median_nloc"}
KS = list(range(2, 8))


def latest(pat): return sorted(glob.glob(str(pat)))[-1]


def versioned(stem, ext, base):
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"{stem}_v{n}.{ext}").exists(): n += 1
    return base / f"{stem}_v{n}.{ext}"


def load(cols):
    linked = {r["hash"] for r in csv.DictReader(open(latest(LINK / "linked_students_v*.csv")))}
    feats = {r["hash"]: r for r in csv.DictReader(open(latest(FEAT / "missions_features_all_v*.csv")))}
    grades = {r["hash"]: float(r["exam_grade"]) for r in csv.DictReader(open(latest(CLU / "exam_score_categories_v*.csv")))}
    hashes = [h for h in feats if h in linked and h in grades]

    def val(h, c):
        try: return float(feats[h][c])
        except: return np.nan
    X = np.nan_to_num(np.array([[val(h, c) for c in cols] for h in hashes], float), nan=0.0)
    for j, c in enumerate(cols):
        if c in LOG: X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    return StandardScaler().fit_transform(X), np.array([grades[h] for h in hashes]), hashes


def stability(Xs, k, B=40, frac=0.8):
    rng = np.random.default_rng(0)
    full = fcluster(linkage(Xs, "ward"), k, "maxclust")
    n = len(Xs); m = int(frac * n); a = []
    for _ in range(B):
        idx = np.sort(rng.choice(n, m, replace=False))
        a.append(adjusted_rand_score(full[idx], fcluster(linkage(Xs[idx], "ward"), k, "maxclust")))
    return float(np.mean(a))


def curve(cols, label):
    Xs, grade, _ = load(cols)
    sil, stab = [], []
    for k in KS:
        lab = fcluster(linkage(Xs, "ward"), k, "maxclust")
        sil.append(silhouette_score(Xs, lab)); stab.append(stability(Xs, k))
    return Xs, grade, sil, stab


def main():
    Xf, gf, silF, stabF = curve(YEAR_FULL, "full")
    Xd, gd, silD, stabD = curve(YEAR_DEDUP, "dedup")
    L = [f"YEAR de-dup clustering (569 students). FULL={len(YEAR_FULL)} feats vs DEDUP={len(YEAR_DEDUP)} feats.",
         f"dropped (collinear): {sorted(DROP)}", "",
         f"{'k':>2}  {'sil_full':>8} {'stab_full':>9}  {'sil_dedup':>9} {'stab_dedup':>10}"]
    for i, k in enumerate(KS):
        L.append(f"{k:>2}  {silF[i]:8.3f} {stabF[i]:9.2f}  {silD[i]:9.3f} {stabD[i]:10.2f}")

    # k=3 dedup signatures
    lab = fcluster(linkage(Xd, "ward"), 3, "maxclust")
    order = {c: i + 1 for i, c in enumerate(sorted(set(lab), key=lambda c: gd[lab == c].mean()))}
    lab = np.array([order[c] for c in lab])
    L.append("\nDEDUP k=3 clusters (labelled by ascending mean exam grade):")
    for c in sorted(set(lab)):
        m = lab == c; sig = Xd[m].mean(0)
        top = np.argsort(np.abs(sig))[::-1][:4]
        tops = "; ".join(f"{'↑' if sig[j] > 0 else '↓'}{DISPLAY[YEAR_DEDUP[j]]}({sig[j]:+.1f})" for j in top)
        L.append(f"  Y{c}: n={int(m.sum())}, mean exam grade {gd[m].mean():.0f} | {tops}")

    # agreement full-k3 vs dedup-k3 (did the grouping change?)
    labF = fcluster(linkage(Xf, "ward"), 3, "maxclust")
    ari = adjusted_rand_score(labF, fcluster(linkage(Xd, "ward"), 3, "maxclust"))
    L.append(f"\nARI(full k=3, dedup k=3) = {ari:.2f}  (1=same grouping, low=de-dup changed the structure)")

    ps = versioned("year_dedup_summary", "txt", CLU); Path(ps).write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"wrote {ps.name}")

    # figure: stability full vs dedup
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(KS, stabF, marker="o", lw=1.4, label="stability — full 14 feats")
    ax.plot(KS, stabD, marker="s", lw=1.4, label="stability — dedup 11 feats")
    ax.set_xlabel("k"); ax.set_ylabel("subsample stability (ARI)"); ax.set_ylim(0, 1)
    ax.set_title("Year clustering stability: full vs de-duplicated features", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    p = versioned("fig_year_dedup_stability", "pdf", FIGS)
    for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig); print(f"fig {p.stem}")

    # figure: dedup k=3 signatures
    M = np.array([Xd[lab == c].mean(0) for c in sorted(set(lab))])
    fig, ax = plt.subplots(figsize=(2 + 0.55 * len(YEAR_DEDUP), 2.6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(YEAR_DEDUP))); ax.set_xticklabels([DISPLAY[c] for c in YEAR_DEDUP], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels([f"Y{c} (grade {gd[lab==c].mean():.0f})" for c in sorted(set(lab))])
    for i in range(3):
        for j in range(len(YEAR_DEDUP)):
            ax.text(j, i, f"{M[i,j]:+.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j]) > 1.2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="mean z")
    ax.set_title("Year (de-dup) k=3 — cluster signatures", fontsize=11)
    p = versioned("fig_year_dedup_signatures", "pdf", FIGS)
    for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig); print(f"fig {p.stem}")


if __name__ == "__main__":
    main()
