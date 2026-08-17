"""28 - Rigorous k-selection on the DE-DUP clustering input (31-07/W6 ask: "year looks
like k=3, exam like k=5 - but check rigorously").

Four criteria across k=2..8, both datasets (Ward, z-scored de-dup features from 22):
  - silhouette                (higher better)  - separation/compactness
  - subsample stability (ARI) (higher better)  - reproducibility under 80% resampling
  - Calinski-Harabasz         (higher better)  - between/within variance ratio
  - Davies-Bouldin            (lower  better)  - avg cluster-pair similarity
No single index is trusted alone; we report the whole curve + a rank-aggregated hint, and
let the human pick. The point is to CONFIRM OR REFUTE the visual impression, not to trust
one number.

Outputs -> clustering/k_selection_summary_v*.txt ; figures -> plots/fig_k_selection_v*
"""
import csv, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             calinski_harabasz_score, davies_bouldin_score)

REPO = Path(__file__).resolve().parents[3]
FEAT = REPO / "data/v2/res_python/features"
LINK = REPO / "data/v2/res_python/linkage"
CLU = REPO / "data/v2/res_python/clustering"
FIGS = REPO / "data/v2/res_python/plots"

from common import load_kept, KMAP   # kept feature set written by script 22 (no hardcoded list) + chosen k

LOG = {"median_attempts", "median_mean_delta_sec", "median_edit_size", "median_nloc"}
FEATS_FILE = {"exam": "exam_features_all", "year": "missions_features_all"}
KS = list(range(2, 9))


def latest(pat): return sorted(glob.glob(str(pat)))[-1]


def versioned(stem, ext, base):
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"{stem}_v{n}.{ext}").exists(): n += 1
    return base / f"{stem}_v{n}.{ext}"


def load(ds):
    cols = load_kept(ds)
    linked = {r["hash"] for r in csv.DictReader(open(latest(LINK / "linked_students_v*.csv")))}
    feats = {r["hash"]: r for r in csv.DictReader(open(latest(FEAT / f"{FEATS_FILE[ds]}_v*.csv")))}
    grades = {r["hash"] for r in csv.DictReader(open(latest(CLU / "exam_score_categories_v*.csv")))}
    hashes = [h for h in feats if h in linked and h in grades]

    def val(h, c):
        try: return float(feats[h][c])
        except: return np.nan
    X = np.nan_to_num(np.array([[val(h, c) for c in cols] for h in hashes], float), nan=0.0)
    for j, c in enumerate(cols):
        if c in LOG: X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    return StandardScaler().fit_transform(X)


def stability(Xs, k, B=50, frac=0.8):
    rng = np.random.default_rng(0)
    full = fcluster(linkage(Xs, "ward"), k, "maxclust")
    n = len(Xs); m = int(frac * n); a = []
    for _ in range(B):
        idx = np.sort(rng.choice(n, m, replace=False))
        a.append(adjusted_rand_score(full[idx], fcluster(linkage(Xs[idx], "ward"), k, "maxclust")))
    return float(np.mean(a))


def curves(ds):
    Xs = load(ds)
    Z = linkage(Xs, "ward")
    sil, stab, ch, db = [], [], [], []
    for k in KS:
        lab = fcluster(Z, k, "maxclust")
        sil.append(silhouette_score(Xs, lab))
        stab.append(stability(Xs, k))
        ch.append(calinski_harabasz_score(Xs, lab))
        db.append(davies_bouldin_score(Xs, lab))
    return dict(n=len(Xs), sil=sil, stab=stab, ch=ch, db=db)


def rank_hint(c):
    """Rank-aggregate the 4 criteria (higher-better for sil/stab/ch, lower-better for db).
    Returns the k with the best (lowest) mean rank."""
    def ranks(vals, higher_better=True):
        order = np.argsort(vals)[::-1] if higher_better else np.argsort(vals)
        r = np.empty(len(vals), int)
        for pos, i in enumerate(order): r[i] = pos
        return r
    R = (ranks(c["sil"]) + ranks(c["stab"]) + ranks(c["ch"]) + ranks(c["db"], higher_better=False))
    return KS[int(np.argmin(R))], R


def main():
    res = {ds: curves(ds) for ds in ("exam", "year")}
    L = ["Rigorous k-selection on the DE-DUP clustering input (Ward). k=2..8, 569 students.",
         "sil/stab/CH higher = better; DB lower = better. No single index trusted alone.", ""]
    for ds in ("exam", "year"):
        c = res[ds]; best, R = rank_hint(c)
        L.append(f"=== {ds.upper()} (n={c['n']}) ===")
        L.append(f"  {'k':>2}  {'silhouette':>10} {'stability':>10} {'Calinski-H':>11} {'Davies-B':>9} {'rank-sum':>9}")
        for i, k in enumerate(KS):
            L.append(f"  {k:>2}  {c['sil'][i]:10.3f} {c['stab'][i]:10.2f} {c['ch'][i]:11.0f} {c['db'][i]:9.2f} {R[i]:9d}")
        L.append(f"  best silhouette: k={KS[int(np.argmax(c['sil']))]}; best stability: k={KS[int(np.argmax(c['stab']))]}; "
                 f"best CH: k={KS[int(np.argmax(c['ch']))]}; best DB: k={KS[int(np.argmin(c['db']))]}")
        L.append(f"  --> rank-aggregated hint: k={best}  (a hint, not a verdict; read the curve)\n")
    ps = versioned("k_selection_summary", "txt", CLU); Path(ps).write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"wrote {ps.name}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, ds in zip(axes, ("exam", "year")):
        c = res[ds]
        ax.plot(KS, c["sil"], marker="o", lw=1.5, color="#0072B2", label="silhouette (higher better)")
        ax.plot(KS, c["stab"], marker="s", lw=1.5, color="#009E73", label="subsample stability, ARI (higher better)")
        ax.plot(KS, np.array(c["ch"]) / max(c["ch"]), marker="D", lw=1.2, color="#CC79A7",
                ls=":", label="Calinski-Harabasz (higher better, rescaled)")
        ax.plot(KS, np.array(c["db"]) / max(c["db"]), marker="^", lw=1.2, color="#D55E00",
                ls="--", label="Davies-Bouldin (lower better, rescaled)")
        ax.axvline(KMAP[ds], color="0.4", lw=1.2, ls=":", alpha=0.9)
        ax.text(KMAP[ds] + 0.12, 0.03, f"chosen k = {KMAP[ds]}", fontsize=8, color="0.3")
        ax.set_xlabel("number of groups k"); ax.set_ylim(0, 1.02)
        ax.set_title(f"{ds.upper()} (n={c['n']})", fontsize=11)
    axes[0].set_ylabel("criterion value")
    # one shared legend below the panels, so it never sits on top of the curves
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.01),
               ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Choosing the number of groups on the cleaned features (Ward linkage)",
                 fontsize=12, y=1.02)
    p = versioned("fig_k_selection", "pdf", FIGS)
    for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig); print(f"fig {p.stem}")


if __name__ == "__main__":
    main()
