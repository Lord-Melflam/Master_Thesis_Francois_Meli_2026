"""35 - The same four k-selection criteria as script 28, but for the OTHER linkage rules.

Kim (v1 chapter 4, page 7): "can you give an explanation of why you think it is that Ward
linkage achieves the desired effect of better balancing the clusters?" Script 28 answers
"which k" for Ward. This script answers "why Ward" with the same evidence: it recomputes
silhouette, subsample stability (ARI), Calinski-Harabasz and Davies-Bouldin across k=2..8
for complete, average and single linkage, on the same de-duplicated, z-scored input.

One figure per alternative linkage (exam and year side by side), built to match
fig_k_selection so the four can be read against each other. Also prints, per linkage, the
size of the largest group at the chosen k, which is the concrete form of "balanced".

Outputs -> clustering/k_selection_by_linkage_summary_v*.txt
           plots/fig_k_selection_{complete,average,single}_v*
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

from common import load_kept, KMAP

LOG = {"median_attempts", "median_mean_delta_sec", "median_edit_size", "median_nloc"}
FEATS_FILE = {"exam": "exam_features_all", "year": "missions_features_all"}
KS = list(range(2, 9))
METHODS = ["ward", "complete", "average", "single"]


def latest(pat): return sorted(glob.glob(str(pat)))[-1]


def versioned(stem, ext, base):
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"{stem}_v{n}.{ext}").exists(): n += 1
    return base / f"{stem}_v{n}.{ext}"


def load(ds):
    """Identical input to script 28: de-dup kept features, log1p on skewed, z-scored."""
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


def stability(Xs, k, method, B=50, frac=0.8):
    rng = np.random.default_rng(0)
    full = fcluster(linkage(Xs, method), k, "maxclust")
    n = len(Xs); m = int(frac * n); a = []
    for _ in range(B):
        idx = np.sort(rng.choice(n, m, replace=False))
        a.append(adjusted_rand_score(full[idx], fcluster(linkage(Xs[idx], method), k, "maxclust")))
    return float(np.mean(a))


def curves(Xs, method):
    Z = linkage(Xs, method)
    sil, stab, ch, db, big = [], [], [], [], []
    for k in KS:
        lab = fcluster(Z, k, "maxclust")
        if len(set(lab)) < 2:          # degenerate cut: every point in one group
            sil.append(np.nan); stab.append(np.nan); ch.append(np.nan); db.append(np.nan)
            big.append(len(lab)); continue
        sil.append(silhouette_score(Xs, lab))
        stab.append(stability(Xs, k, method))
        ch.append(calinski_harabasz_score(Xs, lab))
        db.append(davies_bouldin_score(Xs, lab))
        big.append(int(max(np.bincount(lab)[1:])))
    return dict(n=len(Xs), sil=sil, stab=stab, ch=ch, db=db, biggest=big)


def panel(ax, c, ds):
    ax.plot(KS, c["sil"], marker="o", lw=1.5, color="#0072B2", label="silhouette (higher better)")
    ax.plot(KS, c["stab"], marker="s", lw=1.5, color="#009E73", label="subsample stability, ARI (higher better)")
    mx_ch = np.nanmax(c["ch"]) or 1.0
    mx_db = np.nanmax(c["db"]) or 1.0
    ax.plot(KS, np.array(c["ch"]) / mx_ch, marker="D", lw=1.2, color="#CC79A7",
            ls=":", label="Calinski-Harabasz (higher better, rescaled)")
    ax.plot(KS, np.array(c["db"]) / mx_db, marker="^", lw=1.2, color="#D55E00",
            ls="--", label="Davies-Bouldin (lower better, rescaled)")
    ax.axvline(KMAP[ds], color="0.4", lw=1.2, ls=":", alpha=0.9)
    ax.text(KMAP[ds] + 0.12, 0.03, f"k = {KMAP[ds]}", fontsize=8, color="0.3")
    ax.set_xlabel("number of groups k"); ax.set_ylim(0, 1.02)
    ax.set_title(f"{ds.upper()} (n={c['n']})", fontsize=11)


def main():
    X = {ds: load(ds) for ds in ("exam", "year")}
    res = {m: {ds: curves(X[ds], m) for ds in ("exam", "year")} for m in METHODS}

    L = ["The four k-selection criteria under each linkage rule, same de-dup input as script 28.",
         "'largest group' is the size of the biggest group at that k: the concrete meaning of 'balanced'.", ""]
    for ds in ("exam", "year"):
        L.append(f"=== {ds.upper()} (n={res['ward'][ds]['n']}, chosen k={KMAP[ds]}) ===")
        L.append(f"  {'linkage':>9} {'silhouette':>11} {'stability':>10} {'Calinski-H':>11} {'Davies-B':>9} {'largest group':>14}")
        i = KS.index(KMAP[ds])
        for m in METHODS:
            c = res[m][ds]
            L.append(f"  {m:>9} {c['sil'][i]:11.3f} {c['stab'][i]:10.2f} {c['ch'][i]:11.0f} "
                     f"{c['db'][i]:9.2f} {c['biggest'][i]:9d} / {c['n']}")
        L.append("")
    p = versioned("k_selection_by_linkage_summary", "txt", CLU)
    Path(p).write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"wrote {p.name}")

    for m in [x for x in METHODS if x != "ward"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for ax, ds in zip(axes, ("exam", "year")):
            panel(ax, res[m][ds], ds)
        axes[0].set_ylabel("criterion value")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                   ncol=2, frameon=False, fontsize=9)
        fig.suptitle(f"The same four criteria under {m} linkage", fontsize=12, y=1.02)
        q = versioned(f"fig_k_selection_{m}", "pdf", FIGS)
        for e in ("pdf", "png"): fig.savefig(q.with_suffix("." + e), bbox_inches="tight")
        plt.close(fig); print(f"fig {q.stem}")


if __name__ == "__main__":
    main()
