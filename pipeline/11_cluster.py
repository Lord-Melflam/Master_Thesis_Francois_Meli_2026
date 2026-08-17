"""11 - Behaviour-first clustering (the profiles). Exam cohort.

Clusters students on BEHAVIOUR + code-quality features (score/outcome held ASIDE),
compares K-means vs hierarchical (ward/complete/average/single) across k, picks the best
non-degenerate solution by silhouette, checks stability (bootstrap ARI) and DECOUPLING
from score, then characterises each cluster by its feature signature (for naming).
n clusters = n profiles; named from evidence, not an imposed taxonomy.

Outputs -> data/v2/res_python/clustering/  and figures (PDF+PNG, versioned) -> plots/
"""
import csv, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from common import versioned_path

REPO = Path(__file__).resolve().parents[3]
FEAT = REPO / "data/v2/res_python/features"
OUT = REPO / "data/v2/res_python/clustering"
FIGS = REPO / "data/v2/res_python/plots"

# curated, non-redundant behaviour + code-quality features (cluster ON these)
BEHAV = ["questions_attempted", "median_attempts", "median_fast_retry_ratio",
         "median_long_pause_ratio", "median_mean_delta_sec", "median_time_to_success_sec",
         "median_improving_ratio", "median_edit_size", "churn_ratio", "breakthrough_ratio",
         "median_nloc", "median_comment_ratio", "median_n_concepts"]
LOG = {"median_attempts", "median_mean_delta_sec", "median_time_to_success_sec",
       "median_edit_size", "median_nloc"}
OUTCOME = "median_best_score"   # held aside, for the decoupling check


def latest(stem):
    return sorted(glob.glob(str(FEAT / f"{stem}_v*.csv")))[-1]


def save_fig(fig, stem):
    FIGS.mkdir(parents=True, exist_ok=True)
    n = 1
    while (FIGS / f"{stem}_v{n}.pdf").exists() or (FIGS / f"{stem}_v{n}.png").exists():
        n += 1
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{stem}_v{n}.{ext}", bbox_inches="tight")
    plt.close(fig); print(f"  fig {stem}_v{n}.pdf/.png")


def main():
    rows = [r for r in csv.DictReader(open(latest("exam_features_all")))]
    # complete cases on curated features
    def val(r, c):
        try: return float(r[c])
        except: return np.nan
    keep = [r for r in rows if all(np.isfinite(val(r, c)) for c in BEHAV)]
    X = np.array([[val(r, c) for c in BEHAV] for r in keep], float)
    best = np.array([val(r, OUTCOME) for r in keep])
    hashes = [r["hash"] for r in keep]
    print(f"exam: {len(keep)}/{len(rows)} complete-case students, {len(BEHAV)} features")

    Xt = X.copy()
    for j, c in enumerate(BEHAV):
        if c in LOG: Xt[:, j] = np.log1p(np.clip(Xt[:, j], 0, None))
    Xs = StandardScaler().fit_transform(Xt)

    # ---- compare methods x k ----
    methods = {"kmeans": lambda k: KMeans(k, n_init=10, random_state=0),
               "ward": lambda k: AgglomerativeClustering(k, linkage="ward"),
               "complete": lambda k: AgglomerativeClustering(k, linkage="complete"),
               "average": lambda k: AgglomerativeClustering(k, linkage="average"),
               "single": lambda k: AgglomerativeClustering(k, linkage="single")}
    ks = range(2, 7)
    sils = {m: [] for m in methods}
    minsize = max(3, int(0.03 * len(keep)))   # reject degenerate/singleton clusters
    table, best_cfg = [], None
    for m in methods:
        for k in ks:
            lab = methods[m](k).fit_predict(Xs)
            s = silhouette_score(Xs, lab)
            sils[m].append(s)
            sizes = sorted(np.bincount(lab).tolist(), reverse=True)
            ok = min(sizes) >= minsize
            table.append((m, k, s, sizes, ok))
            if ok and (best_cfg is None or s > best_cfg[2]):
                best_cfg = (m, k, s, sizes)
    print("\nmethod   k  silhouette  sizes            non-degenerate")
    for m, k, s, sizes, ok in table:
        print(f"  {m:8s} {k}  {s:6.3f}   {str(sizes):22s} {'yes' if ok else 'NO'}")
    m, k, s, sizes = best_cfg
    print(f"\nCHOSEN: {m} k={k}  silhouette={s:.3f}  sizes={sizes}")

    lab = methods[m](k).fit_predict(Xs)

    # ---- stability: bootstrap ARI ----
    aris = []
    rng = np.random.default_rng(0)
    for _ in range(10):
        idx = rng.choice(len(keep), len(keep), replace=True)
        u = np.unique(idx)
        l2 = methods[m](k).fit_predict(Xs[u])
        # compare on the shared points via a fresh fit on all vs subset labels is tricky;
        # simple proxy: refit on subset, ARI vs chosen labels restricted to u
        aris.append(adjusted_rand_score(lab[u], l2))
    print(f"stability (bootstrap ARI, mean over 10): {np.mean(aris):.2f}")

    # ---- decoupling from score ----
    from numpy import corrcoef
    # point-biserial-ish: correlation of each cluster indicator with best_score, take max |.|
    cs = max(abs(corrcoef((lab == c).astype(float), best)[0, 1]) for c in set(lab))
    print(f"max |corr(cluster, best_score)| = {cs:.2f}  (lower = more behaviour-driven, less score)")

    # ---- characterise (z-score signature per cluster) ----
    OUT.mkdir(parents=True, exist_ok=True)
    sig = np.array([Xs[lab == c].mean(0) for c in sorted(set(lab))])
    pa = versioned_path(OUT, "exam_cluster_assignments", "csv")
    with open(pa, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["hash", "cluster"])
        for h, c in zip(hashes, lab): w.writerow([h, int(c)])
    ps = versioned_path(OUT, "exam_cluster_summary", "txt")
    L = [f"CHOSEN {m} k={k} sil={s:.3f} sizes={sizes} stability_ARI={np.mean(aris):.2f} max|corr(cluster,score)|={cs:.2f}",
         "per-cluster mean best_score (outcome, aside):"]
    for c in sorted(set(lab)):
        L.append(f"  cluster {c} (n={int((lab==c).sum())}): best_score={best[lab==c].mean():.1f}")
    L.append("\nz-score signature (cluster x feature):")
    L.append("cluster  " + "  ".join(f"{c[:10]:>10}" for c in BEHAV))
    for i, c in enumerate(sorted(set(lab))):
        L.append(f"  {c}     " + "  ".join(f"{v:10.2f}" for v in sig[i]))
    Path(ps).write_text("\n".join(L) + "\n"); print("\n" + "\n".join(L))

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for mm in methods:
        ax.plot(list(ks), sils[mm], marker="o", label=mm)
    ax.set_xlabel("number of clusters k"); ax.set_ylabel("silhouette")
    ax.set_title("Cluster quality: K-means vs hierarchical linkages", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    save_fig(fig, "fig_cluster_silhouette_comparison")

    fig, ax = plt.subplots(figsize=(10, 0.7 + 0.5 * k))
    im = ax.imshow(sig, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(BEHAV))); ax.set_xticklabels(BEHAV, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(k)); ax.set_yticklabels([f"C{c} (n={int((lab==c).sum())})" for c in sorted(set(lab))])
    for i in range(k):
        for j in range(len(BEHAV)):
            ax.text(j, i, f"{sig[i,j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(sig[i, j]) > 1.2 else "black")
    fig.colorbar(im, ax=ax, label="mean z-score", shrink=0.7)
    ax.set_title(f"Behavioural profile signatures ({m}, k={k}) — score held aside", fontsize=12)
    save_fig(fig, "fig_cluster_profile_signatures")


if __name__ == "__main__":
    main()
