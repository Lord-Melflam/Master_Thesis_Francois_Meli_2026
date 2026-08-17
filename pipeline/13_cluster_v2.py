"""13 - Behaviour-first clustering (exam), on the LINKED cohort. Follows Kim's notes.

Cohort: the students present in BOTH the year and the exam, staff/outliers removed
(data/v2/res_python/linkage/linked_students_v*.csv). Same population everywhere so the
year<->exam comparison stays valid.

- Clusters on BEHAVIOUR + code-quality only (score held aside). All students kept
  (iteration features imputed to 0 = "that behaviour did not occur"; time-to-success dropped).
- Compares EVERY method x many k: k-means + hierarchical (ward/complete/average/single),
  k = 2..8, by silhouette (+ cluster sizes to expose degenerate splits).
- Reads the Ward hierarchy level by level (k=2,3,4): per cluster, size, mean exam grade,
  score-quartile mix, and the features that are higher/lower than average (preponderance).
- Cross-tabs clusters vs the balanced score quartiles (recomputed within this cohort).
- Reader-friendly feature names; neutral, descriptive cluster wording (no loaded labels).

Outputs -> data/v2/res_python/clustering/ ; figures -> plots/ (versioned PDF+PNG)
"""
import csv, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from common import versioned_path

REPO = Path(__file__).resolve().parents[3]
FEAT = REPO / "data/v2/res_python/features"
CLU = REPO / "data/v2/res_python/clustering"
LINK = REPO / "data/v2/res_python/linkage"
FIGS = REPO / "data/v2/res_python/plots"

# feature -> reader-friendly display name (no ambiguous terms)
DISPLAY = {
    "questions_attempted": "questions attempted",
    "median_attempts": "attempts per question",
    "median_fast_retry_ratio": "quick resubmissions (share)",
    "median_long_pause_ratio": "long pauses (share)",
    "median_mean_delta_sec": "typical gap between tries (s)",
    "median_improving_ratio": "tries that improved the score (share)",
    "median_edit_size": "lines changed per edit",
    "churn_ratio": "edits with no score gain (share)",
    "breakthrough_ratio": "small change, big score gain (share)",
    "median_nloc": "code lines (non-comment)",
    "median_comment_ratio": "comment lines (share)",
    "median_n_concepts": "distinct concepts used",
}
BEHAV = list(DISPLAY)
DISP = [DISPLAY[c] for c in BEHAV]
LOG = {"median_attempts", "median_mean_delta_sec", "median_edit_size", "median_nloc"}
LINKAGES = ["ward", "complete", "average", "single"]
KS = list(range(2, 9))


def latest(stem, base=FEAT):
    return sorted(glob.glob(str(base / f"{stem}_v*.csv")))[-1]


def savefig(fig, stem):
    FIGS.mkdir(parents=True, exist_ok=True)
    n = 1
    while (FIGS / f"{stem}_v{n}.pdf").exists() or (FIGS / f"{stem}_v{n}.png").exists():
        n += 1
    for e in ("pdf", "png"):
        fig.savefig(FIGS / f"{stem}_v{n}.{e}", bbox_inches="tight")
    plt.close(fig); print(f"  fig {stem}_v{n}")


def describe(sig, top=3):
    """neutral wording from a cluster's z-signature."""
    idx = sorted(range(len(BEHAV)), key=lambda j: -abs(sig[j]))[:top]
    parts = [f"{'higher' if sig[j] > 0 else 'lower'}-than-average {DISPLAY[BEHAV[j]]}" for j in idx]
    return "; ".join(parts)


def main():
    linked = {r["hash"] for r in csv.DictReader(open(latest("linked_students", LINK)))}
    feats = {r["hash"]: r for r in csv.DictReader(open(latest("exam_features_all")))}
    grades = {r["hash"]: float(r["exam_grade"]) for r in csv.DictReader(open(latest("exam_score_categories", CLU)))}
    hashes = [h for h in feats if h in linked and h in grades]
    print(f"LINKED cohort (year & exam, staff removed): {len(hashes)} students")

    # recompute balanced quartiles WITHIN this cohort
    g = np.array([grades[h] for h in hashes])
    qb = np.percentile(g, [25, 50, 75])
    qnames = ["Q1_lowest", "Q2", "Q3", "Q4_highest"]
    quart = np.array([qnames[int(np.searchsorted(qb, grades[h], side="right"))] for h in hashes])

    def val(h, c):
        try: return float(feats[h][c])
        except: return np.nan
    X = np.nan_to_num(np.array([[val(h, c) for c in BEHAV] for h in hashes], float), nan=0.0)
    Xt = X.copy()
    for j, c in enumerate(BEHAV):
        if c in LOG: Xt[:, j] = np.log1p(np.clip(Xt[:, j], 0, None))
    Xs = StandardScaler().fit_transform(Xt)

    # ---- full comparison: method x k ----
    minsize = max(3, int(0.03 * len(hashes)))
    sil = {"kmeans": []}
    for lk in LINKAGES: sil[lk] = []
    rows_tbl, best = [], None
    for k in KS:
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(Xs)
        s = silhouette_score(Xs, lab); sil["kmeans"].append(s)
        sizes = sorted(np.bincount(lab).tolist(), reverse=True)
        ok = min(sizes) >= minsize; rows_tbl.append(("kmeans", k, s, sizes, ok))
        if ok and (best is None or s > best[0]): best = (s, "kmeans", k)
        for lk in LINKAGES:
            lab = AgglomerativeClustering(k, linkage=lk).fit_predict(Xs)
            s = silhouette_score(Xs, lab); sil[lk].append(s)
            sizes = sorted(np.bincount(lab).tolist(), reverse=True)
            ok = min(sizes) >= minsize; rows_tbl.append((lk, k, s, sizes, ok))
            if ok and (best is None or s > best[0]): best = (s, lk, k)
    print("\nmethod    k  silhouette  sizes                    non-degenerate")
    for m, k, s, sizes, ok in rows_tbl:
        print(f"  {m:8s} {k}  {s:6.3f}  {str(sizes):26s} {'yes' if ok else 'NO'}")
    print(f"chosen: {best[1]} k={best[2]} sil={best[0]:.3f}")

    # ---- Ward hierarchy, level by level ----
    Z = linkage(Xs, method="ward")
    L = [f"EXAM clustering — LINKED cohort {len(hashes)} students.",
         "features: " + ", ".join(DISP),
         f"chosen by silhouette: {best[1]} k={best[2]} (sil={best[0]:.3f})", ""]
    for k in (2, 3, 4):
        lab = fcluster(Z, t=k, criterion="maxclust")
        L.append(f"=== Ward cut into {k} groups ===")
        for c in sorted(set(lab)):
            m = lab == c
            sig = Xs[m].mean(0)
            qd = {q: int((quart[m] == q).sum()) for q in qnames}
            L.append(f"  group {c}: {int(m.sum())} students, mean exam grade {g[m].mean():.0f}")
            L.append(f"     {describe(sig)}")
            L.append(f"     score-quartile mix: {qd}")
        L.append("")

    # chosen labels + cross-tab
    lab = (fcluster(Z, t=best[2], criterion="maxclust") if best[1] == "ward"
           else KMeans(best[2], n_init=10, random_state=0).fit_predict(Xs) if best[1] == "kmeans"
           else AgglomerativeClustering(best[2], linkage=best[1]).fit_predict(Xs))
    cl = sorted(set(lab))
    ct = np.array([[int(((lab == c) & (quart == q)).sum()) for q in qnames] for c in cl])
    L.append("cross-tab  group x exam score-quartile (counts):")
    L.append("          " + "  ".join(f"{q:>10}" for q in qnames))
    for i, c in enumerate(cl):
        L.append(f"  group {c} " + "  ".join(f"{ct[i,j]:10d}" for j in range(4)))
    L.append("  (a group spread across quartiles => behaviour partly independent of grade)")

    CLU.mkdir(parents=True, exist_ok=True)
    ps = versioned_path(CLU, "exam_cluster_v2_summary", "txt"); Path(ps).write_text("\n".join(L) + "\n")
    pa = versioned_path(CLU, "exam_cluster_v2_assignments", "csv")
    with open(pa, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["hash", "group", "exam_grade", "score_quartile"])
        for h, c, gg, q in zip(hashes, lab, g, quart): w.writerow([h, int(c), round(gg, 1), q])
    print("\n".join(L)); print(f"wrote {ps.name}, {pa.name}")

    # ---- figures ----
    # comparison: silhouette vs k, every method (thin lines)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for m in ["kmeans"] + LINKAGES:
        ax.plot(KS, sil[m], marker="o", ms=4, lw=1.2, label=m)
    ax.set_xlabel("number of groups k"); ax.set_ylabel("silhouette (higher = better separated)")
    ax.set_title("Cluster quality across methods and k", fontsize=12)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    savefig(fig, "fig_exam_method_comparison")

    # dendrogram, FINE lines
    fig, ax = plt.subplots(figsize=(11, 4.2))
    dendrogram(Z, no_labels=True, color_threshold=Z[-(best[2] - 1), 2], ax=ax)
    for coll in ax.collections:
        coll.set_linewidth(0.4)
    ax.set_title(f"Exam behaviour — hierarchical tree (Ward), cut into {best[2]} groups", fontsize=12)
    ax.set_ylabel("merge distance")
    savefig(fig, "fig_exam_dendrogram")

    # cross-tab heatmap
    fig, ax = plt.subplots(figsize=(6.5, 0.9 + 0.5 * len(cl)))
    ctn = ct / ct.sum(1, keepdims=True)
    im = ax.imshow(ctn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["lowest", "Q2", "Q3", "highest"])
    ax.set_yticks(range(len(cl))); ax.set_yticklabels([f"group {c}" for c in cl])
    for i in range(len(cl)):
        for j in range(4):
            ax.text(j, i, f"{ctn[i,j]:.0%}", ha="center", va="center",
                    color="white" if ctn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("exam score quartile"); fig.colorbar(im, ax=ax, shrink=0.7, label="share of group")
    ax.set_title("Behaviour group vs exam score quartile", fontsize=12)
    savefig(fig, "fig_exam_cluster_vs_quartile")


if __name__ == "__main__":
    main()
