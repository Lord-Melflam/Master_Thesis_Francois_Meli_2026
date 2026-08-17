"""14 - Behaviour-first clustering on the COURSEWORK (year), LINKED cohort.

Same method as the exam (13) but on the Q1 mission behaviour, and the payoff cross-tab is
against the EXAM score quartile: does how a student worked during the year relate to how
they did in the exam? Cohort = the 569 students in both, staff removed.

Same rules: cluster on behaviour + code-quality (score aside); compare every method x k;
read the Ward tree level by level; reader-friendly names; neutral wording.

Outputs -> data/v2/res_python/clustering/ ; figures -> plots/ (versioned)
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

DISPLAY = {
    "questions_attempted": "questions attempted (year)",
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
    "active_weeks": "active weeks in the semester",
    "active_days": "active days in the semester",
}
BEHAV = list(DISPLAY)
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
    idx = sorted(range(len(BEHAV)), key=lambda j: -abs(sig[j]))[:top]
    return "; ".join(f"{'higher' if sig[j] > 0 else 'lower'}-than-average {DISPLAY[BEHAV[j]]}" for j in idx)


def main():
    linked = {r["hash"] for r in csv.DictReader(open(latest("linked_students", LINK)))}
    feats = {r["hash"]: r for r in csv.DictReader(open(latest("missions_features_all")))}
    grades = {r["hash"]: float(r["exam_grade"]) for r in csv.DictReader(open(latest("exam_score_categories", CLU)))}
    hashes = [h for h in feats if h in linked and h in grades]
    print(f"LINKED cohort on coursework: {len(hashes)} students")

    # EXAM score quartile (recomputed within cohort) — the OUTCOME to relate year behaviour to
    g = np.array([grades[h] for h in hashes])
    qb = np.percentile(g, [25, 50, 75]); qn = ["Q1_lowest", "Q2", "Q3", "Q4_highest"]
    exq = np.array([qn[int(np.searchsorted(qb, grades[h], side="right"))] for h in hashes])

    def val(h, c):
        try: return float(feats[h][c])
        except: return np.nan
    X = np.nan_to_num(np.array([[val(h, c) for c in BEHAV] for h in hashes], float), nan=0.0)
    Xt = X.copy()
    for j, c in enumerate(BEHAV):
        if c in LOG: Xt[:, j] = np.log1p(np.clip(Xt[:, j], 0, None))
    Xs = StandardScaler().fit_transform(Xt)

    minsize = max(3, int(0.03 * len(hashes)))
    sil = {"kmeans": []}; [sil.setdefault(lk, []) for lk in LINKAGES]
    tbl, best = [], None
    for k in KS:
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(Xs)
        s = silhouette_score(Xs, lab); sil["kmeans"].append(s)
        sizes = sorted(np.bincount(lab).tolist(), reverse=True); ok = min(sizes) >= minsize
        tbl.append(("kmeans", k, s, sizes, ok)); best = best if not ok or (best and s <= best[0]) else (s, "kmeans", k)
        for lk in LINKAGES:
            lab = AgglomerativeClustering(k, linkage=lk).fit_predict(Xs)
            s = silhouette_score(Xs, lab); sil[lk].append(s)
            sizes = sorted(np.bincount(lab).tolist(), reverse=True); ok = min(sizes) >= minsize
            tbl.append((lk, k, s, sizes, ok))
            if ok and (best is None or s > best[0]): best = (s, lk, k)
    print("\nmethod    k  silhouette  sizes                 non-degen")
    for m, k, s, sizes, ok in tbl:
        print(f"  {m:8s} {k}  {s:6.3f}  {str(sizes):24s} {'yes' if ok else 'NO'}")
    print(f"chosen: {best[1]} k={best[2]} sil={best[0]:.3f}")

    Z = linkage(Xs, method="ward")
    L = [f"COURSEWORK clustering — LINKED cohort {len(hashes)} students.",
         "features: " + ", ".join(DISPLAY.values()),
         f"chosen by silhouette: {best[1]} k={best[2]} (sil={best[0]:.3f})", ""]
    for k in (2, 3, 4):
        lab = fcluster(Z, t=k, criterion="maxclust")
        L.append(f"=== Ward cut into {k} groups (year behaviour) ===")
        for c in sorted(set(lab)):
            m = lab == c
            L.append(f"  group {c}: {int(m.sum())} students, mean EXAM grade {g[m].mean():.0f}")
            L.append(f"     {describe(Xs[m].mean(0))}")
            L.append(f"     exam-quartile mix: {{{', '.join(f'{q}:{int((exq[m]==q).sum())}' for q in qn)}}}")
        L.append("")

    lab = (fcluster(Z, t=best[2], criterion="maxclust") if best[1] == "ward"
           else KMeans(best[2], n_init=10, random_state=0).fit_predict(Xs) if best[1] == "kmeans"
           else AgglomerativeClustering(best[2], linkage=best[1]).fit_predict(Xs))
    cl = sorted(set(lab))
    ct = np.array([[int(((lab == c) & (exq == q)).sum()) for q in qn] for c in cl])
    L.append("cross-tab  year-behaviour group x EXAM score-quartile (counts):")
    L.append("          " + "  ".join(f"{q:>10}" for q in qn))
    for i, c in enumerate(cl):
        L.append(f"  group {c} " + "  ".join(f"{ct[i,j]:10d}" for j in range(4)))

    CLU.mkdir(parents=True, exist_ok=True)
    ps = versioned_path(CLU, "missions_cluster_summary", "txt"); Path(ps).write_text("\n".join(L) + "\n")
    pa = versioned_path(CLU, "missions_cluster_assignments", "csv")
    with open(pa, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["hash", "group", "exam_grade", "exam_quartile"])
        for h, c, gg, q in zip(hashes, lab, g, exq): w.writerow([h, int(c), round(gg, 1), q])
    print("\n".join(L)); print(f"wrote {ps.name}, {pa.name}")

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for m in ["kmeans"] + LINKAGES:
        ax.plot(KS, sil[m], marker="o", ms=4, lw=1.2, label=m)
    ax.set_xlabel("number of groups k"); ax.set_ylabel("silhouette")
    ax.set_title("Coursework clustering — quality across methods and k", fontsize=12)
    ax.legend(frameon=False, fontsize=9, ncol=2); savefig(fig, "fig_missions_method_comparison")

    fig, ax = plt.subplots(figsize=(11, 4.2))
    dendrogram(Z, no_labels=True, color_threshold=Z[-(best[2] - 1), 2], ax=ax)
    for coll in ax.collections: coll.set_linewidth(0.4)
    ax.set_title(f"Coursework (year) behaviour — hierarchical tree (Ward), cut into {best[2]} groups", fontsize=12)
    ax.set_ylabel("merge distance"); savefig(fig, "fig_missions_dendrogram")

    fig, ax = plt.subplots(figsize=(6.5, 0.9 + 0.5 * len(cl)))
    ctn = ct / ct.sum(1, keepdims=True)
    im = ax.imshow(ctn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["lowest", "Q2", "Q3", "highest"])
    ax.set_yticks(range(len(cl))); ax.set_yticklabels([f"group {c}" for c in cl])
    for i in range(len(cl)):
        for j in range(4):
            ax.text(j, i, f"{ctn[i,j]:.0%}", ha="center", va="center",
                    color="white" if ctn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("EXAM score quartile"); fig.colorbar(im, ax=ax, shrink=0.7, label="share of group")
    ax.set_title("Year-behaviour group vs EXAM outcome", fontsize=12); savefig(fig, "fig_missions_group_vs_exam")


if __name__ == "__main__":
    main()
