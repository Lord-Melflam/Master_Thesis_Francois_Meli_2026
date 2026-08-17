"""24 - Annotated Ward dendrogram: interpret ON the diagram (31-07 ask).

"Aller plus loin que juste inclure l'image; interpreter ce qu'on voit sur le diagramme,
mettre un petit rectangle / small grouping per colour."

For BOTH datasets, on the de-duplicated clustering input from script 22, draw the Ward
dendrogram of the 569 students, colour it AT the k=3 cut (three coloured groups), draw the
horizontal cut line, and overlay a labelled rectangle under each cluster block giving its
id, size and mean exam grade (E1/E2/E3, Y1/Y2/Y3 ordered by ascending grade). This turns a
bare tree into a read: "where are the groups, how big, and how do they relate to outcome".

Outputs (versioned PDF+PNG) -> data/v2/res_python/plots/fig_{ds}_dendro_annotated_v*
"""
import csv, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
FEAT = REPO / "data/v2/res_python/features"
LINK = REPO / "data/v2/res_python/linkage"
CLU = REPO / "data/v2/res_python/clustering"
FIGS = REPO / "data/v2/res_python/plots"

LOG = {"median_attempts", "median_mean_delta_sec", "median_edit_size", "median_nloc"}
# de-dup clustering inputs (script 22)
FEATS_FILE = {"exam": "exam_features_all", "year": "missions_features_all"}
from common import load_kept, KMAP   # kept set (written by 22) + k; single source of truth
DISPLAY = {
    "questions_attempted": "questions attempted", "median_attempts": "attempts per question",
    "median_fast_retry_ratio": "quick resubmissions", "median_long_pause_ratio": "long pauses",
    "median_mean_delta_sec": "gap between tries",
    "median_improving_ratio": "tries that improved", "median_edit_size": "lines changed / edit",
    "churn_ratio": "edits, no score gain", "breakthrough_ratio": "small fix, big gain",
    "median_nloc": "code lines",
    "median_comment_ratio": "comment share", "median_n_concepts": "distinct concepts",
    "active_weeks": "active weeks", "active_days": "active days",
}


def block_lines(sig, cols, gmean, gmed, n=4):
    """Per-cluster info block: top-n features with SIGNED z-value (SD from cohort mean) + grade."""
    idx = np.argsort(np.abs(sig))[::-1][:n]
    lines = [f"{'▲' if sig[j] > 0 else '▼'} {DISPLAY.get(cols[j], cols[j])}  {sig[j]:+.1f}" for j in idx]
    lines.append(f"exam grade  mean {gmean:.0f} · med {gmed:.0f}")
    return lines


def draw_blocks(fig, order, color, size, sig, cols, gmean, gmed, ds):
    """One separate info block per cluster, laid out in a row beneath the figure.

    `order` is the left-to-right order the clusters appear in the tree, NOT E1..Ek.
    Kim (v2, scan p12) drew two arrows between the E4 block in the tree and the E4
    box below and asked "why crossed?": the names run by rising mean grade while the
    tree lays the blocks out by its own merges, so the two orders do not coincide.
    Laying the boxes out in tree order removes the crossing.
    """
    k = len(order)
    y0 = -0.04
    for i, c in enumerate(order):
        x = (i + 0.5) / k
        fig.text(x, y0, f"{ds[0].upper()}{c}  (n={size[c]})", ha="center", va="top",
                 fontsize=9, fontweight="bold", color=color[c])
        body = block_lines(sig[c], cols, gmean[c], gmed[c])
        for j, ln in enumerate(body):
            fig.text(x, y0 - 0.045 * (j + 1), ln, ha="center", va="top",
                     fontsize=7.5, family="monospace",
                     color="#333333" if j < len(body) - 1 else "#666666")


def latest(pat): return sorted(glob.glob(str(pat)))[-1]


def versioned(stem):
    FIGS.mkdir(parents=True, exist_ok=True)
    n = 1
    while (FIGS / f"{stem}_v{n}.pdf").exists() or (FIGS / f"{stem}_v{n}.png").exists():
        n += 1
    return n


def load(ds):
    cols = load_kept(ds)
    linked = {r["hash"] for r in csv.DictReader(open(latest(LINK / "linked_students_v*.csv")))}
    feats = {r["hash"]: r for r in csv.DictReader(open(latest(FEAT / f"{FEATS_FILE[ds]}_v*.csv")))}
    grades = {r["hash"]: float(r["exam_grade"]) for r in csv.DictReader(open(latest(CLU / "exam_score_categories_v*.csv")))}
    hashes = [h for h in feats if h in linked and h in grades]

    def val(h, c):
        try: return float(feats[h][c])
        except: return np.nan
    X = np.nan_to_num(np.array([[val(h, c) for c in cols] for h in hashes], float), nan=0.0)
    for j, c in enumerate(cols):
        if c in LOG: X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    Xs = StandardScaler().fit_transform(X)
    grade = np.array([grades[h] for h in hashes])
    return Xs, grade, cols


def annotated(ds):
    k = KMAP[ds]
    Xs, grade, cols = load(ds)
    Z = linkage(Xs, method="ward")
    lab = fcluster(Z, k, "maxclust")
    # grade-ordered ids: 1..k by ascending mean exam grade
    order = {c: i + 1 for i, c in enumerate(sorted(set(lab), key=lambda c: grade[lab == c].mean()))}
    lab = np.array([order[c] for c in lab])
    gmean = {c: grade[lab == c].mean() for c in sorted(set(lab))}
    gmed = {c: float(np.median(grade[lab == c])) for c in sorted(set(lab))}
    size = {c: int((lab == c).sum()) for c in sorted(set(lab))}
    sig = {c: Xs[lab == c].mean(0) for c in sorted(set(lab))}   # z-signature per cluster

    ct = Z[-(k - 1), 2] - 1e-9          # colour threshold = just below the k->k-1 merge
    fig, ax = plt.subplots(figsize=(11, 5))
    dn = dendrogram(Z, no_labels=True, color_threshold=ct,
                    above_threshold_color="#BBBBBB", ax=ax)
    for coll in ax.collections:
        coll.set_linewidth(0.5)
    ax.axhline(ct, color="#444444", ls="--", lw=0.8)
    ax.text(ax.get_xlim()[1], ct, f"  k={k} cut", va="center", ha="left", fontsize=8, color="#444444")

    # contiguous runs of the same cluster label along the leaf order = the coloured blocks
    leaves = dn["leaves"]
    lab_ordered = lab[leaves]
    x = 5.0  # scipy places leaves at 5, 15, 25, ...
    runs = []
    start = 0
    for i in range(1, len(lab_ordered) + 1):
        if i == len(lab_ordered) or lab_ordered[i] != lab_ordered[start]:
            runs.append((lab_ordered[start], start, i - 1))
            start = i
    ymax = ax.get_ylim()[1]
    band = ymax * 0.06
    leaf_colors = dn["leaves_color_list"]   # scipy's own link colour per leaf (leaf order)
    cluster_color = {}
    for cid, i0, i1 in runs:
        x0 = 10 * i0 + 0.5
        x1 = 10 * i1 + 9.5
        col = leaf_colors[i0]                # match the tree's colour for this block
        cluster_color[cid] = col
        ax.add_patch(Rectangle((x0, -band), x1 - x0, band, clip_on=False,
                               facecolor=col, alpha=0.25, edgecolor=col, lw=1.0))
        ax.text((x0 + x1) / 2, -band / 2, f"{ds[0].upper()}{cid}\nn={size[cid]}",
                ha="center", va="center", fontsize=8, color="black")
    ax.set_ylim(-band * 1.1, ymax)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("merge distance (Ward)")
    ax.set_title(f"{ds.upper()} behavior: Ward dendrogram, coloured at the k={k} cut (n={Xs.shape[0]})\n"
                 "each cluster is described below by its feature signature; the up/down value is how far its "
                 "students sit from the whole-group average, in standard deviations",
                 fontsize=10)
    tree_order = list(dict.fromkeys(int(r[0]) for r in runs))   # clusters left to right in the tree
    draw_blocks(fig, tree_order, cluster_color, size, sig, cols, gmean, gmed, ds)
    fig.text(0.5, -0.04 - 0.045 * 6, "Clusters are formed from the behavior features only; the exam grade "
             "(mean final score over the 6 exam questions, 0 to 100) is shown only to relate the groups to the outcome.",
             ha="center", va="top", fontsize=7.5, style="italic", color="#666666")
    v = versioned(f"fig_{ds}_dendro_annotated")
    for e in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_{ds}_dendro_annotated_v{v}.{e}", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_{ds}_dendro_annotated_v{v}  | " +
          "  ".join(f"{ds[0].upper()}{c}: n={size[c]}, grade {gmean[c]:.0f}/{gmed[c]:.0f}" for c in sorted(size)))


if __name__ == "__main__":
    for ds in ("exam", "year"):
        annotated(ds)
    print("done.")
