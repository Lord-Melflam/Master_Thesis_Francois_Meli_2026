"""27 - Annotated Ward CLUSTERMAP (companion to the annotated dendrogram, script 24).

Same de-duplicated clustering input and same k=3 cut as 24, but the clustermap view: a
z-scored heatmap of de-dup features (rows) x students (columns), Ward-clustered on both
axes, with a colour band + legend over the columns marking the profile groups
(E1/E2/E3, Y1/Y2/Y3 by ascending mean exam grade). Where the dendrogram shows only *where*
the groups split, the clustermap also shows *which features* run high/low inside each block
(the visual cluster signature) -> lets us compare year vs exam structure directly.

Outputs (versioned PDF+PNG) -> data/v2/res_python/plots/fig_{exam,year}_clustermap_annotated_v*
"""
import csv, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
FEAT = REPO / "data/v2/res_python/features"
LINK = REPO / "data/v2/res_python/linkage"
CLU = REPO / "data/v2/res_python/clustering"
FIGS = REPO / "data/v2/res_python/plots"

DISPLAY = {
    "questions_attempted": "questions attempted", "median_attempts": "attempts per question",
    "median_fast_retry_ratio": "quick resubmissions", "median_long_pause_ratio": "long pauses",
    "median_mean_delta_sec": "gap between tries", "median_improving_ratio": "tries that improved",
    "median_edit_size": "lines changed / edit", "churn_ratio": "edits, no score gain",
    "breakthrough_ratio": "small fix, big gain", "median_nloc": "code lines",
    "median_comment_ratio": "comment share", "median_n_concepts": "distinct concepts",
    "active_weeks": "active weeks", "active_days": "active days",
}
LOG = {"median_attempts", "median_mean_delta_sec", "median_edit_size", "median_nloc"}
FEATS_FILE = {"exam": "exam_features_all", "year": "missions_features_all"}
# grade-ordered cluster palette (Okabe-Ito): low grade -> high grade
CLUSTER_COLORS = {1: "#D55E00", 2: "#E69F00", 3: "#009E73", 4: "#0072B2"}
from common import load_kept, KMAP   # kept set (written by 22) + k; single source of truth

# Kim (v1 chapter 4, pages 11 and 12): "could you overlay small rectangles over the heatmap
# as visual clues to highlight what you are saying here?" Each (group, feature) pair below
# is a band the report text names in prose; the box puts the reader's eye on it.
HIGHLIGHTS = {
    "exam": [(1, "edits, no score gain"), (1, "distinct concepts"),
             (2, "code lines"), (2, "distinct concepts"),
             (4, "small fix, big gain")],
    "year": [(1, "questions attempted"),
             (2, "quick resubmissions"), (2, "attempts per question"),
             (3, "questions attempted"), (3, "active weeks"), (3, "code lines")],
}


def overlay(g, ds, lab, row_names):
    """Box each (group x feature) band the text calls out, in the clustermap's own order."""
    ax = g.ax_heatmap
    rows = [row_names[i] for i in g.dendrogram_row.reordered_ind]
    lab_ord = lab[g.dendrogram_col.reordered_ind]
    for c, feat in HIGHLIGHTS.get(ds, []):
        if feat not in rows:
            print(f"  ! skipped {ds} {ds[0].upper()}{c} x '{feat}': not in the kept feature set")
            continue
        pos = np.flatnonzero(lab_ord == c)
        if pos.size == 0:
            print(f"  ! skipped {ds} group {c}: no students")
            continue
        x0, x1 = int(pos.min()), int(pos.max()) + 1
        if pos.size != x1 - x0:
            print(f"  ! {ds} group {c} is not contiguous in the column order ({pos.size} of {x1 - x0})")
        # inset, so that boxes on neighbouring groups or rows never share an edge and read as one
        px, py = min(1.5, (x1 - x0) / 6), 0.09
        ax.add_patch(Rectangle((x0 + px, rows.index(feat) + py), x1 - x0 - 2 * px, 1 - 2 * py,
                               fill=False, ec="black", lw=1.4, zorder=5))


def block_lines(sig, cols, gmean, gmed, n=3):
    idx = np.argsort(np.abs(sig))[::-1][:n]
    lines = [f"{'▲' if sig[j] > 0 else '▼'} {DISPLAY.get(cols[j], cols[j])}  {sig[j]:+.1f}" for j in idx]
    lines.append(f"exam grade  mean {gmean:.0f} · med {gmed:.0f}")
    return lines


def draw_blocks(fig, order, color, size, sig, cols, gmean, gmed, ds, y0=-0.06):
    """`order` is the clusters left to right as the clustermap columns run, NOT E1..Ek.
    Same fix as script 24: Kim (v2, scan p12) asked "why crossed?" because the names go by
    rising mean grade while the tree lays the blocks out by its own merges."""
    k = len(order)
    for i, c in enumerate(order):
        x = (i + 0.5) / k
        fig.text(x, y0, f"{ds[0].upper()}{c}  (n={size[c]})", ha="center", va="top",
                 fontsize=9, fontweight="bold", color=color[c])
        for j, ln in enumerate(block_lines(sig[c], cols, gmean[c], gmed[c])):
            fig.text(x, y0 - 0.05 * (j + 1), ln, ha="center", va="top", fontsize=7.5,
                     family="monospace", color="#333333")


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
    lab = fcluster(linkage(Xs, "ward"), KMAP[ds], "maxclust")
    grade = np.array([grades[h] for h in hashes])
    order = {c: i + 1 for i, c in enumerate(sorted(set(lab), key=lambda c: grade[lab == c].mean()))}
    lab = np.array([order[c] for c in lab])
    gmean = {c: grade[lab == c].mean() for c in sorted(set(lab))}
    gmed = {c: float(np.median(grade[lab == c])) for c in sorted(set(lab))}
    size = {c: int((lab == c).sum()) for c in sorted(set(lab))}
    return Xs, cols, hashes, lab, gmean, gmed, size


def annotated_clustermap(ds):
    k = KMAP[ds]
    Xs, cols, hashes, lab, gmean, gmed, size = load(ds)
    df = pd.DataFrame(Xs.T, index=[DISPLAY[c] for c in cols], columns=hashes)  # features x students
    col_colors = pd.Series({h: CLUSTER_COLORS[lab[i]] for i, h in enumerate(hashes)}, name=f"profile group (k={k})")

    g = sns.clustermap(df, method="ward", metric="euclidean", cmap="RdBu_r", center=0,
                       vmin=-2.5, vmax=2.5, xticklabels=False, yticklabels=True,
                       col_colors=col_colors, figsize=(12, 5 + 0.28 * len(cols)),
                       cbar_pos=(0.02, 0.83, 0.03, 0.15), dendrogram_ratio=(0.10, 0.16),
                       colors_ratio=0.03)
    g.ax_heatmap.set_xlabel(f"{Xs.shape[0]} students (Ward-ordered); colour band above = profile group; "
                            "black boxes = the bands described in the text")
    g.ax_heatmap.set_ylabel("")
    overlay(g, ds, lab, list(df.index))
    sig = {c: Xs[lab == c].mean(0) for c in sorted(set(lab))}   # z-signature per cluster
    # one separate info block per cluster (values = SD from the whole-cohort average)
    lab_ord = lab[g.dendrogram_col.reordered_ind]               # clusters left to right on the heatmap
    tree_order = list(dict.fromkeys(int(c) for c in lab_ord))
    draw_blocks(g.fig, tree_order, CLUSTER_COLORS, size, sig, cols, gmean, gmed, ds)
    g.fig.suptitle(f"{ds.upper()}: hierarchical clustermap (Ward, standardised cleaned features); "
                   f"each row is a feature, so the colours within a k={k} block are that group's signature",
                   y=1.02, fontsize=11)
    v = versioned(f"fig_{ds}_clustermap_annotated")
    for e in ("pdf", "png"):
        g.savefig(FIGS / f"fig_{ds}_clustermap_annotated_v{v}.{e}", bbox_inches="tight")
    plt.close(g.fig)
    print(f"  fig_{ds}_clustermap_annotated_v{v}  | " +
          "  ".join(f"{ds[0].upper()}{c}: n={size[c]}, grade {gmean[c]:.0f}/med {gmed[c]:.0f}" for c in sorted(size)))


if __name__ == "__main__":
    for ds in ("exam", "year"):
        annotated_clustermap(ds)
    print("done.")
