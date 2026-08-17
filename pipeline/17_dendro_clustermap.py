"""17 - Linkage-comparison dendrograms + WARD hierarchical clustermaps, for BOTH datasets.

For exam AND year (coursework), on the 569 LINKED cohort, clustering on the same
behaviour+code-quality features as 13/14 (score held aside):
  (A) fig_{ds}_linkage_compare       — ward/complete/average/single side by side.
  (B) fig_{ds}_ward_clustermap       — seaborn clustermap, WARD linkage only (both axes),
                                       z-scored features (rows) x students (cols).
The exact feature list per dataset is documented in feedback_tfe/FEATURE_DICTIONARY.md.

Outputs (versioned PDF+PNG) -> data/v2/res_python/plots/
"""
import csv, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
FEAT = REPO / "data/v2/res_python/features"
LINK = REPO / "data/v2/res_python/linkage"
FIGS = REPO / "data/v2/res_python/plots"

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
    "active_weeks": "active weeks in the semester",
    "active_days": "active days in the semester",
}
LOG = {"median_attempts", "median_mean_delta_sec", "median_edit_size", "median_nloc"}
from common import cluster   # canonical clustering input (de-dup features, one recipe)


def latest(pat): return sorted(glob.glob(str(pat)))[-1]


def versioned(stem):
    FIGS.mkdir(parents=True, exist_ok=True)
    n = 1
    while (FIGS / f"{stem}_v{n}.pdf").exists() or (FIGS / f"{stem}_v{n}.png").exists():
        n += 1
    return n


def load_scaled(feats_file, cols):
    linked = {r["hash"] for r in csv.DictReader(open(latest(LINK / "linked_students_v*.csv")))}
    feats = {r["hash"]: r for r in csv.DictReader(open(latest(FEAT / f"{feats_file}_v*.csv")))}
    hashes = [h for h in feats if h in linked]

    def val(h, c):
        try: return float(feats[h][c])
        except: return np.nan
    X = np.nan_to_num(np.array([[val(h, c) for c in cols] for h in hashes], float), nan=0.0)
    for j, c in enumerate(cols):
        if c in LOG: X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    return StandardScaler().fit_transform(X), hashes


def linkage_compare(Xs, ds, n):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, lk in zip(axes.ravel(), ["ward", "complete", "average", "single"]):
        Z = linkage(Xs, method=lk)
        dendrogram(Z, no_labels=True, color_threshold=0, ax=ax)
        for coll in ax.collections:
            coll.set_linewidth(0.4); coll.set_color("#333333")
        ax.set_title(f"{lk} linkage", fontsize=11); ax.set_yticks([])
    # Kim (v2, scan p9/p10): "make broader to fit the page width". The panels were narrower
    # than the text block because this suptitle ran wider than them and set the saved bounding
    # box. The long second line was also a caption baked into the image, which he asked to move
    # into the text. Keeping the title to one short line lets the panels fill \textwidth.
    fig.suptitle(f"{ds} behavior: hierarchical tree by linkage (n={Xs.shape[0]} students)",
                 fontsize=12, y=1.0)
    fig.subplots_adjust(left=0.02, right=0.99, wspace=0.06)
    v = versioned(f"fig_{ds}_linkage_compare")
    for e in ("pdf", "png"): fig.savefig(FIGS / f"fig_{ds}_linkage_compare_v{v}.{e}", bbox_inches="tight")
    plt.close(fig); print(f"  fig_{ds}_linkage_compare_v{v}")


def ward_clustermap(Xs, cols, ds):
    import pandas as pd
    df = pd.DataFrame(Xs.T, index=[DISPLAY[c] for c in cols])   # features x students
    g = sns.clustermap(df, method="ward", metric="euclidean", cmap="RdBu_r", center=0,
                       vmin=-2.5, vmax=2.5, xticklabels=False, yticklabels=True,
                       figsize=(12, 5 + 0.25 * len(cols)), cbar_pos=(0.02, 0.83, 0.03, 0.15),
                       dendrogram_ratio=(0.12, 0.18))
    g.ax_heatmap.set_xlabel(f"{Xs.shape[0]} students (Ward-ordered)")
    g.ax_heatmap.set_ylabel("")
    g.fig.suptitle(f"{ds} — hierarchical clustermap (Ward, z-scored features)", y=1.02, fontsize=12)
    v = versioned(f"fig_{ds}_ward_clustermap")
    for e in ("pdf", "png"): g.savefig(FIGS / f"fig_{ds}_ward_clustermap_v{v}.{e}", bbox_inches="tight")
    plt.close(g.fig); print(f"  fig_{ds}_ward_clustermap_v{v}")


if __name__ == "__main__":
    for ds in ("exam", "year"):
        hashes, Xs, lab, grade, cols = cluster(ds)   # canonical de-dup input
        print(f"{ds}: {len(hashes)} students, {len(cols)} features")
        linkage_compare(Xs, ds, len(hashes))
        ward_clustermap(Xs, cols, ds)
    print("done.")
