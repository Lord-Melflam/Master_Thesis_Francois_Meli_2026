"""20 - Feature <-> feature correlation matrix, exam & year (569 linked cohort).

Spearman (rank) correlation between the clustering features — scale-free, so the log/z
transforms don't matter. Reveals redundant feature pairs and near-constant features
(e.g. comment share). Features ordered by hierarchical clustering of 1-|corr| so
correlated blocks sit together.

Outputs -> data/v2/res_python/clustering/{ds}_feature_correlation_v*.csv
Figures -> data/v2/res_python/plots/fig_{ds}_feature_correlation_v*.{pdf,png}
"""
import csv, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

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
EXAM_FEATS = [c for c in DISPLAY if c not in ("active_weeks", "active_days")]
YEAR_FEATS = list(DISPLAY)


def latest(pat): return sorted(glob.glob(str(pat)))[-1]


def versioned(stem, ext, base):
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"{stem}_v{n}.{ext}").exists(): n += 1
    return base / f"{stem}_v{n}.{ext}"


def run(ds, feats_file, cols):
    linked = {r["hash"] for r in csv.DictReader(open(latest(LINK / "linked_students_v*.csv")))}
    feats = {r["hash"]: r for r in csv.DictReader(open(latest(FEAT / f"{feats_file}_v*.csv")))}
    hashes = [h for h in feats if h in linked]

    def val(h, c):
        try: return float(feats[h][c])
        except: return np.nan
    X = np.nan_to_num(np.array([[val(h, c) for c in cols] for h in hashes], float), nan=0.0)

    C, _ = spearmanr(X)            # feature x feature Spearman
    C = np.atleast_2d(C)
    near_const = [cols[j] for j in range(len(cols)) if np.nanstd(X[:, j]) < 1e-9]
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)

    # order features so correlated blocks cluster
    D = 1 - np.abs(C); np.fill_diagonal(D, 0.0)
    order = leaves_list(linkage(squareform(D, checks=False), "average"))
    Co = C[np.ix_(order, order)]; labels = [DISPLAY[cols[j]] for j in order]

    # save matrix
    p = versioned(f"{ds}_feature_correlation", "csv", CLU)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow([""] + [cols[j] for j in order])
        for i in order:
            w.writerow([cols[i]] + [f"{C[i,j]:.3f}" for j in order])

    # strongly correlated pairs (|rho|>=0.6), redundancy candidates
    pairs = []
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            if abs(C[a, b]) >= 0.6:
                pairs.append((abs(C[a, b]), cols[a], cols[b], C[a, b]))
    pairs.sort(reverse=True)

    print(f"\n===== {ds}: {len(hashes)} students, {len(cols)} features =====")
    print("near-constant features (undefined correlation):", [DISPLAY[c] for c in near_const] or "none")
    print("strongly correlated pairs (|rho|>=0.6):")
    for _, a, b, r in pairs:
        print(f"  {r:+.2f}  {DISPLAY[a]}  <->  {DISPLAY[b]}")
    if not pairs:
        print("  none")

    # heatmap drawn as solid vector rectangles (NOT imshow): matplotlib's PDF backend
    # renders imshow images with a diagonal interpolation moiré in some viewers; flat
    # Rectangle patches are pure vector fills and stay crisp at any zoom.
    from matplotlib.patches import Rectangle
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    n = len(cols)
    cmap = plt.cm.RdBu_r; norm = Normalize(-1, 1)
    fig, ax = plt.subplots(figsize=(1.4 + 0.6 * n, 1.2 + 0.55 * n))
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=cmap(norm(Co[i, j])),
                                   edgecolor="white", linewidth=0.5))
            ax.text(j + 0.5, i + 0.5, f"{Co[i,j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(Co[i, j]) > 0.6 else "black")
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_xticks([j + 0.5 for j in range(n)]); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(n)]); ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, label="Spearman ρ")
    ax.set_title(f"{ds}: correlation between features (Spearman, {len(hashes)} students)", fontsize=11)
    pf = versioned(f"fig_{ds}_feature_correlation", "pdf", FIGS)
    for e in ("pdf", "png"): fig.savefig(pf.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.name}, {pf.stem}")


if __name__ == "__main__":
    run("exam", "exam_features_all", EXAM_FEATS)
    run("year", "missions_features_all", YEAR_FEATS)
    print("\ndone.")
