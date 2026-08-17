"""18 - Do NOT fix k. Study the Ward hierarchy across k: membership, stability, and which
features explain each separation. For exam AND year (569 linked cohort).

For each k = 2..7 (Ward):
  - assignments saved per student (hash x k) -> membership tracking across k.
  - STABILITY by subsampling: cluster an 80% subsample B times, compare to the full
    clustering on the shared students (Adjusted Rand Index). High = the split is real,
    not an artefact of who's in the sample. (This is how we judge a good cut WITHOUT
    pre-committing to k.)
  - silhouette (separation quality).
  - FEATURE ATTRIBUTION: eta^2 per feature = between-cluster variance / total variance
    (0..1) — how much each feature explains that k-partition.
  - per-cluster size + mean exam grade (outcome, aside) + top explaining features.

Outputs -> data/v2/res_python/clustering/  ; figures -> data/v2/res_python/plots/
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
EXAM_FEATS = [c for c in DISPLAY if c not in ("active_weeks", "active_days")]
YEAR_FEATS = list(DISPLAY)
LOG = {"median_attempts", "median_mean_delta_sec", "median_edit_size", "median_nloc"}
KS = list(range(2, 8))


def latest(pat): return sorted(glob.glob(str(pat)))[-1]


def versioned(stem, ext, base):
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"{stem}_v{n}.{ext}").exists():
        n += 1
    return base / f"{stem}_v{n}.{ext}"


def load(feats_file, cols):
    linked = {r["hash"] for r in csv.DictReader(open(latest(LINK / "linked_students_v*.csv")))}
    feats = {r["hash"]: r for r in csv.DictReader(open(latest(FEAT / f"{feats_file}_v*.csv")))}
    grades = {r["hash"]: float(r["exam_grade"]) for r in csv.DictReader(open(latest(CLU / "exam_score_categories_v*.csv")))}
    hashes = [h for h in feats if h in linked and h in grades]

    def val(h, c):
        try: return float(feats[h][c])
        except: return np.nan
    X = np.nan_to_num(np.array([[val(h, c) for c in cols] for h in hashes], float), nan=0.0)
    for j, c in enumerate(cols):
        if c in LOG: X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    return StandardScaler().fit_transform(X), hashes, np.array([grades[h] for h in hashes])


def sizes_of(lab):
    """cluster sizes, largest first (fcluster labels are 1..k; drop the empty 0 bin)."""
    return sorted((int(s) for s in np.bincount(lab) if s > 0), reverse=True)


def eta2(x, lab):
    ss_tot = float(((x - x.mean()) ** 2).sum())
    if ss_tot == 0:
        return 0.0
    ss_bet = sum(len(x[lab == c]) * (x[lab == c].mean() - x.mean()) ** 2 for c in set(lab))
    return float(ss_bet / ss_tot)


def subsample_stability(Xs, k, B=40, frac=0.8, seed=0):
    rng = np.random.default_rng(seed)
    Zf = linkage(Xs, "ward"); full = fcluster(Zf, k, "maxclust")
    n = len(Xs); m = int(frac * n); aris = []
    for _ in range(B):
        idx = np.sort(rng.choice(n, m, replace=False))
        lab = fcluster(linkage(Xs[idx], "ward"), k, "maxclust")
        aris.append(adjusted_rand_score(full[idx], lab))
    return float(np.mean(aris)), float(np.std(aris))


def run(ds, feats_file, cols):
    Xs, hashes, grade = load(feats_file, cols)
    Z = linkage(Xs, "ward")
    print(f"\n===== {ds}: {len(hashes)} students, {len(cols)} features =====")
    print(f"{'k':>2} {'sizes':28} {'silhouette':>10} {'stability(ARI)':>15}  top explaining features (eta^2)")
    rows_k = {}
    eta_mat = np.zeros((len(cols), len(KS)))
    summary = [f"{ds.upper()} — varying-k Ward analysis, {len(hashes)} students",
               "k | sizes | silhouette | subsample-stability ARI (mean±sd) | top explaining features (eta^2) | per-cluster grade"]
    for ki, k in enumerate(KS):
        lab = fcluster(Z, k, "maxclust")
        rows_k[k] = lab
        sizes = sizes_of(lab)
        sil = silhouette_score(Xs, lab)
        stab_m, stab_s = subsample_stability(Xs, k)
        etas = np.array([eta2(Xs[:, j], lab) for j in range(len(cols))])
        eta_mat[:, ki] = etas
        top = np.argsort(etas)[::-1][:4]
        topstr = ", ".join(f"{DISPLAY[cols[j]]}({etas[j]:.2f})" for j in top)
        print(f"{k:>2} {str(sizes):28} {sil:>10.3f} {stab_m:>9.2f}±{stab_s:.2f}  {topstr}")
        grades_by_c = "; ".join(f"C{c}:n{int((lab==c).sum())},grade{grade[lab==c].mean():.0f}" for c in sorted(set(lab)))
        summary.append(f"k={k} | {sizes} | sil={sil:.3f} | stab={stab_m:.2f}±{stab_s:.2f} | {topstr} | {grades_by_c}")

    # assignments across k (membership tracking; hashes saved)
    pa = versioned(f"{ds}_assignments_across_k", "csv", CLU)
    with open(pa, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["hash", "exam_grade"] + [f"k{k}" for k in KS])
        for i, h in enumerate(hashes):
            w.writerow([h, round(grade[i], 1)] + [int(rows_k[k][i]) for k in KS])

    # pick candidate "best separation" = highest stability among k with all clusters >=3% (not fixed to 2)
    minsize = max(3, int(0.03 * len(hashes)))
    cand = [(k, subsample_stability(Xs, k)[0]) for k in KS
            if min(sizes_of(fcluster(Z, k, "maxclust"))) >= minsize]
    best_k = max(cand, key=lambda t: t[1])[0] if cand else 2
    summary.append(f"\nCANDIDATE best separation (highest subsample-stability among non-degenerate k): k={best_k}")
    summary.append("(reported, NOT hard-fixed — see the stability curve; interpret with the feature attribution)")

    ps = versioned(f"{ds}_k_stability_summary", "txt", CLU); Path(ps).write_text("\n".join(summary) + "\n")
    print(f"  wrote {pa.name}, {ps.name}")

    # figure: silhouette + stability vs k
    sils = [silhouette_score(Xs, fcluster(Z, k, "maxclust")) for k in KS]
    stabs = [subsample_stability(Xs, k)[0] for k in KS]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(KS, sils, marker="o", lw=1.4, label="silhouette (separation)")
    ax.plot(KS, stabs, marker="s", lw=1.4, label="stability (subsample ARI)")
    ax.axvline(best_k, color="#8C8C8C", ls="--", lw=1, label=f"most stable non-degenerate k={best_k}")
    ax.set_xlabel("number of clusters k"); ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.set_title(f"{ds} — choosing k by separation AND stability (not fixed)", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    p1 = versioned(f"fig_{ds}_k_stability", "pdf", FIGS)
    for e in ("pdf", "png"): fig.savefig(p1.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig)

    # figure: feature-attribution heatmap (eta^2, features x k)
    fig, ax = plt.subplots(figsize=(1.6 + 0.7 * len(KS), 0.6 + 0.4 * len(cols)))
    im = ax.imshow(eta_mat, cmap="magma", vmin=0, vmax=max(0.3, eta_mat.max()), aspect="auto")
    ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_yticks(range(len(cols))); ax.set_yticklabels([DISPLAY[c] for c in cols])
    for i in range(len(cols)):
        for j in range(len(KS)):
            ax.text(j, i, f"{eta_mat[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if eta_mat[i, j] < 0.6 * eta_mat.max() else "black")
    fig.colorbar(im, ax=ax, shrink=0.6, label="variance explained (eta²)")
    ax.set_title(f"{ds} — which features explain the split, per k", fontsize=12)
    p2 = versioned(f"fig_{ds}_feature_attribution", "pdf", FIGS)
    for e in ("pdf", "png"): fig.savefig(p2.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig)
    print(f"  figs fig_{ds}_k_stability, fig_{ds}_feature_attribution")
    return best_k


if __name__ == "__main__":
    run("exam", "exam_features_all", EXAM_FEATS)
    run("year", "missions_features_all", YEAR_FEATS)
    print("\ndone.")
