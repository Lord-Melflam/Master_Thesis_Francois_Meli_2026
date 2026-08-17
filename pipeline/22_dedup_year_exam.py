"""22 - Principled feature de-duplication for BOTH datasets (year & exam), and the
direct answer to the supervisors' 31-07 question: "are the redundant features the same
during the year as during the exam?".

Motivation (meeting 31-07): "AVOID features that are too dependent (correlation matrix),
otherwise we don't benefit fully from the clustering." Instead of hand-dropping features
(as the earlier year-only script 21 did), we apply ONE transparent TWO-STAGE rule to each
dataset (identically), so the removed features are an OBSERVATION, not a choice:

  Stage A - degenerate/near-constant filter:
    drop a feature whose middle 50% is a single value (raw IQR == 0). Such a feature is
    flat for the bulk of students and only encodes a handful of outliers, so it cannot
    partition the cohort and instead spawns singleton clusters (e.g. `comment share`:
    novices don't comment -> one lone commenter dominates at +23 sigma). KEPT as a
    descriptive FINDING in the dictionary; only excluded from the clustering INPUT.

  Stage B - greedy correlation de-dup at |Spearman rho| >= TAU (=0.80):
    while some pair of remaining features has |rho| >= TAU:
        drop the feature with the highest mean |rho| to the other remaining features
        (it carries the least unique information); tie-break by higher single max |rho|.
    -> the KEPT set has no pair above the threshold.

Comparing what each dataset removes answers the supervisors' "same features?" directly.

For each dataset we then compare FULL vs DEDUP clustering: silhouette + subsample
stability (ARI) across k, ARI(full-k3, dedup-k3), and the k=3 dedup signatures + mean
exam grade. De-dup changes the clustering INPUT only; all features are kept for
description (comment share etc. stay in the dictionary as findings).

Outputs -> clustering/dedup_compare_summary_v*.txt
figures -> plots/fig_dedup_stability_compare_v* , fig_{exam,year}_dedup_signatures_v*
"""
import csv, glob, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from common import KMAP   # k per dataset (single source of truth); 22 WRITES the kept set below

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
EXAM_FEATS = [c for c in DISPLAY if c not in ("active_weeks", "active_days")]  # 12 (no year-only rhythm)
YEAR_FEATS = list(DISPLAY)                                                     # 14
LOG = {"median_attempts", "median_mean_delta_sec", "median_edit_size", "median_nloc"}
TAU = 0.80
KS = list(range(2, 8))


def latest(pat): return sorted(glob.glob(str(pat)))[-1]


def versioned(stem, ext, base):
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"{stem}_v{n}.{ext}").exists(): n += 1
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
    return StandardScaler().fit_transform(X), np.array([grades[h] for h in hashes])


def degenerate(Xs, cols, min_frac=0.01):
    """Stage A: a feature is degenerate for clustering iff its middle 50% is a single
    value (raw IQR == 0) AND fewer than `min_frac` of students deviate from that value.
    That second clause is what separates a LONE-OUTLIER artifact (e.g. `comment share`:
    ~1 commenter in 569) from a genuine rare-but-real SUBGROUP (e.g. `small fix, big
    gain`: ~8% of students are the tiny-edit high-performers) — only the former is
    dropped. Returns the list of degenerate column names."""
    q75, q25 = np.percentile(Xs, [75, 25], axis=0)
    med = np.median(Xs, axis=0)
    out = []
    for j in range(len(cols)):
        if (q75[j] - q25[j]) < 1e-9:                       # flat middle 50%
            off = float(np.mean(np.abs(Xs[:, j] - med[j]) > 1e-9))
            if off < min_frac:                             # only a lone-outlier tail
                out.append(cols[j])
    return out


def greedy_dedup(Xs, cols, tau=TAU):
    """Return (kept_cols, dropped_in_order). Drop, one at a time, the most redundant
    remaining feature until no |Spearman| pair exceeds tau."""
    C = np.abs(spearmanr(Xs).correlation)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)
    keep = list(range(len(cols)))
    dropped = []
    while True:
        sub = C[np.ix_(keep, keep)]
        if sub.max() < tau:
            break
        # among kept, the one with the highest mean |rho| to the rest (least unique);
        # tie-break by higher single max |rho|.
        mean_r = sub.mean(1)
        mx_r = sub.max(1)
        cand = int(np.lexsort((mx_r, mean_r))[-1])  # last = highest mean, then highest max
        gi = keep[cand]
        dropped.append((cols[gi], float(mx_r[cand]), float(mean_r[cand])))
        keep.remove(gi)
    return [cols[i] for i in keep], dropped


def stability(Xs, k, B=40, frac=0.8):
    rng = np.random.default_rng(0)
    full = fcluster(linkage(Xs, "ward"), k, "maxclust")
    n = len(Xs); m = int(frac * n); a = []
    for _ in range(B):
        idx = np.sort(rng.choice(n, m, replace=False))
        a.append(adjusted_rand_score(full[idx], fcluster(linkage(Xs[idx], "ward"), k, "maxclust")))
    return float(np.mean(a))


def curve(Xs):
    sil, stab = [], []
    for k in KS:
        lab = fcluster(linkage(Xs, "ward"), k, "maxclust")
        sil.append(silhouette_score(Xs, lab)); stab.append(stability(Xs, k))
    return sil, stab


def analyse(name, feats_file, cols):
    k = KMAP[name.lower()]                          # data-driven k (script 28): exam 4, year 2
    Xfull, grade = load(feats_file, cols)          # naive baseline: every feature as-is
    degen = degenerate(Xfull, cols)                # stage A
    cols_nd = [c for c in cols if c not in degen]
    Xnd, _ = load(feats_file, cols_nd)
    kept, dropped = greedy_dedup(Xnd, cols_nd)     # stage B
    Xd, _ = load(feats_file, kept)                 # re-standardise on the kept subset
    silF, stabF = curve(Xfull)
    silD, stabD = curve(Xd)
    labF = fcluster(linkage(Xfull, "ward"), k, "maxclust")
    labD = fcluster(linkage(Xd, "ward"), k, "maxclust")
    ari = adjusted_rand_score(labF, labD)
    # dedup k signatures, labelled by ascending mean grade
    order = {c: i + 1 for i, c in enumerate(sorted(set(labD), key=lambda c: grade[labD == c].mean()))}
    lab = np.array([order[c] for c in labD])
    sig = {c: Xd[lab == c].mean(0) for c in sorted(set(lab))}
    gmean = {c: grade[lab == c].mean() for c in sig}
    size = {c: int((lab == c).sum()) for c in sig}
    return dict(name=name, cols=cols, degen=degen, kept=kept, dropped=dropped, sil=(silF, silD),
                stab=(stabF, stabD), ari=ari, sig=sig, gmean=gmean, size=size)


def fmt_sig(sig_vec, cols, n=4):
    idx = np.argsort(np.abs(sig_vec))[::-1][:n]
    return "; ".join(f"{'↑' if sig_vec[j] > 0 else '↓'}{DISPLAY[cols[j]]}({sig_vec[j]:+.1f})" for j in idx)


def main():
    ex = analyse("EXAM", "exam_features_all", EXAM_FEATS)
    yr = analyse("YEAR", "missions_features_all", YEAR_FEATS)

    # WRITE the de-duplicated feature set to the single canonical file that every
    # downstream script reads via common.load_kept (no hardcoded lists anywhere).
    CLU.mkdir(parents=True, exist_ok=True)
    (CLU / "kept_features.json").write_text(json.dumps({"exam": ex["kept"], "year": yr["kept"]}, indent=2))
    print(f"wrote kept_features.json  (exam {len(ex['kept'])} feats, year {len(yr['kept'])} feats)")

    L = [f"Two-stage feature cleaning for the clustering INPUT (569 linked students).",
         f"Stage A: drop degenerate (raw IQR==0). Stage B: greedy de-dup at |Spearman| >= {TAU}.",
         "Removed features stay in the dictionary as descriptive findings; only excluded from input.", ""]
    for r in (ex, yr):
        L.append(f"=== {r['name']} — {len(r['cols'])} features -> kept {len(r['kept'])} for clustering ===")
        L.append(f"  A. degenerate (near-constant, IQR==0): {sorted(DISPLAY[c] for c in r['degen']) or '(none)'}")
        L.append("  B. redundant, dropped (most redundant first):")
        for c, mx, mn in r["dropped"]:
            L.append(f"       - {DISPLAY[c]:<22} (max|rho|={mx:.2f}, mean|rho|={mn:.2f})")
        L.append("  kept: " + ", ".join(DISPLAY[c] for c in r["kept"]))
        L.append("")

    # answer the supervisors' question directly (both stages combined)
    ex_rm = set(ex["degen"]) | {c for c, _, _ in ex["dropped"]}
    yr_rm = set(yr["degen"]) | {c for c, _, _ in yr["dropped"]}
    L.append("--- Are the removed (degenerate+redundant) features the same in year and exam? ---")
    L.append(f"  removed in BOTH : {sorted(DISPLAY[c] for c in ex_rm & yr_rm) or '(none)'}")
    L.append(f"  removed in EXAM only: {sorted(DISPLAY[c] for c in ex_rm - yr_rm) or '(none)'}")
    L.append(f"  removed in YEAR only: {sorted(DISPLAY[c] for c in yr_rm - ex_rm) or '(none)'}")
    L.append("")

    for r in (ex, yr):
        L.append(f"=== {r['name']} stability: FULL vs DEDUP ===")
        L.append(f"  {'k':>2}  {'sil_full':>8} {'stab_full':>9}   {'sil_dedup':>9} {'stab_dedup':>10}")
        for i, k in enumerate(KS):
            L.append(f"  {k:>2}  {r['sil'][0][i]:8.3f} {r['stab'][0][i]:9.2f}   {r['sil'][1][i]:9.3f} {r['stab'][1][i]:10.2f}")
        kk = KMAP[r['name'].lower()]
        L.append(f"  ARI(full k={kk}, dedup k={kk}) = {r['ari']:.2f}  (1=same grouping; low=de-dup changed structure)")
        L.append(f"  DEDUP k={kk} clusters (labelled by ascending mean exam grade):")
        for c in sorted(r["sig"]):
            L.append(f"    {r['name'][0]}{c}: n={r['size'][c]}, mean exam grade {r['gmean'][c]:.0f} | {fmt_sig(r['sig'][c], r['kept'])}")
        L.append("")

    ps = versioned("dedup_compare_summary", "txt", CLU); Path(ps).write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"wrote {ps.name}")

    # figure: stability full vs dedup, both datasets
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, r in zip(axes, (ex, yr)):
        ax.plot(KS, r["stab"][0], marker="o", lw=1.4, label="full")
        ax.plot(KS, r["stab"][1], marker="s", lw=1.4, label=f"de-dup ({len(r['kept'])} feats)")
        ax.set_xlabel("k"); ax.set_ylim(0, 1)
        ax.set_title(f"{r['name']} ({len(r['cols'])} feats)", fontsize=11)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("subsample stability (ARI)")
    fig.suptitle(f"Clustering stability: full vs de-duplicated features (|rho|>={TAU})", fontsize=12)
    p = versioned("fig_dedup_stability_compare", "pdf", FIGS)
    for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig); print(f"fig {p.stem}")

    # figure: dedup k=3 signatures per dataset
    for r in (ex, yr):
        cols = r["kept"]; M = np.array([r["sig"][c] for c in sorted(r["sig"])])
        fig, ax = plt.subplots(figsize=(2 + 0.55 * len(cols), 2.6))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([DISPLAY[c] for c in cols], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(r["sig"]))); ax.set_yticklabels([f"{r['name'][0]}{c} (grade {r['gmean'][c]:.0f})" for c in sorted(r["sig"])])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:+.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(M[i, j]) > 1.2 else "black")
        fig.colorbar(im, ax=ax, shrink=0.7, label="mean z")
        ax.set_title(f"{r['name']} (de-dup) k=3 — cluster signatures", fontsize=11)
        p = versioned(f"fig_{r['name'].lower()}_dedup_signatures", "pdf", FIGS)
        for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
        plt.close(fig); print(f"fig {p.stem}")


if __name__ == "__main__":
    main()
