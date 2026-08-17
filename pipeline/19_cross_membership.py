"""19 - At k=3: (a) do students land in the "same" group in the YEAR and in the EXAM?
(cross-membership of the two behaviour clusterings on the same 569 students), and
(b) the most prominent variables characterising each of the 3 clusters, per dataset.

- exam clusters (12 features) and year clusters (14 features) are two labelings of the
  SAME students. Cross-tab year-cluster x exam-cluster (+ row %), plus their ARI.
- per cluster: mean exam grade + top features by |z-signature| (deviation from average),
  in plain names — "what explains this cluster".

Outputs -> data/v2/res_python/clustering/ ; figures -> data/v2/res_python/plots/
"""
import csv, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

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
K = 3


def latest(pat): return sorted(glob.glob(str(pat)))[-1]


def versioned(stem, ext, base):
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"{stem}_v{n}.{ext}").exists(): n += 1
    return base / f"{stem}_v{n}.{ext}"


def cluster_ds(feats_file, cols):
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
    Xs = StandardScaler().fit_transform(X)
    lab = fcluster(linkage(Xs, "ward"), K, "maxclust")
    grade = np.array([grades[h] for h in hashes])
    # relabel clusters by mean exam grade ascending -> stable, readable ids 1..K
    order = {c: i + 1 for i, c in enumerate(sorted(set(lab), key=lambda c: grade[lab == c].mean()))}
    lab = np.array([order[c] for c in lab])
    sig = {c: Xs[lab == c].mean(0) for c in sorted(set(lab))}
    return dict(zip(hashes, lab)), {c: grade[lab == c].mean() for c in sig}, sig, cols


def top_feats(sig_vec, cols, n=4):
    idx = np.argsort(np.abs(sig_vec))[::-1][:n]
    return "; ".join(f"{'↑' if sig_vec[j] > 0 else '↓'}{DISPLAY[cols[j]]}({sig_vec[j]:+.1f})" for j in idx)


def main():
    ex_lab, ex_grade, ex_sig, ex_cols = cluster_ds("exam_features_all", EXAM_FEATS)
    yr_lab, yr_grade, yr_sig, yr_cols = cluster_ds("missions_features_all", YEAR_FEATS)
    common = sorted(set(ex_lab) & set(yr_lab))
    L = [f"k=3 cross-membership & signatures — {len(common)} students in both.", ""]

    L.append("EXAM clusters (labelled 1..3 by ascending mean exam grade):")
    for c in sorted(ex_sig):
        n = sum(1 for h in common if ex_lab[h] == c)
        L.append(f"  E{c}: n={n}, mean exam grade {ex_grade[c]:.0f} | {top_feats(ex_sig[c], ex_cols)}")
    L.append("\nYEAR clusters (labelled 1..3 by ascending mean exam grade):")
    for c in sorted(yr_sig):
        n = sum(1 for h in common if yr_lab[h] == c)
        L.append(f"  Y{c}: n={n}, mean exam grade {yr_grade[c]:.0f} | {top_feats(yr_sig[c], yr_cols)}")

    # cross-tab year (rows) x exam (cols)
    ex_ids = sorted(ex_sig); yr_ids = sorted(yr_sig)
    ct = np.zeros((len(yr_ids), len(ex_ids)), int)
    for h in common:
        ct[yr_ids.index(yr_lab[h]), ex_ids.index(ex_lab[h])] += 1
    ari = adjusted_rand_score([yr_lab[h] for h in common], [ex_lab[h] for h in common])
    L.append(f"\nCross-membership  YEAR-cluster (rows) x EXAM-cluster (cols)  — ARI={ari:.2f} (0=independent,1=identical)")
    L.append("counts:")
    L.append("        " + "  ".join(f"E{c:>3}" for c in ex_ids) + "   row total")
    for i, yc in enumerate(yr_ids):
        L.append(f"  Y{yc}   " + "  ".join(f"{ct[i,j]:4d}" for j in range(len(ex_ids))) + f"   {ct[i].sum()}")
    L.append("row % (of students in each YEAR cluster, where they land in the EXAM):")
    rown = ct / ct.sum(1, keepdims=True)
    for i, yc in enumerate(yr_ids):
        L.append(f"  Y{yc}   " + "  ".join(f"{100*rown[i,j]:3.0f}%" for j in range(len(ex_ids))))

    ps = versioned("k3_cross_membership_summary", "txt", CLU); Path(ps).write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"\nwrote {ps.name}")

    # figure: cross-membership heatmap (row-normalised)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ex_ids))); ax.set_xticklabels([f"exam E{c}\n(grade {ex_grade[c]:.0f})" for c in ex_ids])
    ax.set_yticks(range(len(yr_ids))); ax.set_yticklabels([f"year Y{c}\n(grade {yr_grade[c]:.0f})" for c in yr_ids])
    for i in range(len(yr_ids)):
        for j in range(len(ex_ids)):
            ax.text(j, i, f"{100*rown[i,j]:.0f}%\n({ct[i,j]})", ha="center", va="center",
                    color="white" if rown[i, j] > 0.5 else "black", fontsize=9)
    ax.set_title(f"Where each YEAR behaviour group lands in the EXAM (k=3, ARI={ari:.2f})", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.7, label="row share")
    p = versioned("fig_k3_cross_membership", "pdf", FIGS)
    for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig); print(f"fig {p.stem}")

    # figure: per-cluster signature heatmaps (exam & year)
    for ds, sig, cols, gr in [("exam", ex_sig, ex_cols, ex_grade), ("year", yr_sig, yr_cols, yr_grade)]:
        M = np.array([sig[c] for c in sorted(sig)])
        fig, ax = plt.subplots(figsize=(2 + 0.55 * len(cols), 2.6))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([DISPLAY[c] for c in cols], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(sig))); ax.set_yticklabels([f"{ds[0].upper()}{c} (grade {gr[c]:.0f})" for c in sorted(sig)])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:+.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(M[i, j]) > 1.2 else "black")
        ax.set_title(f"{ds} k=3 — cluster signatures (z vs average; ↑/↓ = prominent)", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.7, label="mean z")
        p = versioned(f"fig_{ds}_k3_signatures", "pdf", FIGS)
        for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
        plt.close(fig); print(f"fig {p.stem}")


if __name__ == "__main__":
    main()
