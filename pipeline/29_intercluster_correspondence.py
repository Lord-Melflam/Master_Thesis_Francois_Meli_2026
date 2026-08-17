"""29 - Inter-cluster correspondence: year profile <-> exam profile (Siegfried, W6).

W6: "before the common-student proportion (ARI~0.01), compute the correlation/correspondence
between the year and exam clusters, then read the grouping - more precise. Give a correlation
image for inter-cluster." So instead of a single scalar, we give the full YxE correspondence
matrix (row-normalised) with a proper association test:
  - chi-square test of independence on the YxE contingency of the 569 common students
  - Cramer's V  = sqrt(chi2 / (n * (min(r,c)-1)))   (0 = no association, 1 = perfect)
  - ARI kept only as ONE summary of the same table.
Each dataset is clustered at its OWN data-driven k (script 28): exam k=4, year k=2 - NOT a
forced common k. Careful wording: a low V + non-significant chi-square means "no DETECTABLE
association", not a proof of independence.

Outputs -> clustering/intercluster_correspondence_summary_v*.txt
figures  -> plots/fig_intercluster_correspondence_v*
"""
import csv, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

REPO = Path(__file__).resolve().parents[3]
FEAT = REPO / "data/v2/res_python/features"
LINK = REPO / "data/v2/res_python/linkage"
CLU = REPO / "data/v2/res_python/clustering"
FIGS = REPO / "data/v2/res_python/plots"

LOG = {"median_attempts", "median_mean_delta_sec", "median_edit_size", "median_nloc"}
FEATS_FILE = {"exam": "exam_features_all", "year": "missions_features_all"}
from common import load_kept, KMAP   # kept set (written by 22) + k; single source of truth


def latest(pat): return sorted(glob.glob(str(pat)))[-1]


def versioned(stem, ext, base):
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"{stem}_v{n}.{ext}").exists(): n += 1
    return base / f"{stem}_v{n}.{ext}"


def cluster_labels(ds):
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
    return dict(zip(hashes, lab)), gmean


def cramers_v(ct):
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.sum()
    v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if min(ct.shape) > 1 and n > 0 else 0.0
    return chi2, p, dof, v


def main():
    ex_lab, ex_g = cluster_labels("exam")
    yr_lab, yr_g = cluster_labels("year")
    common = sorted(set(ex_lab) & set(yr_lab))
    yr_ids = sorted(set(yr_lab.values())); ex_ids = sorted(set(ex_lab.values()))
    ct = np.zeros((len(yr_ids), len(ex_ids)), int)
    for h in common:
        ct[yr_ids.index(yr_lab[h]), ex_ids.index(ex_lab[h])] += 1

    chi2, p, dof, V = cramers_v(ct)
    ari = adjusted_rand_score([yr_lab[h] for h in common], [ex_lab[h] for h in common])
    rown = ct / ct.sum(1, keepdims=True)

    L = [f"Inter-cluster correspondence — YEAR profile (k={KMAP['year']}) x EXAM profile (k={KMAP['exam']}).",
         f"{len(common)} students in both. Clustered at each dataset's own data-driven k (script 28).", "",
         f"chi-square = {chi2:.1f}, dof = {dof}, p = {p:.3f}", f"Cramer's V = {V:.3f}  (0=no association, 1=perfect)",
         f"ARI = {ari:.3f}  (one scalar summary of the same table)",
         "", "counts  (rows = year Yx, cols = exam Ex):",
         "        " + "  ".join(f"E{c}(g{ex_g[c]:.0f})" for c in ex_ids) + "   row total"]
    for i, yc in enumerate(yr_ids):
        L.append(f"  Y{yc}(g{yr_g[yc]:.0f})  " + "  ".join(f"{ct[i,j]:6d}" for j in range(len(ex_ids))) + f"   {ct[i].sum()}")
    L.append("row % (of each YEAR cluster, where they land in the EXAM):")
    for i, yc in enumerate(yr_ids):
        L.append(f"  Y{yc}       " + "  ".join(f"{100*rown[i,j]:5.0f}%" for j in range(len(ex_ids))))
    if p >= 0.05:
        verdict = "no statistically significant association (p>=0.05) -> groupings independent at this k"
    else:
        strength = "weak" if V < 0.2 else ("moderate" if V < 0.4 else "strong")
        verdict = (f"a {strength} but STATISTICALLY SIGNIFICANT association (V={V:.2f}, p={p:.3f}) -> "
                   f"NOT independent, though the year explains little of the exam grouping")
    small = int((chi2_contingency(ct)[3] < 5).sum())
    L.append(""); L.append(f"reading: {verdict}.")
    L.append(f"  NB1: a near-zero ARI ({ari:.2f}) alone would have wrongly read as 'independent' — the "
             f"chi-square/V is the precise test (Siegfried's point).")
    L.append(f"  NB2: {small} of {ct.size} cells have expected count <5 (small E2/E4 columns) -> the "
             f"chi-square p is fragile here; treat the association as weak-and-tentative, not strong.")

    ps = versioned("intercluster_correspondence_summary", "txt", CLU)
    Path(ps).write_text("\n".join(L) + "\n"); print("\n".join(L)); print(f"wrote {ps.name}")

    # Draw the heatmap as solid vector rectangles (NOT imshow): matplotlib's PDF backend
    # renders imshow images with a diagonal interpolation moiré in some viewers; flat
    # Rectangle patches are pure vector fills and render cleanly at any zoom.
    from matplotlib.patches import Rectangle
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    nyr, nex = len(yr_ids), len(ex_ids)
    cmap = plt.cm.Blues; norm = Normalize(0, 1)
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * nex, 1.6 + 0.9 * nyr))
    for i in range(nyr):
        for j in range(nex):
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=cmap(norm(rown[i, j])),
                                   edgecolor="white", linewidth=1.0))
            ax.text(j + 0.5, i + 0.5, f"{100*rown[i,j]:.0f}%\n({ct[i,j]})", ha="center", va="center",
                    color="white" if rown[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlim(0, nex); ax.set_ylim(0, nyr); ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_xticks([j + 0.5 for j in range(nex)]); ax.set_xticklabels([f"exam E{c}\n(grade {ex_g[c]:.0f})" for c in ex_ids])
    ax.set_yticks([i + 0.5 for i in range(nyr)]); ax.set_yticklabels([f"year Y{c}\n(grade {yr_g[c]:.0f})" for c in yr_ids])
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    pstr = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    ax.set_title(f"Year to exam profile correspondence\nCramér's V={V:.3f}, χ²={chi2:.0f} ({pstr}), ARI={ari:.2f}",
                 fontsize=11)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.8, label="row share")
    p_ = versioned("fig_intercluster_correspondence", "pdf", FIGS)
    for e in ("pdf", "png"): fig.savefig(p_.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig); print(f"fig {p_.stem}")


if __name__ == "__main__":
    main()
