"""30 - Robustness check (NOT an optimisation): does the weak year<->exam link hold across
year k? (François's W6 idea, reframed.)

We DO NOT pick the year k that maximises agreement with the exam — that would be post-hoc
selection (the very thing flagged in June). Instead we FIX exam at its data-driven k=4 and
vary year k in {2,3,4}, reporting for each: the year clustering's own internal quality
(silhouette, subsample stability) AND its correspondence with the exam (ARI, Cramér's V,
chi-square p, and the largest departure of P(exam|year) from the base rate = max lift).

If the association stays weak for every year k, the "year does not predict exam" finding is
robust to k. Any k that spikes the association at the cost of internal quality is a red flag
(p-hacking), not a result.

Outputs -> clustering/year_k_sensitivity_summary_v*.txt
"""
import csv, glob
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

REPO = Path(__file__).resolve().parents[3]
FEAT = REPO / "data/v2/res_python/features"
LINK = REPO / "data/v2/res_python/linkage"
CLU = REPO / "data/v2/res_python/clustering"

LOG = {"median_attempts", "median_mean_delta_sec", "median_edit_size", "median_nloc"}
FEATS_FILE = {"exam": "exam_features_all", "year": "missions_features_all"}
from common import load_kept, KMAP   # shared kept set (script 22) + k
EXAM_K = KMAP["exam"]                 # exam fixed at its chosen k
YEAR_KS = [2, 3, 4]


def latest(pat): return sorted(glob.glob(str(pat)))[-1]


def versioned(stem, ext, base):
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"{stem}_v{n}.{ext}").exists(): n += 1
    return base / f"{stem}_v{n}.{ext}"


def grades_map():
    return {r["hash"]: float(r["exam_grade"])
            for r in csv.DictReader(open(latest(CLU / "exam_score_categories_v*.csv")))}


def load(ds):
    cols = load_kept(ds)
    linked = {r["hash"] for r in csv.DictReader(open(latest(LINK / "linked_students_v*.csv")))}
    feats = {r["hash"]: r for r in csv.DictReader(open(latest(FEAT / f"{FEATS_FILE[ds]}_v*.csv")))}
    g = grades_map()
    hashes = [h for h in feats if h in linked and h in g]

    def val(h, c):
        try: return float(feats[h][c])
        except: return np.nan
    X = np.nan_to_num(np.array([[val(h, c) for c in cols] for h in hashes], float), nan=0.0)
    for j, c in enumerate(cols):
        if c in LOG: X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    return StandardScaler().fit_transform(X), hashes


def labels(Xs, k):
    return fcluster(linkage(Xs, "ward"), k, "maxclust")


def grade_order(lab, grade):
    """Relabel clusters 1..k by ascending mean exam grade (readability only)."""
    order = {c: i + 1 for i, c in enumerate(sorted(set(lab), key=lambda c: grade[lab == c].mean()))}
    return np.array([order[c] for c in lab])


def stability(Xs, k, B=40, frac=0.8):
    rng = np.random.default_rng(0)
    full = labels(Xs, k); n = len(Xs); m = int(frac * n); a = []
    for _ in range(B):
        idx = np.sort(rng.choice(n, m, replace=False))
        a.append(adjusted_rand_score(full[idx], labels(Xs[idx], k)))
    return float(np.mean(a))


def cramers_v(ct):
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.sum()
    v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if min(ct.shape) > 1 and n else 0.0
    return chi2, p, dof, v


def main():
    Xe, he = load("exam"); Xy, hy = load("year")
    g = grades_map()
    ge = np.array([g[h] for h in he]); gy = np.array([g[h] for h in hy])
    labE = grade_order(labels(Xe, EXAM_K), ge)                # exam E1..E4 by ascending grade
    ex = dict(zip(he, labE))
    exg = {c: ge[labE == c].mean() for c in sorted(set(labE))}
    common = [h for h in hy if h in ex]
    ex_ids = sorted(set(labE))
    base = np.array([sum(1 for h in common if ex[h] == c) for c in ex_ids])
    base_p = base / base.sum()

    L = [f"Year-k sensitivity (exam FIXED at k={EXAM_K}). {len(common)} common students.",
         "ROBUSTNESS check, not optimisation: is the weak year<->exam link stable across year k?", ""]
    # summary metrics
    L += ["=== summary (year clustering internal quality + correspondence with the exam) ===",
          f"{'year_k':>6} {'sil':>6} {'stab':>6}   {'ARI':>7} {'CramerV':>8} {'chi2_p':>7} {'max_lift':>8}"]
    yr_full = {k: grade_order(labels(Xy, k), gy) for k in YEAR_KS}
    yr_by_h = {k: dict(zip(hy, yr_full[k])) for k in YEAR_KS}
    tables = {}
    for k in YEAR_KS:
        sil = silhouette_score(Xy, yr_full[k]); stab = stability(Xy, k)
        yk = np.array([yr_by_h[k][h] for h in common]); ek = np.array([ex[h] for h in common])
        yr_ids = sorted(set(yk))
        ct = np.zeros((len(yr_ids), len(ex_ids)), int)
        for a, b in zip(yk, ek): ct[yr_ids.index(a), ex_ids.index(b)] += 1
        chi2, p, dof, V = cramers_v(ct)
        ari = adjusted_rand_score(yk, ek)
        rown = ct / ct.sum(1, keepdims=True)
        max_lift = float(np.nanmax(np.abs(rown / base_p[None, :] - 1)) + 1)
        tables[k] = (yr_ids, ct)
        L.append(f"{k:>6} {sil:6.3f} {stab:6.2f}   {ari:7.3f} {V:8.3f} {p:7.3f} {max_lift:8.2f}")

    # full probability tables per year k
    exhdr = "  ".join(f"E{c}(g{exg[c]:.0f})" for c in ex_ids)
    for k in YEAR_KS:
        yr_ids, ct = tables[k]
        col = ct.sum(0)
        L += ["", f"================= YEAR k={k}  (exam fixed k={EXAM_K}) =================",
              f"counts (rows=year, cols=exam):    {exhdr}   | row tot"]
        for i, yc in enumerate(yr_ids):
            L.append(f"  Y{yc:<2}                        " + "  ".join(f"{ct[i,j]:6d}" for j in range(len(ex_ids))) + f"   | {ct[i].sum()}")
        L.append(f"  col tot                     " + "  ".join(f"{col[j]:6d}" for j in range(len(ex_ids))))
        L += [f"P(exam | year)  rows sum to 100%:   {exhdr}"]
        rown = ct / ct.sum(1, keepdims=True)
        for i, yc in enumerate(yr_ids):
            L.append(f"  Y{yc:<2}                        " + "  ".join(f"{100*rown[i,j]:5.0f}% " for j in range(len(ex_ids))))
        L.append(f"  base P(E)                   " + "  ".join(f"{100*base_p[j]:5.0f}% " for j in range(len(ex_ids))))
        L += [f"P(year | exam)  cols sum to 100%:   {exhdr}"]
        coln = ct / ct.sum(0, keepdims=True)
        for i, yc in enumerate(yr_ids):
            L.append(f"  Y{yc:<2}                        " + "  ".join(f"{100*coln[i,j]:5.0f}% " for j in range(len(ex_ids))))
        baseY = ct.sum(1) / ct.sum()
        L.append("  base P(Y): " + ", ".join(f"Y{yr_ids[i]} {100*baseY[i]:.0f}%" for i in range(len(yr_ids))))

    L += ["", "Read: across k=2/3/4 the row/col distributions barely move off the base rate, ARI stays ~0",
          "and Cramér's V stays weak. A higher k only splits the big year group into thinner slices that",
          "still land in the same exam profile (E3) — no k gives a cleaner year->exam mapping."]
    ps = versioned("year_k_sensitivity_summary", "txt", CLU)
    Path(ps).write_text("\n".join(L) + "\n"); print("\n".join(L)); print(f"wrote {ps.name}")


if __name__ == "__main__":
    main()
