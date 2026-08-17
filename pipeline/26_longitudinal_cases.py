"""26 - Longitudinal (within-student) case-studies: the EVOLUTION the cross-sectional
case-studies (23) can't show.

23 shows exam medoids in the exam and year medoids in the year (different people, one
circumstance each). Behaviour-as-evolution is a property of ONE student across circumstances.
So here we ANCHOR on each EXAM medoid (the representative of E1..E4) and follow each of
them BACK INTO THE YEAR: their year profile cluster, their year episode-behaviour mix,
and their richest year trajectory + code snapshots. Put beside their exam mix, this shows how
a single student's behaviour changes from coursework to exam.

This also makes the weak year-cluster <-> exam-cluster correspondence concrete (Cramer's V=0.16,
ARI~0.07): an exam high-performer need not have been a year standout. A few anchors = an
ILLUSTRATION, not proof (recul).

Outputs -> case_studies/longitudinal/ (card + year snapshots)
figures -> plots/fig_long_case_{E1,E2,E3}_yearVsExam_v* , fig_long_case_{E1,E2,E3}_year_traj_v*
"""
import csv, glob, difflib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import cluster as canon_cluster, KMAP

REPO = Path(__file__).resolve().parents[3]
FEAT = REPO / "data/v2/res_python/features"
LINK = REPO / "data/v2/res_python/linkage"
CLU = REPO / "data/v2/res_python/clustering"
FIGS = REPO / "data/v2/res_python/plots"
CASE = REPO / "data/v2/res_python/case_studies/longitudinal"
EXAM_TRUE_GLOB = str(REPO / "data/v2/res_python/true_scores/exam_clean_true_v*.csv")
EXAM_CLEAN_GLOB = str(REPO / "data/v2/res_python/exam_audit/exam_clean_v*.csv")
EXAM_FLAG_GLOB = str(REPO / "data/v2/res_python/exam_audit/exam_flagged_hashes_v*.csv")
EXAM_BASE = REPO / "data/v2/last_archive/2026.exam/2026.01_comment"
MISS_GLOB = str(REPO / "data/v2/last_archive/2025.Q1/mission_*_comment/*/data.csv")
CASE23_GLOB = str(REPO / "data/v2/res_python/case_studies/exam/case_manifest_exam_v*.csv")  # medoid+contrast hashes from script 23

PASS = 50.0
BREAK_EDIT, BREAK_DSCORE = 3, 50.0
SHAPES = ["one-shot", "steady-climb", "breakthrough", "pass after reversals",
          "many tries, no pass", "few tries, no pass"]
SHAPE_COLORS = {"one-shot": "#009E73", "steady-climb": "#56B4E9", "breakthrough": "#0072B2",
                "pass after reversals": "#E69F00", "many tries, no pass": "#D55E00", "few tries, no pass": "#999999"}


def latest(pat): return sorted(glob.glob(str(pat)))[-1]


def versioned(stem, ext, base):
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"{stem}_v{n}.{ext}").exists(): n += 1
    return base / f"{stem}_v{n}.{ext}"


def read_lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.readlines()
    except FileNotFoundError:
        return None


def edit_size(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal")


def cluster(ds):
    """Canonical clustering (common.cluster): E1-E4 / Y1-Y3 by ascending mean exam grade.
    Returns hashes, Xs, labels, grade, per-cluster mean grade."""
    hashes, Xs, lab, grade, cols = canon_cluster(ds)
    gmean = {int(c): float(grade[lab == c].mean()) for c in sorted(set(lab))}
    return hashes, Xs, lab, grade, gmean


def medoid(hashes, Xs, lab, c):
    idx = np.where(lab == c)[0]
    d = np.linalg.norm(Xs[idx] - Xs[idx].mean(0), axis=1)
    return hashes[idx[int(np.argmin(d))]]


def records(ds, keep):
    recs = []
    if ds == "exam":
        _t = sorted(glob.glob(EXAM_TRUE_GLOB))
        for r in csv.DictReader(open(_t[-1] if _t else latest(EXAM_CLEAN_GLOB))):
            if r["hash"] in keep:
                recs.append((r["hash"], r["qname"], int(r["n_submission"]), float(r["score"]),
                             EXAM_BASE / r["qname"] / "code" / r["file"]))
    else:
        staff = {r["hash"] for r in csv.DictReader(open(latest(EXAM_FLAG_GLOB)))}
        for p in sorted(glob.glob(MISS_GLOB)):
            qdir = Path(p).parent; mission = qdir.parent.name
            for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                if r["hash"] in keep and r["hash"] not in staff:
                    recs.append((r["hash"], mission + "/" + r["qname"], int(r["n_submission"]),
                                 float(r["score"]), qdir / "code" / r["file"]))
    return recs


def episode_steps(seq):
    """seq: (n, score, path) -> ordered steps with edit sizes; and (scores, edits)."""
    seq = sorted(seq, key=lambda x: x[0])
    steps, prev = [], None
    for n, sc, path in seq:
        lines = read_lines(path)
        if lines is None:
            continue
        es = edit_size(prev, lines) if prev is not None else 0
        dsc = sc - steps[-1]["score"] if steps else 0.0
        steps.append({"n": n, "score": sc, "edit": es, "dscore": dsc, "lines": lines})
        prev = lines
    return steps


def classify(steps):
    if not steps:
        return None
    scores = [s["score"] for s in steps]
    nsub = len(scores); best = max(scores)
    n_drop = sum(1 for i in range(1, len(scores)) if scores[i] < scores[i - 1])
    has_break = any(s["edit"] <= BREAK_EDIT and s["dscore"] >= BREAK_DSCORE for s in steps[1:])
    if nsub <= 2 and best >= PASS: return "one-shot"
    if best < PASS and nsub <= 3: return "few tries, no pass"
    if best < PASS and nsub >= 4: return "many tries, no pass"
    if best >= PASS and has_break: return "breakthrough"
    if best >= PASS and n_drop >= 2: return "pass after reversals"
    return "steady-climb"


def mix_and_episodes(recs, h):
    by_q = defaultdict(list)
    for hh, q, n, sc, path in recs:
        if hh == h:
            by_q[q].append((n, sc, path))
    mix = Counter(); episodes = {}
    for q, seq in by_q.items():
        steps = episode_steps(seq)
        shp = classify(steps)
        if shp is None:
            continue
        mix[shp] += 1
        episodes[q] = (steps, shp)
    return mix, episodes


def richest(episodes):
    """The (question, steps) with the most submissions; tie -> higher final score."""
    if not episodes:
        return None, None
    q = max(episodes, key=lambda q: (len(episodes[q][0]), episodes[q][0][-1]["score"]))
    return q, episodes[q][0]


def telling(steps):
    first, final = 0, len(steps) - 1
    jump = int(np.argmax([s["dscore"] for s in steps])) if len(steps) > 1 else 0
    keep = sorted(set([first, jump, final]))
    tag = {first: "first", final: "final"}; tag.setdefault(jump, "breakthrough")
    return [(i, tag.get(i, "step")) for i in keep]


def year_trajectory(steps, keep, tag, cid, gmean_year, q):
    ns = [s["n"] for s in steps]; sc = [s["score"] for s in steps]; ed = [s["edit"] for s in steps]
    fig, ax1 = plt.subplots(figsize=(7.5, 3.8))
    ax1.plot(ns, sc, marker="o", lw=1.5, color="#0072B2"); ax1.set_ylim(-3, 103)
    ax1.set_xlabel("submission # (year)"); ax1.set_ylabel("true score", color="#0072B2")
    ax2 = ax1.twinx(); ax2.bar(ns, ed, width=0.6, color="#E69F00", alpha=0.45)
    ax2.set_ylabel("edit size", color="#E69F00")
    for i, tg in keep:
        ax1.annotate(tg, (steps[i]["n"], steps[i]["score"]), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
    ax1.set_title(f"{tag} medoid: YEAR trajectory (year cluster Y{cid}, grade {gmean_year[cid]:.0f}) on {q}",
                  fontsize=9)
    p = versioned(f"fig_long_case_{tag}_year_traj", "pdf", FIGS)
    for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig); return p.stem


def main():
    ex_h, ex_X, ex_lab, ex_g, ex_gm = cluster("exam")
    yr_h, yr_X, yr_lab, yr_g, yr_gm = cluster("year")
    yr_lab_by_h = dict(zip(yr_h, yr_lab))
    ex_lab_by_h = dict(zip(ex_h, ex_lab))

    ex_cids = sorted(set(ex_lab))                                   # exam clusters (k from KMAP)
    man = list(csv.DictReader(open(latest(CASE23_GLOB))))           # medoid + contrast per cluster (script 23)
    cases = [(int(r["cluster"]), r["role"], r["hash"]) for r in man]
    anchors = {c: h for c, role, h in cases if role == "medoid"}    # one medoid per exam cluster (for the card)
    keep = {h for _, _, h in cases}                                 # all 8 case students
    ex_recs = records("exam", keep)
    yr_recs = records("year", keep)

    L = [f"Longitudinal case-studies: the {len(ex_cids)} EXAM medoids followed back into the YEAR.",
         f"Shows one student's behaviour across circumstances (year -> exam); {len(ex_cids)} anchors = illustration, not proof.", ""]
    mixes = {}
    for c in ex_cids:
        h = anchors[c]
        ex_mix, _ = mix_and_episodes(ex_recs, h)
        yr_mix, yr_eps = mix_and_episodes(yr_recs, h)
        mixes[c] = (yr_mix, ex_mix)
        yc = yr_lab_by_h.get(h, None)
        # year trajectory + snapshots for this anchor
        q, steps = richest(yr_eps)
        traj = None
        if steps and len(steps) >= 2 and yc is not None:
            keeps = telling(steps)
            traj = year_trajectory(steps, keeps, f"E{c}", yc, yr_gm, q)
            CASE.mkdir(parents=True, exist_ok=True)
            qsafe = q.replace("/", "_")
            for i, tg in keeps:
                s = steps[i]
                (CASE / f"E{c}_{h[:8]}_YEAR_{qsafe}_n{s['n']}_{tg.replace(' ', '')}.py").write_text(
                    "".join(s["lines"]), encoding="utf-8")
        L.append(f"=== EXAM cluster E{c} (grade {ex_gm[c]:.0f}) — medoid {h} ===")
        L.append(f"  YEAR profile : Y{yc} (grade {yr_gm[yc]:.0f})   |   EXAM profile : E{c} (grade {ex_gm[c]:.0f})")
        L.append(f"  YEAR episode mix : " + ", ".join(f"{s} {yr_mix.get(s,0)}" for s in SHAPES if yr_mix.get(s, 0)))
        L.append(f"  EXAM episode mix : " + ", ".join(f"{s} {ex_mix.get(s,0)}" for s in SHAPES if ex_mix.get(s, 0)))
        L.append(f"  year total episodes {sum(yr_mix.values())}, exam total {sum(ex_mix.values())}"
                 + (f"; year trajectory fig {traj}" if traj else "; (no multi-attempt year trace)"))
        L.append("")
    ps = versioned("longitudinal_cases", "txt", CASE); Path(ps).write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"wrote {ps}")

    # one figure per case student (medoid AND contrast): year mix vs exam mix (share).
    # Title names the year and exam GROUPS (no group-mean grades, which read as the student's own).
    for c, role, h in cases:
        yr_mix, _ = mix_and_episodes(yr_recs, h)
        ex_mix, _ = mix_and_episodes(ex_recs, h)
        yc = yr_lab_by_h.get(h)
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        bottom = np.zeros(2)
        for s in SHAPES:
            yv = 100 * yr_mix.get(s, 0) / (sum(yr_mix.values()) or 1)
            ev = 100 * ex_mix.get(s, 0) / (sum(ex_mix.values()) or 1)
            ax.bar(["year", "exam"], [yv, ev], bottom=bottom, color=SHAPE_COLORS[s], label=s)
            bottom += [yv, ev]
        ax.set_ylim(0, 100); ax.set_ylabel("% of the student's episodes")
        ax.set_title(f"E{c} {role}: year group Y{yc} → exam group E{c}", fontsize=9)
        ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        p = versioned(f"fig_long_case_E{c}_{role}_yearVsExam", "pdf", FIGS)
        for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
        plt.close(fig); print(f"fig {p.stem}")


if __name__ == "__main__":
    main()
