"""25 - Episode-level behaviour (the construct in feedback_tfe/BEHAVIOUR_DEFINITION.md).

behaviour := the SHAPE of one (student, question) attempt-sequence = an EPISODE. This is the
order-sensitive object the thesis is really about; per-student aggregate clusters are a coarser
PROFILE. Here we (a) classify every episode into a small vocabulary of shapes, (b) report the
shape mix per CIRCUMSTANCE (exam vs year), (c) show each profile's mix of episode
shapes, and (d) check that the 'breakthrough' shape is not merely an artefact of questions
with a high pass rate (per-question 'breakthrough' share vs per-question pass rate).
Clustering comes from common.cluster (canonical: exam k=4, year k=3, labels by ascending mean exam
grade) so E1-E4 / Y1-Y3 match Ch4.

Shape names are DESCRIPTIVE of the observable pattern only (no inferred intent). An episode = the
ordered submissions for one (student, question), each with true score + edit size (difflib vs the
previous submission). Classified (first match wins), PASS = score >= 50:
  one-shot        nsub<=2 and best>=PASS       reached pass within the first two submissions
  few tries, no pass    best<PASS and nsub<=3        three or fewer submissions, never reached pass
  many tries, no pass   best<PASS and nsub>=4        four or more submissions, never reached pass
  breakthrough   best>=PASS and a small edit (<=3 lines) preceded a >=50 score rise
  pass after reversals  best>=PASS and the score went down >=2 times before the end
  steady-climb          best>=PASS (the rest)        reached pass through gradual score rises

Outputs -> clustering/episode_archetypes_summary_v*.txt
figures -> plots/fig_episode_shapes_by_circumstance_v* , fig_profile_episode_mix_{exam,year}_v*,
           fig_smalledit_pass_vs_passrate_{exam,year}_v*
"""
import csv, glob, difflib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import cluster as canon_cluster

REPO = Path(__file__).resolve().parents[3]
CLU = REPO / "data/v2/res_python/clustering"
FIGS = REPO / "data/v2/res_python/plots"
EXAM_TRUE_GLOB = str(REPO / "data/v2/res_python/true_scores/exam_clean_true_v*.csv")
EXAM_CLEAN_GLOB = str(REPO / "data/v2/res_python/exam_audit/exam_clean_v*.csv")
EXAM_FLAG_GLOB = str(REPO / "data/v2/res_python/exam_audit/exam_flagged_hashes_v*.csv")
EXAM_BASE = REPO / "data/v2/last_archive/2026.exam/2026.01_comment"
MISS_GLOB = str(REPO / "data/v2/last_archive/2025.Q1/mission_*_comment/*/data.csv")

PASS = 50.0
BREAK_EDIT, BREAK_DSCORE = 3, 50.0
MIN_Q_EP = 20   # min episodes on a question to include it in the pass-rate check
SMALL_EDIT = "breakthrough"
SHAPES = ["one-shot", "steady-climb", "breakthrough", "pass after reversals",
          "many tries, no pass", "few tries, no pass"]
SHAPE_COLORS = {  # Okabe-Ito, CVD-safe; green/blue = reached pass, red/grey = never reached pass
    "one-shot": "#009E73", "steady-climb": "#56B4E9", "breakthrough": "#0072B2",
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


def records(ds, keep_hashes):
    recs = []
    if ds == "exam":
        _t = sorted(glob.glob(EXAM_TRUE_GLOB))
        rows = list(csv.DictReader(open(_t[-1] if _t else latest(EXAM_CLEAN_GLOB))))
        for r in rows:
            if r["hash"] in keep_hashes:
                recs.append((r["hash"], r["qname"], int(r["n_submission"]), float(r["score"]),
                             EXAM_BASE / r["qname"] / "code" / r["file"]))
    else:
        staff = {r["hash"] for r in csv.DictReader(open(latest(EXAM_FLAG_GLOB)))}
        for p in sorted(glob.glob(MISS_GLOB)):
            qdir = Path(p).parent; mission = qdir.parent.name
            for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                if r["hash"] in staff or r["hash"] not in keep_hashes:
                    continue
                recs.append((r["hash"], mission + "/" + r["qname"], int(r["n_submission"]),
                             float(r["score"]), qdir / "code" / r["file"]))
    return recs


def classify(seq):
    """seq: list of (n, score, path) for one episode, any order. -> shape name or None.
    Names describe the observable pattern only (tries, whether pass reached, edit/score shape)."""
    seq = sorted(seq, key=lambda x: x[0])
    scores, prev, edits = [], None, []
    for _, sc, path in seq:
        lines = read_lines(path)
        if lines is None:
            continue
        if prev is not None:
            edits.append((edit_size(prev, lines), sc - scores[-1]))
        scores.append(sc); prev = lines
    if not scores:
        return None
    nsub = len(scores); best = max(scores)
    n_drop = sum(1 for i in range(1, len(scores)) if scores[i] < scores[i - 1])
    small_edit_gain = any(e <= BREAK_EDIT and d >= BREAK_DSCORE for e, d in edits)
    if nsub <= 2 and best >= PASS:
        return "one-shot"
    if best < PASS and nsub <= 3:
        return "few tries, no pass"
    if best < PASS and nsub >= 4:
        return "many tries, no pass"
    if best >= PASS and small_edit_gain:
        return "breakthrough"
    if best >= PASS and n_drop >= 2:
        return "pass after reversals"
    return "steady-climb"


def cluster_labels(ds):
    """Canonical clustering (common.cluster): E1-E4 / Y1-Y3 by ascending mean exam grade."""
    hashes, Xs, lab, grade, cols = canon_cluster(ds)
    lab_map = dict(zip(hashes, lab.tolist()))
    gmean = {int(c): float(grade[lab == c].mean()) for c in sorted(set(lab))}
    gmed = {int(c): float(np.median(grade[lab == c])) for c in sorted(set(lab))}
    return lab_map, gmean, gmed


def episodes(ds):
    lab, gmean, gmed = cluster_labels(ds)
    recs = records(ds, set(lab))
    by_ep = defaultdict(list)
    for h, q, n, sc, path in recs:
        by_ep[(h, q)].append((n, sc, path))
    overall = Counter()
    per_cluster = {c: Counter() for c in sorted(set(lab.values()))}
    by_q_shapes = defaultdict(Counter)          # question -> shape counts
    by_q_pass = defaultdict(lambda: [0, 0])      # question -> [n_episodes, n_passed]
    n_ep = 0
    for (h, q), seq in by_ep.items():
        shape = classify(seq)
        if shape is None:
            continue
        overall[shape] += 1; per_cluster[lab[h]][shape] += 1; n_ep += 1
        by_q_shapes[q][shape] += 1
        best = max(sc for _, sc, _ in seq)
        by_q_pass[q][0] += 1; by_q_pass[q][1] += (1 if best >= PASS else 0)
    return dict(overall=overall, per_cluster=per_cluster, gmean=gmean, gmed=gmed,
                n_ep=n_ep, n_students=len(lab),
                by_q_shapes=by_q_shapes, by_q_pass=by_q_pass)


def shares(counter):
    tot = sum(counter.values()) or 1
    return {s: 100 * counter.get(s, 0) / tot for s in SHAPES}


def passrate_rows(res):
    """Per question with >= MIN_Q_EP episodes: (q, n, pass_rate%, small_edit_pass_share%)."""
    rows = []
    for q, cnt in res["by_q_shapes"].items():
        n = sum(cnt.values())
        if n < MIN_Q_EP:
            continue
        npass = res["by_q_pass"][q][1]
        rows.append((q, n, 100 * npass / n, 100 * cnt.get(SMALL_EDIT, 0) / n))
    return rows


def main():
    ex = episodes("exam")
    yr = episodes("year")

    L = ["Episode-level behaviour shapes (order-sensitive). behaviour = one (student,question) attempt-sequence.",
         f"PASS = score >= {PASS:.0f}. Shapes (observable patterns): {', '.join(SHAPES)}.",
         "Clusters from common.cluster (exam k=4, year k=3).", ""]
    for name, res in [("EXAM", ex), ("YEAR", yr)]:
        n = res["n_ep"]
        L.append(f"=== {name}: {n} episodes over {res['n_students']} students -- shape distribution (circumstance) ===")
        for s in SHAPES:
            L.append(f"    {s:<22} {res['overall'].get(s,0):6d}  ({100*res['overall'].get(s,0)/(n or 1):5.1f}%)")
        L.append("")
    L.append("--- Profile cluster x episode-shape mix (row % of that cluster's episodes) ---")
    for name, res in [("EXAM", ex), ("YEAR", yr)]:
        pc, gm, gmd = res["per_cluster"], res["gmean"], res["gmed"]
        L.append(f"  {name}:")
        L.append("    cluster                    " + "  ".join(f"{s[:11]:>11}" for s in SHAPES))
        for c in sorted(pc):
            sh = shares(pc[c])
            L.append(f"    {name[0]}{c} (grade {gm[c]:>2.0f}/med {gmd[c]:>2.0f})  " + "  ".join(f"{sh[s]:10.1f}%" for s in SHAPES))
        L.append("")

    # (d) 'breakthrough' vs question pass rate: does this shape only appear where the pass rate is high?
    L.append("--- 'breakthrough' share vs question pass rate (does the shape only appear on high-pass-rate questions?) ---")
    diff = {}
    for name, res in [("EXAM", ex), ("YEAR", yr)]:
        rows = passrate_rows(res); diff[name] = rows
        if not rows:
            L.append(f"  {name}: no question with >= {MIN_Q_EP} episodes."); continue
        pr = np.array([r[2] for r in rows]); bs = np.array([r[3] for r in rows])
        med = np.median(pr)
        low = bs[pr <= med]; high = bs[pr > med]   # low pass-rate half vs high pass-rate half
        rho = np.corrcoef(pr, bs)[0, 1] if pr.std() > 0 and bs.std() > 0 else float("nan")
        L.append(f"  {name}: {len(rows)} questions (>= {MIN_Q_EP} episodes). "
                 f"'breakthrough' share on the LOWER-pass-rate half = {low.mean():.1f}%, "
                 f"on the HIGHER-pass-rate half = {high.mean():.1f}%; corr(pass rate, share) r={rho:+.2f}.")
    L.append("  note: reaching the pass mark is part of this shape, so some link with the pass rate is expected;"
             " the point is whether the shape still appears where the pass rate is lower.")
    L.append("")

    ps = versioned("episode_archetypes_summary", "txt", CLU); Path(ps).write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"wrote {ps.name}")

    # figure A: shape distribution exam vs year (circumstance)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(SHAPES)); w = 0.38
    ax.bar(x - w/2, [100*ex["overall"].get(s,0)/(ex["n_ep"] or 1) for s in SHAPES], w, label=f"exam ({ex['n_ep']} episodes)", color="#0072B2")
    ax.bar(x + w/2, [100*yr["overall"].get(s,0)/(yr["n_ep"] or 1) for s in SHAPES], w, label=f"year ({yr['n_ep']} episodes)", color="#E69F00")
    ax.set_xticks(x); ax.set_xticklabels(SHAPES, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("% of episodes"); ax.legend(frameon=False)
    ax.set_title("Episode shapes by setting (exam vs coursework)", fontsize=12)
    p = versioned("fig_episode_shapes_by_circumstance", "pdf", FIGS)
    for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig); print(f"fig {p.stem}")

    # figure B: per dataset, stacked bar of each profile cluster's episode-shape mix
    for name, res in [("exam", ex), ("year", yr)]:
        pc, gm = res["per_cluster"], res["gmean"]
        clusters = sorted(pc)
        fig, ax = plt.subplots(figsize=(7.8, 4.2))
        bottom = np.zeros(len(clusters))
        for s in SHAPES:
            vals = np.array([shares(pc[c])[s] for c in clusters])
            ax.bar([f"{name[0].upper()}{c}\n(grade {gm[c]:.0f})" for c in clusters], vals, bottom=bottom,
                   label=s, color=SHAPE_COLORS[s])
            bottom += vals
        ax.set_ylabel("% of the cluster's episodes"); ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.set_title(f"{name.upper()}: each profile's mix of episode shapes", fontsize=12)
        p = versioned(f"fig_profile_episode_mix_{name}", "pdf", FIGS)
        for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
        plt.close(fig); print(f"fig {p.stem}")

    # figure C: 'breakthrough' share vs question pass rate (the pass-rate check)
    for name in ("EXAM", "YEAR"):
        rows = diff[name]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        pr = [r[2] for r in rows]; bs = [r[3] for r in rows]
        ax.scatter(pr, bs, s=[min(6 + r[1] / 30, 120) for r in rows], color="#0072B2", alpha=0.7, edgecolor="white")
        ax.set_xlabel("question pass rate (%)"); ax.set_ylabel("'breakthrough' share (%)")
        ax.set_xlim(0, 100); ax.set_ylim(bottom=0)
        ax.set_title(f"{name.title()}: 'breakthrough' share vs question pass rate\n(each point a question; marker size = episodes)", fontsize=10)
        p = versioned(f"fig_smalledit_pass_vs_passrate_{name.lower()}", "pdf", FIGS)
        for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
        plt.close(fig); print(f"fig {p.stem}")


if __name__ == "__main__":
    main()
