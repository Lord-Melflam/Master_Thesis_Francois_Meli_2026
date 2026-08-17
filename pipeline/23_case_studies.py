"""23 - Qualitative code case-studies: put a FACE on each behaviour cluster.

Answers the supervisor asks (qualitative):
  - pick a REPRESENTATIVE student (the medoid), not a cherry-picked one;
  - read the DYNAMIC (the ordered submission sequence), not a static snapshot;
  - show only the telling submissions (first / breakthrough / big rewrite / final);
  - method: medoid -> student -> subset of submissions -> two competing hypotheses -> which the trace leans to;
  - "same feature signature could be two different behaviours" -> we show TWO students per cluster:
       the medoid (most typical) AND a contrasting case that leans to the other hypothesis.

Clustering comes from common.cluster (canonical: exam k=4, year k=3, labels by ascending mean exam
grade) so E1-E4 / Y1-Y3 match Chapter 4. A medoid ILLUSTRATES a cluster, it does not prove it (recul).

Outputs -> data/v2/res_python/case_studies/<ds>/  (cards, code snapshots)
figures -> data/v2/res_python/plots/fig_case_<ds>_<cluster>_<role>_trajectory_v*
"""
import csv, glob, sys, difflib, unicodedata
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import cluster as canon_cluster, KMAP

# one colour per exam group (matches \definecolor caseE1..E4 in the report)
CASE_HEX = {1: "#E69F00", 2: "#56B4E9", 3: "#009E73", 4: "#0072B2"}


def _ascii(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

REPO = Path(__file__).resolve().parents[3]
CLU = REPO / "data/v2/res_python/clustering"
FIGS = REPO / "data/v2/res_python/plots"
CASE = REPO / "data/v2/res_python/case_studies"
EXAM_TRUE_GLOB = str(REPO / "data/v2/res_python/true_scores/exam_clean_true_v*.csv")
EXAM_CLEAN_GLOB = str(REPO / "data/v2/res_python/exam_audit/exam_clean_v*.csv")
EXAM_FLAG_GLOB = str(REPO / "data/v2/res_python/exam_audit/exam_flagged_hashes_v*.csv")
EXAM_BASE = REPO / "data/v2/last_archive/2026.exam/2026.01_comment"
MISS_GLOB = str(REPO / "data/v2/last_archive/2025.Q1/mission_*_comment/*/data.csv")

N_CAND = 25   # how many nearest-to-centroid members to scan for a contrasting case


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


def nloc(lines):
    return sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))


def by_hash_records(ds):
    """hash -> list of (qname, n_submission, score, path)."""
    bh = defaultdict(list)
    if ds == "exam":
        _t = sorted(glob.glob(EXAM_TRUE_GLOB))
        rows = list(csv.DictReader(open(_t[-1] if _t else latest(EXAM_CLEAN_GLOB))))
        for r in rows:
            bh[r["hash"]].append((r["qname"], int(r["n_submission"]), float(r["score"]),
                                  EXAM_BASE / r["qname"] / "code" / r["file"]))
    else:
        staff = {r["hash"] for r in csv.DictReader(open(latest(EXAM_FLAG_GLOB)))}
        for p in sorted(glob.glob(MISS_GLOB)):
            qdir = Path(p).parent; mission = qdir.parent.name
            for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                if r["hash"] in staff:
                    continue
                bh[r["hash"]].append((mission + "/" + r["qname"], int(r["n_submission"]),
                                      float(r["score"]), qdir / "code" / r["file"]))
    return bh


def trace(bh, h):
    """Richest-trace question for student h -> (q, steps, dist). steps = ordered attempts w/ metrics.
    The richest-trace question is the student's MOST-iterated one (chosen to reveal a dynamic), not
    their typical one; dist reports their per-question attempt spread to make that explicit."""
    by_q = defaultdict(list)
    for q, n, sc, path in bh.get(h, []):
        by_q[q].append((n, sc, path))
    if not by_q:
        return None, None, None
    counts = sorted(len(v) for v in by_q.values())
    dist = {"n_questions": len(counts), "min": counts[0],
            "median": int(np.median(counts)), "max": counts[-1]}
    def keyf(q):
        seq = sorted(by_q[q]); return (len(seq), seq[-1][1])
    q = max(by_q, key=keyf)
    seq = sorted(by_q[q])
    steps, prev = [], None
    for n, sc, path in seq:
        lines = read_lines(path)
        if lines is None:
            continue
        es = edit_size(prev, lines) if prev is not None else 0
        dsc = (sc - steps[-1]["score"]) if steps else 0.0
        steps.append({"n": n, "score": sc, "edit": es, "dscore": dsc, "nloc": nloc(lines), "lines": lines})
        prev = lines
    return q, steps, dist


def telling(steps):
    """Indices of FIRST, biggest score-JUMP, biggest REWRITE, FINAL attempts."""
    first, final = 0, len(steps) - 1
    jump = int(np.argmax([s["dscore"] for s in steps])) if len(steps) > 1 else 0
    rewrite = int(np.argmax([s["edit"] for s in steps])) if len(steps) > 1 else 0
    keep = sorted(set([first, jump, rewrite, final]))
    tag = {first: "first", final: "final"}
    tag.setdefault(jump, "breakthrough"); tag.setdefault(rewrite, "largest edit")
    return [(i, tag.get(i, "step")) for i in keep]


def hypotheses(ds, c, k, steps):
    """Two competing readings for the cluster + which the trace leans to. A medoid ILLUSTRATES."""
    n = len(steps)
    churn = float(np.mean([(s["edit"] >= 5 and s["dscore"] <= 0) for s in steps[1:]])) if n > 1 else 0.0
    total_gain = steps[-1]["score"] - steps[0]["score"]
    final = steps[-1]["score"]
    tiny_big = any(s["edit"] <= 3 and s["dscore"] >= 50 for s in steps)
    if ds == "exam" and c == 1:            # lowest grade
        H1 = "few submissions and little code before stopping"
        H2 = "many submissions, but the score never reached the pass mark"
        lean = "H1" if n <= 3 else ("H2" if final < 50 else "mixed")
    elif ds == "exam" and c == k:          # top cluster
        H1 = "a high score is reached early, with almost no later edits"
        H2 = "the pass mark is reached through small, targeted edits (a small edit precedes a large score rise)"
        lean = "H2" if tiny_big else "H1"
    else:                                  # middle / year clusters
        H1 = "edits are followed by net score gains"
        H2 = "many edits with little net change in score"
        lean = "H1" if total_gain >= 50 else ("H2" if churn >= 0.15 else "mixed")
    return H1, H2, lean, churn


def snapshot_code(steps, keep, outdir, ds, c, role, h, q):
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []
    qsafe = q.replace("/", "_")
    for i, tg in keep:
        s = steps[i]
        fn = outdir / f"{ds}_{ds[0].upper()}{c}_{role}_{h[:8]}_{qsafe}_n{s['n']}_{tg.replace(' ', '')}.py"
        fn.write_text("".join(s["lines"]), encoding="utf-8")
        saved.append((tg, s["n"], fn.name))
    return saved


def key_transition(steps):
    """Index j (>=1) of the decisive step: largest |score change|; ties / all-zero -> largest edit.
    The decisive edit is between submission j-1 and j."""
    if len(steps) < 2:
        return None
    return max(range(1, len(steps)), key=lambda j: (abs(steps[j]["dscore"]), steps[j]["edit"]))


def _strip_run(run_lines):
    """Strip one maximal run of removed (-) lines: drop blanks, comments, and docstring blocks
    whose opening AND closing triple-quote both lie inside this run (a lone quote, e.g. a
    docstring closed from outside the diff window, is kept so real code is never lost)."""
    contents = [l[1:] for l in run_lines]
    keep = [True] * len(contents)
    for i, c in enumerate(contents):
        s = c.strip()
        if s == "" or s.startswith("#"):
            keep[i] = False
    i = 0
    while i < len(contents):
        s = contents[i].strip()
        q = '"""' if s.startswith('"""') else ("'''" if s.startswith("'''") else None)
        if q:
            if s.count(q) >= 2:                 # one-line docstring
                keep[i] = False; i += 1; continue
            j = i + 1
            while j < len(contents) and q not in contents[j]:
                j += 1
            if j < len(contents):               # balanced open..close inside the run -> drop it
                for k in range(i, j + 1):
                    keep[k] = False
                i = j + 1; continue
        i += 1
    return [run_lines[i] for i in range(len(run_lines)) if keep[i]]


def _strip_diff(body):
    """For a large diff, keep EVERY added (+) line and every context line, but drop the removed
    (-) lines that carry no information: blank lines, comments, and self-contained docstring
    blocks. Real removed code lines are kept (they show what an edit broke or replaced). Matches
    the report rule 'strip the (-) where not necessary, never the (+)'."""
    out, buf = [], []
    for ln in body:
        if ln.startswith("-"):
            buf.append(ln)
            continue
        if buf:
            out.extend(_strip_run(buf)); buf = []
        out.append(ln)
    if buf:
        out.extend(_strip_run(buf))
    return out


def emit_diff(steps, outdir, ds, c, role):
    """Write the decisive edit as a unified diff (ASCII, headers stripped) for colouring in the
    report; return (path, meta) with the before/after submission numbers, scores and edit size.
    Diffs longer than 5 changed lines are passed through _strip_diff (keep every +, drop
    unnecessary -)."""
    j = key_transition(steps)
    if j is None:
        return None, None
    a, b = steps[j - 1]["lines"], steps[j]["lines"]
    diff = difflib.unified_diff(a, b, n=2, lineterm="")
    body = [_ascii(ln.rstrip("\n")) for ln in diff if not ln.startswith(("---", "+++", "@@"))]
    if sum(1 for l in body if l[:1] in "+-") > 5:
        body = _strip_diff(body)
    dd = outdir / "diffs"; dd.mkdir(parents=True, exist_ok=True)
    p = dd / f"{ds}_{ds[0].upper()}{c}_{role}.diff"
    p.write_text("\n".join(body) + "\n", encoding="ascii")
    meta = {"n0": steps[j - 1]["n"], "n1": steps[j]["n"], "s0": steps[j - 1]["score"],
            "s1": steps[j]["score"], "edit": steps[j]["edit"], "nlines": len(body)}
    return p, meta


def trajectory_fig(steps, keep, ds, c, role, h, grade_mean, grade_med, n_cluster, q):
    color = CASE_HEX.get(c, "#444444")
    ns = [s["n"] for s in steps]; sc = [s["score"] for s in steps]; ed = [s["edit"] for s in steps]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.0))
    ax1.plot(ns, sc, marker="o", lw=1.6, color="#0072B2", label="score")
    ax1.set_xlabel("submission #"); ax1.set_ylabel("score", color="#0072B2"); ax1.set_ylim(-16, 116)
    ax2 = ax1.twinx()
    ax2.bar(ns, ed, width=0.6, color="#E69F00", alpha=0.45, label="edit size (lines changed)")
    ax2.set_ylabel("edit size (lines vs previous)", color="#E69F00")
    # stagger the telling-point labels (alternate above/below, ordered left to right) so they don't overlap
    for k, (i, tg) in enumerate(sorted(keep)):
        dy = 11 if k % 2 == 0 else -13
        ax1.annotate(tg, (steps[i]["n"], steps[i]["score"]), textcoords="offset points",
                     xytext=(0, dy), ha="center", va=("bottom" if dy > 0 else "top"), fontsize=7.5)
    roled = {"medoid": "representative (medoid)", "contrast": "contrasting case"}[role]
    ax1.set_title(f"{ds.upper()} {ds[0].upper()}{c} (hash {h[:8]}, n={n_cluster}, grade mean {grade_mean:.0f} / median {grade_med:.0f}): "
                  f"{roled} on {q}", fontsize=9, color=color)
    # frame the plot in the case colour so figure, code box and hash share one colour
    for sp in list(ax1.spines.values()) + list(ax2.spines.values()):
        sp.set_edgecolor(color); sp.set_linewidth(1.8)
    p = versioned(f"fig_case_{ds}_{ds[0].upper()}{c}_{role}_trajectory", "pdf", FIGS)
    for e in ("pdf", "png"): fig.savefig(p.with_suffix("." + e), bbox_inches="tight")
    plt.close(fig); return p.stem


def emit_case(L, ds, c, k, role, h, nclu, gmean, gmed, bh, outdir):
    q, steps, dist = trace(bh, h)
    roled = {"medoid": "representative (medoid)", "contrast": "contrasting case"}[role]
    if not steps or len(steps) < 2:
        L.append(f"  [{roled}] medoid {h[:8]} has no multi-attempt trace; skipped."); return None
    keep = telling(steps)
    H1, H2, lean, churn = hypotheses(ds, c, k, steps)
    saved = snapshot_code(steps, keep, outdir, ds, c, role, h, q)
    stem = trajectory_fig(steps, keep, ds, c, role, h, gmean, gmed, nclu, q)
    dpath, dmeta = emit_diff(steps, outdir, ds, c, role)
    L.append(f"  --- {roled}: student {h} ---")
    L.append(f"     attempts/question: {dist['n_questions']} questions, min {dist['min']} / median {dist['median']} / max {dist['max']} "
             f"(trace below = their MOST-iterated question, not their typical one)")
    L.append(f"     richest-trace question: {q}  ({len(steps)} attempts, final {steps[-1]['score']:.0f}, churn {churn:.2f})")
    tagmap = dict(keep)
    L.append("     per-attempt trace (n: score | edit | code-lines):")
    for i, s in enumerate(steps):
        tg = tagmap.get(i, "")
        L.append(f"       n={s['n']:>3}: score {s['score']:6.1f} | edit {s['edit']:>3} | nloc {s['nloc']:>3}"
                 + (f"   <- {tg}" if tg else ""))
    L.append(f"     telling snapshots: " + ", ".join(f"{tg}(n={n})->{fn}" for tg, n, fn in saved))
    L.append(f"     trajectory figure: {stem}")
    if dmeta:
        L.append(f"     decisive edit: submission {dmeta['n0']}->{dmeta['n1']}, score {dmeta['s0']:.0f}->{dmeta['s1']:.0f}, "
                 f"{dmeta['edit']} lines changed; diff {dpath.name} ({dmeta['nlines']} diff lines)")
    L.append(f"     Hypothesis 1: {H1}")
    L.append(f"     Hypothesis 2: {H2}")
    L.append(f"     this trace leans toward: {lean}")
    return lean


def run(ds):
    k = KMAP[ds]
    hashes, Xs, lab, grade, cols = canon_cluster(ds)
    bh = by_hash_records(ds)
    outdir = CASE / ds
    L = [f"CASE STUDIES -- {ds.upper()} (canonical k={k}; medoid = student nearest cluster centroid).",
         "TWO students per cluster: the medoid (most typical) and a CONTRASTING case that leans to the",
         "other hypothesis -- same signature can arise from different episode behaviours. A case ILLUSTRATES,",
         "it does not prove (recul).", ""]
    manifest = []   # (cluster, role, hash) for the chosen medoid + contrast per cluster; read by script 26
    for c in range(1, k + 1):
        idx = np.where(lab == c)[0]
        centroid = Xs[idx].mean(0)
        order = idx[np.argsort(np.linalg.norm(Xs[idx] - centroid, axis=1))]   # nearest-to-centroid first
        gmean = float(grade[lab == c].mean()); gmed = float(np.median(grade[lab == c]))
        L.append(f"=== {ds[0].upper()}{c}  n={len(idx)}, grade mean {gmean:.0f} / median {gmed:.0f} ===")
        # case A = medoid = nearest-to-centroid with a valid multi-attempt trace
        a_h = None; a_lean = None
        for j in order:
            q, steps, _ = trace(bh, hashes[j])
            if steps and len(steps) >= 2:
                a_h = hashes[j]; break
        if a_h is None:
            L.append("  no member with a multi-attempt trace; skipped.\n"); continue
        a_lean = emit_case(L, ds, c, k, "medoid", a_h, len(idx), gmean, gmed, bh, outdir)
        manifest.append((c, "medoid", a_h))
        # case B = a contrasting case: nearest-to-centroid member whose trace leans to the OTHER reading
        b_h = None
        for j in order[:N_CAND]:
            hh = hashes[j]
            if hh == a_h:
                continue
            q, steps, _ = trace(bh, hh)
            if not steps or len(steps) < 2:
                continue
            _, _, lean, _ = hypotheses(ds, c, k, steps)
            if a_lean is not None and lean != a_lean and lean != "mixed":
                b_h = hh; break
        if b_h is None:   # fallback: the farthest-from-centroid member with a valid trace
            for j in order[::-1]:
                hh = hashes[j]
                if hh == a_h:
                    continue
                q, steps, _ = trace(bh, hh)
                if steps and len(steps) >= 2:
                    b_h = hh; break
        if b_h is not None:
            emit_case(L, ds, c, k, "contrast", b_h, len(idx), gmean, gmed, bh, outdir)
            manifest.append((c, "contrast", b_h))
        else:
            L.append("  no contrasting case with a multi-attempt trace found.")
        L.append("")
    ps = versioned(f"case_studies_{ds}", "txt", outdir)
    Path(ps).write_text("\n".join(L) + "\n")
    mp = versioned(f"case_manifest_{ds}", "csv", outdir)
    with open(mp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["cluster", "role", "hash"])
        for c_, role_, h_ in manifest:
            w.writerow([c_, role_, h_])
    print("\n".join(L)); print(f"wrote {ps}\nwrote {mp}")


def main():
    which = [a.lower() for a in sys.argv[1:]] or ["exam", "year"]
    for ds in which:
        run(ds)


if __name__ == "__main__":
    main()
