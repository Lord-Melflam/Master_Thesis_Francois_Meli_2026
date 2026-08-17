"""Shared helpers for v2 data audit / processing. KISS.

- versioned_path(): never overwrite; append _v1/_v2/... by checking existence.
- parse_ts(): parse the extractor timestamp format.
- parse_fname(): pull (student, submission, grade, hash) out of a submission filename.
- load_rows(): read a list of data.csv files into a list of dicts.

Submission CSV schema: hash,year,qname,n_submission,score,status,timestamp,file
Filename convention:      q{N}_{student}_{submission}_{grade}_{hash}.py
"""
from pathlib import Path
from datetime import datetime
import csv
import re
import json
import glob

TS_FMT = "%Y-%m-%d-%H:%M:%S"

# --- single source of truth for the clustering config + recipe (avoids drift across scripts) ---
_REPO = Path(__file__).resolve().parents[3]
_CLU = _REPO / "data/v2/res_python/clustering"
_FEAT = _REPO / "data/v2/res_python/features"
_LINK = _REPO / "data/v2/res_python/linkage"

# k per dataset — a DECISION (from the k-selection review, script 28). Change here ONLY.
# Official re-grade (2026-08): exam best-supported at k=4 (rank-sum 4; top stability+DB), year at k=3.
KMAP = {"exam": 4, "year": 3}

# features that are log1p-transformed before z-scoring (right-skewed counts/times)
LOG_FEATS = {"median_attempts", "median_mean_delta_sec", "median_edit_size", "median_nloc"}
FEATS_FILE = {"exam": "exam_features_all", "year": "missions_features_all"}


def _latest(pat):
    return sorted(glob.glob(str(pat)))[-1]


def load_kept(ds):
    """The de-duplicated clustering feature set for a dataset, WRITTEN by script 22
    (kept_features.json) and READ by every downstream script. No hardcoded lists."""
    return json.load(open(_CLU / "kept_features.json"))[ds]


def grades_map():
    """hash -> exam_grade (mean final true score over the 6 exam questions, 0-100)."""
    return {r["hash"]: float(r["exam_grade"])
            for r in csv.DictReader(open(_latest(_CLU / "exam_score_categories_v*.csv")))}


def cluster(ds):
    """THE canonical clustering — one recipe, used by every script, fully deterministic
    (Ward has no random component). Returns (hashes, Xs, labels, grade, cols).
    Recipe: 569 linked cohort -> kept features (load_kept) -> log1p on skewed -> z-score
    -> Ward at KMAP[ds] -> labels renumbered 1..k by ascending mean exam grade."""
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.preprocessing import StandardScaler
    cols = load_kept(ds)
    feats = {r["hash"]: r for r in csv.DictReader(open(_latest(_FEAT / f"{FEATS_FILE[ds]}_v*.csv")))}
    linked = {r["hash"] for r in csv.DictReader(open(_latest(_LINK / "linked_students_v*.csv")))}
    grades = grades_map()
    hashes = [h for h in feats if h in linked and h in grades]

    def val(h, c):
        try: return float(feats[h][c])
        except Exception: return float("nan")
    X = np.nan_to_num(np.array([[val(h, c) for c in cols] for h in hashes], float), nan=0.0)
    for j, c in enumerate(cols):
        if c in LOG_FEATS:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    Xs = StandardScaler().fit_transform(X)
    lab = fcluster(linkage(Xs, "ward"), KMAP[ds], "maxclust")
    g = np.array([grades[h] for h in hashes])
    order = {c: i + 1 for i, c in enumerate(sorted(set(lab), key=lambda c: g[lab == c].mean()))}
    lab = np.array([order[c] for c in lab])
    return hashes, Xs, lab, g, cols
# leading 'qN_' prefix ignored (it is not the question id; the qname column is)
FNAME_RE = re.compile(r"^q\d+_(\d+)_(\d+)_([\d.]+)_([0-9a-f]{8,})\.py$")


def versioned_path(directory, stem, ext):
    """Return <directory>/<stem>_vN.<ext> with the next free N. Never overwrites."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    n = 1
    while (d / f"{stem}_v{n}.{ext}").exists():
        n += 1
    return d / f"{stem}_v{n}.{ext}"


def parse_ts(s):
    try:
        return datetime.strptime(s.strip(), TS_FMT)
    except Exception:
        return None


def parse_fname(fname):
    """Return {student, submission, grade, hash} from a submission filename, else None
    (returns None for non-submission files such as Test*.py)."""
    m = FNAME_RE.match(fname.strip())
    if not m:
        return None
    return {
        "student": m.group(1),
        "submission": int(m.group(2)),
        "grade": float(m.group(3)),
        "hash": m.group(4),
    }


def load_rows(csv_paths):
    """Read all rows (as dicts) from the given data.csv paths."""
    rows = []
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows
