"""HA5 evaluation harness: architecture-neutral, match-level, pre-registered (design:
docs/HA5_EXPERIMENT_HARNESS.md).

A candidate is anything that writes, for every (match, preset, seed) evaluation copy of the IDSSE
dev matches, a full-length H/A/D/X label array aligned to the copy's frames, plus a manifest that
names, for every evaluated match, the matches its predictor was trained or tuned on. The harness:

  - refuses any match outside DEV for evaluation, any lockbox or competition id anywhere in a
    manifest, and any fold whose training set contains the evaluated match;
  - scores every copy with the OFFICIAL evaluation code (possession_metrics.evaluate, bootstrap off)
    on the arrays exactly as given (temporal order preserved, nothing sorted or resampled);
  - adds diagnostics: spell-count decomposition, direct H/A switch detection, restart-team accuracy,
    duration-binned spell recall/precision, false-possession creation and missed-possession merging,
    and ball-visibility-stratified recall when ball presence is supplied;
  - aggregates per match first (mean over that match's seeds), reports every match, and only then
    the match-level pooled summary (seeds are never treated as independent games);
  - compares against the frozen fold-out HA3 baseline and applies pre-registered GO/NO-GO criteria,
    which must have been registered (sha256 + time) before the evaluation is run.
No model is fitted here.
"""

import datetime
import hashlib
import json
import os

import numpy as np

from pvext.benchmark import pm

# Params
DEV = ("J03WPY", "J03WOH", "J03WOY", "J03WQQ", "J03WR9")
FORBIDDEN = ("J03WMX", "J03WN1", "BAR-ATH")     # lockbox (already seen) and the competition match
PRESETS = ("mild", "doc", "severe")
SEEDS = (0, 1, 2)
FPS = 25
LABELS = ("H", "A", "D", "X")
DURATION_BINS = (("<1s", 0.0, 1.0), ("1-3s", 1.0, 3.0), ("3-10s", 3.0, 10.0), (">=10s", 10.0, float("inf")))
VISIBILITY_BINS = (("vis<25%", 0.0, 0.25), ("vis25-75%", 0.25, 0.75), ("vis>=75%", 0.75, 1.0000001))
SWITCH_TOL_S = 1.0
HEADLINE_IOU = pm.SPELL_IOU_HEADLINE


class HarnessError(ValueError):
    pass


# -- integrity -------------------------------------------------------------------------------------

def check_manifest(manifest, matches):
    """Split integrity for the matches being evaluated. manifest = {"name", "folds": {match: [ids]}}."""
    folds = manifest.get("folds")
    if not isinstance(folds, dict):
        raise HarnessError("manifest has no folds")
    for m in matches:
        if m not in DEV:
            raise HarnessError("{} is not an unrestricted dev match".format(m))
        if m not in folds:
            raise HarnessError("no fold declared for {}".format(m))
        train = list(folds[m])
        if m in train:
            raise HarnessError("fold for {} was trained on {}".format(m, m))
        bad = [t for t in train if t in FORBIDDEN or t not in DEV]
        if bad:
            raise HarnessError("fold for {} uses forbidden or unknown matches {}".format(m, bad))
    for m, train in folds.items():
        if m in FORBIDDEN or any(t in FORBIDDEN for t in train):
            raise HarnessError("manifest mentions a forbidden match")


def check_labels(labels, target):
    labels = np.asarray(labels)
    if labels.shape != np.asarray(target).shape:
        raise HarnessError("prediction length {} != target length {}".format(labels.shape, np.shape(target)))
    bad = set(np.unique(labels).tolist()) - set(LABELS)
    if bad:
        raise HarnessError("labels outside H/A/D/X: {}".format(sorted(bad)))
    return labels.astype("<U1")


# -- per-copy metrics ------------------------------------------------------------------------------

def _runs_of(mask, runs):
    out = []
    for s, e in runs:
        edges = np.diff(np.concatenate(([0], np.asarray(mask[s:e], np.int8), [0])))
        out += [(s + int(a), s + int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
    return out


def _switches(codes, runs):
    """[(frame, new_code)] direct H<->A switches inside runs (frame = first frame of the new team)."""
    out = []
    for s, e in runs:
        c = codes[s:e]
        idx = np.flatnonzero(((c[1:] == pm.H) & (c[:-1] == pm.A)) | ((c[1:] == pm.A) & (c[:-1] == pm.H)))
        out += [(s + int(i) + 1, int(c[i + 1])) for i in idx]
    return out


def match_events(true_ev, pred_ev, tol):
    """One-to-one greedy matching of (frame, code) events with the same code within tol frames,
    closest pairs first (ties by true then predicted order). Returns (tp, fp, fn)."""
    pairs = sorted((abs(t - p), i, j) for i, (t, tc) in enumerate(true_ev)
                   for j, (p, pc) in enumerate(pred_ev) if tc == pc and abs(t - p) <= tol)
    used_t, used_p = set(), set()
    for _, i, j in pairs:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
    tp = len(used_t)
    return tp, len(pred_ev) - tp, len(true_ev) - tp


def _bin(n_frames, bins=DURATION_BINS):
    s = n_frames / FPS
    return next(name for name, lo, hi in bins if lo <= s < hi)


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def copy_metrics(target, labels, ball_present=None):
    """All metrics for one evaluation copy. `target`/`labels` are full-length H/A/D/X arrays."""
    labels = check_labels(labels, target)
    gt = pm.label_codes(target)
    pred = pm.label_codes(labels)
    official = pm.evaluate(gt, pred, bootstrap_n=0)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    out = {"spell_f1": official["spell_f1_mean"], "macro_f1": official["macro_f1"],
           "n_true_spells": official["n_gt_spells"], "n_pred_spells": official["n_pred_spells"]}
    out["spell_count_ratio"] = out["n_pred_spells"] / out["n_true_spells"] if out["n_true_spells"] else None

    # spell count = live intervals + direct switches
    for name, codes in (("true", gt), ("pred", filled)):
        live = (codes == pm.H) | (codes == pm.A)
        out["{}_live_intervals".format(name)] = len(_runs_of(live, runs))
        out["{}_direct_switches".format(name)] = len(_switches(codes, runs))

    # direct switch detection
    out["switch"] = _prf(*match_events(_switches(gt, runs), _switches(filled, runs),
                                       int(round(SWITCH_TOL_S * FPS))))

    # restart team: the team at the first frame of each true live interval, if predicted live there
    true_live = _runs_of((gt == pm.H) | (gt == pm.A), runs)
    starts = [(a, gt[a]) for a, _ in true_live]
    right = sum(1 for a, c in starts if filled[a] == c)
    dead = sum(1 for a, _ in starts if filled[a] == pm.D)
    out["restart"] = {"n": len(starts), "team_correct": right, "pred_dead": dead,
                      "wrong_team": len(starts) - right - dead}

    # official spells, matching at every IoU and at the headline IoU
    gs = pm.possession_spells(gt, runs)
    ps = pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(gs, ps)
    flags = {t: pm.matched_gt_flags(ov, len(gs), t) for t in pm.SPELL_IOUS}
    gmatch, pmatch = _headline_pairs(ov)

    true_bins = {b[0]: {"n": 0, "matched_headline": 0, "recall_mean_iou_sum": 0.0} for b in DURATION_BINS}
    for i, (a, b, _) in enumerate(gs):
        d = true_bins[_bin(b - a)]
        d["n"] += 1
        d["matched_headline"] += int(i in gmatch)
        d["recall_mean_iou_sum"] += float(np.mean([flags[t][i] for t in pm.SPELL_IOUS]))
    pred_bins = {b[0]: {"n": 0, "matched_headline": 0} for b in DURATION_BINS}
    for j, (a, b, _) in enumerate(ps):
        d = pred_bins[_bin(b - a)]
        d["n"] += 1
        d["matched_headline"] += int(j in pmatch)
    out["true_by_duration"] = true_bins
    out["pred_by_duration"] = pred_bins

    out["false_possessions"] = _false_possessions(gs, ps, gmatch, pmatch, filled)
    out["missed_possessions"] = _missed_possessions(gs, ps, gmatch, filled)

    if ball_present is not None:
        bp = np.asarray(ball_present, bool)
        if bp.shape != gt.shape:
            raise HarnessError("ball_present length differs from the target")
        vis = {b[0]: {"n": 0, "matched_headline": 0} for b in VISIBILITY_BINS}
        for i, (a, b, _) in enumerate(gs):
            share = float(bp[a:b].mean())
            name = next(n for n, lo, hi in VISIBILITY_BINS if lo <= share < hi)
            vis[name]["n"] += 1
            vis[name]["matched_headline"] += int(i in gmatch)
        out["true_by_visibility"] = vis
    return out


def _headline_pairs(ov):
    """{gt index: pred index} and its inverse for the official greedy matching at the headline IoU."""
    flags_g, used_p, pairs = set(), set(), {}
    for iou, i, j in ov:
        if iou < HEADLINE_IOU:
            break
        if i in flags_g or j in used_p:
            continue
        flags_g.add(i)
        used_p.add(j)
        pairs[i] = j
    return pairs, {j: i for i, j in pairs.items()}


def _overlap(a, b, c, d):
    return max(0, min(b, d) - max(a, c))


def _false_possessions(gs, ps, gmatch, pmatch, filled):
    """Unmatched predicted spells (headline IoU): `fragment` when their largest same-team true
    overlap is matched to another predicted spell (a split), `spurious` otherwise."""
    out = {"n": 0, "fragment": 0, "spurious": 0}
    for j, (a, b, c) in enumerate(ps):
        if j in pmatch:
            continue
        out["n"] += 1
        best, best_ov = None, 0
        for i, (x, y, k) in enumerate(gs):
            o = _overlap(a, b, x, y) if k == c else 0
            if o > best_ov:
                best, best_ov = i, o
        out["fragment" if best is not None and best in gmatch else "spurious"] += 1
    return out


def _missed_possessions(gs, ps, gmatch, filled):
    """Unmatched true spells (headline IoU) by cause, mutually exclusive in this order:
    `dead` (>= 50% predicted D), `merged` (the same-team predicted spell overlapping it most also
    covers >= 50% of another true spell), `wrong_team` (live majority the other team), `boundary`."""
    out = {"n": 0, "dead": 0, "merged": 0, "wrong_team": 0, "boundary": 0}
    for i, (a, b, c) in enumerate(gs):
        if i in gmatch:
            continue
        out["n"] += 1
        seg = filled[a:b]
        if np.mean(seg == pm.D) >= 0.5:
            out["dead"] += 1
            continue
        best, best_ov = None, 0
        for j, (x, y, k) in enumerate(ps):
            o = _overlap(a, b, x, y) if k == c else 0
            if o > best_ov:
                best, best_ov = j, o
        merged = False
        if best is not None:
            x, y, _ = ps[best]
            merged = any(_overlap(x, y, u, v) >= 0.5 * (v - u)
                         for k2, (u, v, c2) in enumerate(gs) if k2 != i and c2 == c)
        live = seg[(seg == pm.H) | (seg == pm.A)]
        if merged:
            out["merged"] += 1
        elif len(live) and np.mean(live == c) < 0.5:
            out["wrong_team"] += 1
        else:
            out["boundary"] += 1
    return out


# -- aggregation -----------------------------------------------------------------------------------

def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = prefix + k
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        elif v is not None:
            out[key] = float(v)
    return out


def per_match(copy_results):
    """{(match, preset): {metric: mean over that match's seeds}}; refuses incomplete seed sets."""
    groups = {}
    for (m, p, s), r in copy_results.items():
        groups.setdefault((m, p), {})[s] = _flat(r)
    out = {}
    for key, by_seed in groups.items():
        if sorted(by_seed) != sorted(SEEDS):
            raise HarnessError("{} has seeds {}, expected {}".format(key, sorted(by_seed), list(SEEDS)))
        metrics = set.intersection(*(set(v) for v in by_seed.values()))
        out[key] = {k: float(np.mean([by_seed[s][k] for s in SEEDS])) for k in sorted(metrics)}
    return out


def pooled(match_table, preset, metric, matches=DEV):
    vals = [match_table[(m, preset)][metric] for m in matches]
    return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)),
            "per_match": dict(zip(matches, vals))}


def compare(cand, base, preset, metric, matches=DEV):
    """Per-match deltas first, then the match-level summary."""
    deltas = {m: cand[(m, preset)][metric] - base[(m, preset)][metric] for m in matches}
    return {"per_match": deltas, "mean": float(np.mean(list(deltas.values()))),
            "n_improved": sum(v > 0 for v in deltas.values()), "n_matches": len(matches)}


# -- pre-registration ------------------------------------------------------------------------------

def _sha(path):
    with open(path, "rb") as h:
        return hashlib.sha256(h.read()).hexdigest()


def register(criteria_path, registry):
    """Append the criteria file's sha256 and the time to the registry (append-only JSON lines)."""
    with open(criteria_path) as handle:
        crit = json.load(handle)
    validate_criteria(crit)
    os.makedirs(registry, exist_ok=True)
    entry = {"sha256": _sha(criteria_path), "name": crit["name"],
             "registered": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with open(os.path.join(registry, "REGISTRY.jsonl"), "a") as h:
        h.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def registered(criteria_path, registry):
    sha = _sha(criteria_path)
    path = os.path.join(registry, "REGISTRY.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        for line in handle:
            e = json.loads(line)
            if e["sha256"] == sha:
                return e
    return None


def validate_criteria(crit):
    """{"name", "baseline": "ha3", "rules": [{"metric", "preset", "min_mean_gain",
    "min_matches_improved"?}]}"""
    if not crit.get("name") or not isinstance(crit.get("rules"), list) or not crit["rules"]:
        raise HarnessError("criteria need a name and a non-empty rules list")
    for r in crit["rules"]:
        if r.get("preset") not in PRESETS or "metric" not in r or "min_mean_gain" not in r:
            raise HarnessError("bad rule {}".format(r))


def decide(crit, cand, base, matches=DEV):
    results, ok = [], True
    for r in crit["rules"]:
        c = compare(cand, base, r["preset"], r["metric"], matches)
        passed = c["mean"] >= r["min_mean_gain"] and c["n_improved"] >= r.get("min_matches_improved", 0)
        ok &= passed
        results.append(dict(r, observed=c, passed=passed))
    return {"verdict": "GO" if ok else "NO-GO", "rules": results}
