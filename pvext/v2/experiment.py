"""The M1 experiment: training copies, fold models, held-out evaluation, error analysis, go/no-go.

  1. build   per match: degraded TRAINING copies (presets mild/doc/severe x seeds 100, 101) ->
             write_atd -> atd.load_arrays -> features.compute -> rows every STRIDE frames, cached
             outside the repository. Targets come from convert.layout and are stored next to the rows.
  2. train   per fold: the threshold is chosen by inner leave-one-match-out over the training matches,
             then the model is fit on all training rows. The held-out matches are never read.
  3. evaluate per held-out match: the FROZEN evaluation copies (mild/doc/severe x seeds 0, 1, 2,
             the same as the benchmark) are rebuilt, and frozen V1, M1 (LOMO model) and M1
             (league model) are scored with pvext.benchmark.score on identical Tracking. Missed
             stoppage starts are analysed on doc.
  4. report  per-match tables, paired V1 comparisons and the pre-declared go criteria.
"""

import hashlib
import json
import os
import tempfile

import joblib
import numpy as np

from pitchvision import atd, config, pipeline
from pitchvision import features as v1_features
from pvext import benchmark, convert, degrade, dfl_parse, idsse
from pvext.v2 import features, m1

# Params
FPS = 25
START_TOL = 25            # frames: a start within 1 s counts as matched
OUT_WINDOW = (50, 125)    # frames before/after a true start in which an out-of-play ball marks it
SHORT_STOPPAGE_S = 5.0


# -- pure helpers ----------------------------------------------------------------------------------

def stride_rows(X, target, blocks, stride=m1.STRIDE):
    """Rows every `stride` frames inside each block, with the blocks re-indexed in row space."""
    frames, row_blocks, start = [], [], 0
    for number, first, last in blocks:
        index = np.arange(first, last + 1, stride)
        frames.append(index)
        row_blocks.append((number, start, start + len(index) - 1))
        start += len(index)
    frames = np.concatenate(frames)
    return {"frames": frames, "X": X[frames], "y": target[frames] == "D", "blocks": row_blocks}


def _dead_runs(codes, first, stop):
    dead = codes[first:stop] == "D"
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    return [(first + a, first + b) for a, b in
            zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def start_errors(truth, pred, blocks, ball_out, ball_present):
    """One entry per true stoppage start (H/A -> D inside a block): category matched / late / early /
    missed, the delay of a late detection, the stoppage duration, and simple tracking context."""
    out = []
    for _, first, last in blocks:
        stop = last + 1
        true_runs = [(a, b) for a, b in _dead_runs(truth, first, stop)
                     if a > first and truth[a - 1] in ("H", "A")]
        pred_runs = _dead_runs(pred, first, stop)
        for a, b in true_runs:
            overlapping = [(p, q) for p, q in pred_runs if p < b and q > a]
            entry = {"duration_s": (b - a) / FPS, "delay_s": None,
                     "out_of_play": bool(np.asarray(ball_out[max(first, a - OUT_WINDOW[0]):
                                                             min(stop, a + OUT_WINDOW[1])]).any()),
                     "ball_present": bool(ball_present[a])}
            if not overlapping:
                entry["category"] = "missed"
            else:
                p = min(p for p, _ in overlapping)
                if abs(p - a) <= START_TOL:
                    entry["category"] = "matched"
                elif p > a:
                    entry["category"] = "late"
                    entry["delay_s"] = (p - a) / FPS
                else:
                    entry["category"] = "early"
            out.append(entry)
    return out


def summarize_starts(entries):
    cats = ("matched", "late", "early", "missed")

    def block(items):
        n = len(items)
        return dict({"n": n}, **{c: sum(e["category"] == c for e in items) for c in cats})
    n = len(entries)
    delays = [e["delay_s"] for e in entries if e["category"] == "late"]
    return {
        "n": n,
        "share": {c: (sum(e["category"] == c for e in entries) / n if n else None) for c in cats},
        "late_delay_s_median": float(np.median(delays)) if delays else None,
        "by_context": {"out_of_play": block([e for e in entries if e["out_of_play"]]),
                       "in_play": block([e for e in entries if not e["out_of_play"]])},
        "short_stoppages": block([e for e in entries if e["duration_s"] < SHORT_STOPPAGE_S]),
        "ball_missing_at_start": block([e for e in entries if not e["ball_present"]]),
    }


def check_held_out(bundle, match, name="model"):
    """Raise ValueError unless `match` is in the bundle's test set, absent from its training set,
    and the two sets are disjoint. Runs in every evaluation (not an assert, so never optimised away)."""
    train, test = set(bundle.get("train", ())), set(bundle.get("test", ()))
    if not train or not test:
        raise ValueError("{}: bundle has no train/test membership".format(name))
    if train & test:
        raise ValueError("{}: train and test overlap on {}".format(name, sorted(train & test)))
    if match not in test:
        raise ValueError("{}: {} is not in the model's test set {}".format(name, match, sorted(test)))
    if match in train:
        raise ValueError("{}: {} was used for training".format(name, match))


# -- data building ---------------------------------------------------------------------------------

def _layout(match, root):
    paths = idsse.local_paths(match, root)
    track, labels, meta = dfl_parse.parse(paths["meta"], paths["positions"])
    frames, target, blocks = convert.layout(track, labels, meta)
    return frames, target, blocks, meta


def _tracking(frames, meta, preset, seed, scratch, n):
    degraded = degrade.apply(frames, degrade.PRESETS[preset], seed)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        meta_path, raw_path = convert.write_atd(degraded, meta, tmp)
        input_sha = benchmark._sha256_file(raw_path)
        return atd.load_arrays(meta_path, raw_path, n, meta.pitch), input_sha


def build_training(match, root, work):
    """Cache stride rows of every training copy of one match; returns the cache paths."""
    frames, target, blocks, meta = _layout(match, root)
    os.makedirs(os.path.join(work, "train"), exist_ok=True)
    paths = []
    for preset in m1.TRAIN_PRESETS:
        for seed in m1.TRAIN_SEEDS:
            tr, _ = _tracking(frames, meta, preset, seed, work, len(target))
            names, X = features.compute(tr, meta.pitch, blocks, FPS)
            rows = stride_rows(X, target, blocks)
            path = os.path.join(work, "train", "{}_{}_{}.npz".format(match, preset, seed))
            np.savez(path, X=rows["X"], y=rows["y"], frames=rows["frames"],
                     blocks=json.dumps(rows["blocks"]), names=json.dumps(names))
            paths.append(path)
    return paths


def _series(work, match):
    out = []
    for preset in m1.TRAIN_PRESETS:
        for seed in m1.TRAIN_SEEDS:
            data = np.load(os.path.join(work, "train", "{}_{}_{}.npz".format(match, preset, seed)))
            out.append({"X": data["X"], "y": data["y"],
                        "blocks": [tuple(b) for b in json.loads(str(data["blocks"]))]})
    return out


def train_fold(kind, train_ids, test_ids, work):
    """Select the threshold on the training matches only, fit, save; returns the model path."""
    cfg = config.load(m1.V1_CONFIG)
    train = {m: _series(work, m) for m in train_ids}
    threshold, curve = m1.select_threshold(train, cfg["alive"], fps=FPS / m1.STRIDE)
    X = np.concatenate([s["X"] for items in train.values() for s in items])
    y = np.concatenate([s["y"] for items in train.values() for s in items])
    model = m1.fit(X, y)
    os.makedirs(os.path.join(work, "models"), exist_ok=True)
    path = os.path.join(work, "models", "{}_{}.joblib".format(kind, "-".join(test_ids)))
    joblib.dump({"model": model, "threshold": threshold, "curve": curve, "kind": kind,
                 "train": list(train_ids), "test": list(test_ids), "n_rows": int(len(y))}, path)
    return path


def evaluate_match(match, model_paths, root, work):
    """Score frozen V1 and the M1 fold models for one held-out match on the frozen evaluation
    copies. `model_paths` maps "m1_lomo"/"m1_league" to the fold model whose test set holds
    `match`."""
    cfg = config.load(m1.V1_CONFIG)
    models = {name: joblib.load(path) for name, path in model_paths.items()}
    for name, bundle in models.items():
        check_held_out(bundle, match, name)
    frames, target, blocks, meta = _layout(match, root)
    rows, starts = [], {}
    for preset in m1.EVAL_PRESETS:
        for seed in m1.EVAL_SEEDS:
            tr, input_sha = _tracking(frames, meta, preset, seed, work, len(target))
            names, X = features.compute(tr, meta.pitch, blocks, FPS)
            v1_feat = v1_features.compute(tr, cfg["features"], FPS, meta.pitch)
            predictions = {"v1": pipeline.run(tr, cfg, FPS, meta.pitch, blocks)}
            for name, bundle in models.items():
                prob = bundle["model"].predict_proba(X)[:, 1]
                dead = m1.persist(prob, blocks, cfg["alive"], FPS, bundle["threshold"])
                predictions[name] = m1.run_with_dead(tr, cfg, FPS, meta.pitch, blocks, dead)
            for method, pred in predictions.items():
                row = {"match": match, "preset": preset, "seed": seed, "method": method,
                       "input_sha256": input_sha,
                       "prediction_sha256": hashlib.sha256(
                           "".join(pred.tolist()).encode("ascii")).hexdigest()}
                row.update(benchmark.score(target, pred, blocks))
                rows.append(benchmark.flatten(row))
                if preset == "doc":
                    starts.setdefault(method, []).extend(
                        start_errors(target, pred, blocks, v1_feat.ball_out, tr.ball_present))
    return rows, {m: summarize_starts(e) for m, e in starts.items()}


# -- decisions -------------------------------------------------------------------------------------

def per_match(rows, method, preset, key):
    out = {}
    for r in rows:
        if r["method"] == method and r["preset"] == preset:
            out.setdefault(r["match"], []).append(r[key])
    return {m: float(np.mean(v)) for m, v in out.items()}


GO_METHODS = ("v1", "m1_lomo", "m1_league")
GO_METRICS = ("d_f1", "boundary_dead_start_tol25_f1", "d_precision", "spell_f1_mean")


def validate_grid(rows, matches=m1.MATCHES, methods=GO_METHODS, presets=m1.EVAL_PRESETS,
                  seeds=m1.EVAL_SEEDS, metrics=GO_METRICS):
    """Raise ValueError unless `rows` is exactly the evaluation grid: every match x method x preset x
    seed once, nothing else, and every required metric finite."""
    expected = {(m, k, p, s) for m in matches for k in methods for p in presets for s in seeds}
    seen = {}
    for r in rows:
        seed = float(r["seed"])
        if seed != int(seed):
            raise ValueError("non-integer seed {!r}".format(r["seed"]))
        key = (r["match"], r["method"], r["preset"], int(seed))
        if key in seen:
            raise ValueError("duplicate evaluation row {}".format(key))
        seen[key] = r
        for metric in metrics:
            value = r.get(metric)
            if value is None or not np.isfinite(float(value)):
                raise ValueError("{} is not finite in {}: {!r}".format(metric, key, value))
    missing, extra = expected - set(seen), set(seen) - expected
    if missing or extra:
        raise ValueError("evaluation grid mismatch: {} missing {}, {} unexpected {}".format(
            len(missing), sorted(missing)[:3], len(extra), sorted(extra)[:3]))
    return len(seen)


def go_criteria(rows):
    """The six pre-declared criteria (design spec), evaluated mechanically on a validated grid."""
    validate_grid(rows)
    def better(method, preset, key):
        a, b = per_match(rows, method, preset, key), per_match(rows, "v1", preset, key)
        return {m: a[m] - b[m] for m in a}
    d_f1 = better("m1_lomo", "doc", "d_f1")
    start = better("m1_lomo", "doc", "boundary_dead_start_tol25_f1")
    precision = better("m1_lomo", "doc", "d_precision")
    spell = better("m1_lomo", "doc", "spell_f1_mean")
    league = {}
    for group, ids in (("bundesliga", m1.BUNDESLIGA), ("second_division", m1.SECOND_DIVISION)):
        cand = per_match(rows, "m1_league", "doc", "d_f1")
        base = per_match(rows, "v1", "doc", "d_f1")
        league[group] = float(np.mean([cand[m] for m in ids]) - np.mean([base[m] for m in ids]))
    reversal = {p: sum(v > 0 for v in better("m1_lomo", p, "d_f1").values())
                for p in ("mild", "severe")}
    c = {
        "1_d_f1_higher_6of7_incl_both_bundesliga": {
            "value": sum(v > 0 for v in d_f1.values()),
            "pass": sum(v > 0 for v in d_f1.values()) >= 6 and all(d_f1[m] > 0 for m in m1.BUNDESLIGA)},
        "2_start_f1_higher_5of7": {"value": sum(v > 0 for v in start.values()),
                                   "pass": sum(v > 0 for v in start.values()) >= 5},
        "3_median_precision_drop_le_0.05": {"value": float(np.median(list(precision.values()))),
                                            "pass": float(np.median(list(precision.values()))) >= -0.05},
        "4_spell_drop_gt_0.02_in_le_2": {"value": sum(v < -0.02 for v in spell.values()),
                                         "pass": sum(v < -0.02 for v in spell.values()) <= 2},
        "5_league_out_both_directions": {"value": league, "pass": all(v > 0 for v in league.values())},
        "6_mild_severe_d_f1_higher_4of7": {"value": reversal,
                                           "pass": all(v >= 4 for v in reversal.values())},
    }
    passed = [k for k, v in c.items() if v["pass"]]
    if len(passed) == len(c):
        verdict = "GO"
    elif c["1_d_f1_higher_6of7_incl_both_bundesliga"]["pass"] and c["3_median_precision_drop_le_0.05"]["pass"]:
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"
    return {"criteria": c, "verdict": verdict}
