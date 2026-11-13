"""External benchmark: frozen V1 against the official fallback on identical degraded ATD inputs.

For every (preset, seed): degrade the converted DFL tracking, write ONE ATD file pair, and run
  V1        pitchvision.atd.load_arrays -> pitchvision.pipeline.run (frozen configs/v1.json)
  fallback  starter.baseline.nearest_player_labels -> baseline.smooth(codes, 50)
on that same file. Both predictions are scored against the DFL targets with the official
possession_metrics, plus dead-ball boundary matching.

Priorities, as decided: alive/dead F1, D precision/recall/F1 and dead-ball boundary accuracy are
primary; possession spellF1 is secondary. The targets are created by convert.layout and touch
nothing but score(): degradation and writing never see them, and run() records a hash of every
prediction so tests can show that changing the targets never changes a prediction.

Results on one match are exploratory diagnostics. They are not an estimate of competition
performance and not a basis for choosing parameters.
"""

import hashlib
import os
import sys
import tempfile

import numpy as np

from pitchvision import atd, config, pipeline
from pvext import convert, degrade, dfl_parse
from starter import baseline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "evaluation") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
V1_CONFIG = os.path.join(ROOT, "configs", "v1.json")
FALLBACK_WINDOW = 50
TOLERANCES = (12, 25, 50)          # frames: 0.5 s, 1 s, 2 s
PRIMARY = ("alive_dead_f1", "d_precision", "d_recall", "d_f1")


def _runs(blocks):
    return [(first, last + 1) for _, first, last in blocks]


def _events(codes, runs):
    starts, ends = [], []
    for frame, before, after in pm.transitions(codes, runs):
        if after == pm.D and before in (pm.H, pm.A):
            starts.append((frame, before, after))
        elif before == pm.D and after in (pm.H, pm.A):
            ends.append((frame, before, after))
    return {"dead_start": starts, "dead_end": ends}


def boundary_scores(target, pred, blocks, tolerances=TOLERANCES):
    """Dead-ball boundaries (H/A -> D starts, D -> H/A ends), matched one-to-one within each
    tolerance in frames. Predictions are X-filled the way the official scorer does it."""
    runs = _runs(blocks)
    truth = _events(pm.label_codes(target), runs)
    guess = _events(pm.fill_abstentions(pm.label_codes(pred), runs), runs)
    out = {}
    for kind in ("dead_start", "dead_end"):
        out[kind] = {}
        for tol in tolerances:
            pairs, _, _ = pm.match_transitions(truth[kind], guess[kind], tol, typed="none")
            n_true, n_pred, hits = len(truth[kind]), len(guess[kind]), len(pairs)
            precision = hits / n_pred if n_pred else 0.0
            recall = hits / n_true if n_true else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            offsets = [abs(p[0] - g[0]) / FPS for g, p in pairs]
            out[kind][tol] = {"precision": precision, "recall": recall, "f1": f1,
                              "n_true": n_true, "n_pred": n_pred,
                              "median_abs_offset_s": float(np.median(offsets)) if offsets else None}
    return out


def score(target, pred, blocks):
    """Primary (alive/dead, D) and secondary (possession spells) metrics for one prediction."""
    truth, guess = pm.label_codes(target), pm.label_codes(pred)
    result = pm.evaluate(truth, guess)
    d = result["per_class"]["D"]
    graded = np.zeros(len(target), dtype=bool)
    for first, last in _runs(blocks):
        graded[first:last] = True
    filled = pm.fill_abstentions(guess, _runs(blocks))
    return {
        "alive_dead_f1": result["alive_dead_f1"],
        "d_precision": d["precision"], "d_recall": d["recall"], "d_f1": d["f1"],
        "d_share_true": float(np.mean(truth[graded] == pm.D)),
        "d_share_pred": float(np.mean(filled[graded] == pm.D)),
        "macro_f1": result["macro_f1"],
        "spell_f1_mean": result["spell_f1_mean"],
        "n_true_spells": result["n_gt_spells"], "n_pred_spells": result["n_pred_spells"],
        "n_graded": int(graded.sum()),
        "boundary": boundary_scores(target, pred, blocks),
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(meta_xml, positions_xml, presets, seeds, work_dir, config_path=V1_CONFIG,
        target_transform=None):
    """One row per (method, preset, seed). `target_transform` exists for leakage tests only: it
    alters the targets after the inputs have been built, and must leave predictions unchanged."""
    track, labels, meta = dfl_parse.parse(meta_xml, positions_xml)
    frames, target, blocks = convert.layout(track, labels, meta)
    if target_transform is not None:
        target = target_transform(target)
    cfg = config.load(config_path)
    n = len(target)
    os.makedirs(work_dir, exist_ok=True)

    rows = []
    for preset_name in presets:
        preset = degrade.PRESETS[preset_name]
        for seed in ([0] if preset.get("identity") else seeds):
            degraded = degrade.apply(frames, preset, seed)
            with tempfile.TemporaryDirectory(dir=work_dir) as scratch:
                meta_path, raw_path = convert.write_atd(degraded, meta, scratch)
                input_sha = _sha256_file(raw_path)
                tr = atd.load_arrays(meta_path, raw_path, n, meta.pitch)
                predictions = {"v1": pipeline.run(tr, cfg, FPS, meta.pitch, blocks)}
                codes = baseline.nearest_player_labels(
                    raw_path, baseline.team_of_each_track(meta_path), n, meta.pitch)
                predictions["fallback"] = baseline.smooth(codes, FALLBACK_WINDOW)
            for method, pred in predictions.items():
                row = {"method": method, "preset": preset_name, "seed": seed,
                       "input_sha256": input_sha,
                       "prediction_sha256": hashlib.sha256(
                           "".join(pred.tolist()).encode("ascii")).hexdigest()}
                row.update(score(target, pred, blocks))
                rows.append(row)
    return rows


def flatten(row):
    """A CSV-friendly copy: boundary metrics become boundary_<kind>_tol<frames>_<metric>."""
    flat = {k: v for k, v in row.items() if k != "boundary"}
    for kind, by_tol in row["boundary"].items():
        for tol, metrics in by_tol.items():
            for name, value in metrics.items():
                flat["boundary_{}_tol{}_{}".format(kind, tol, name)] = value
    return flat


def summarize(rows):
    """{preset: {method: {"n_seeds": k, metric: {"mean", "min", "max"}}}} over numeric metrics."""
    out = {}
    for row in rows:
        out.setdefault(row["preset"], {}).setdefault(row["method"], []).append(flatten(row))
    summary = {}
    for preset, methods in out.items():
        summary[preset] = {}
        for method, flats in methods.items():
            entry = {"n_seeds": len(flats)}
            for key, value in flats[0].items():
                if key in ("method", "preset", "seed") or isinstance(value, str):
                    continue
                values = [f[key] for f in flats if f[key] is not None]
                entry[key] = ({"mean": float(np.mean(values)), "min": float(np.min(values)),
                               "max": float(np.max(values))} if values else None)
            summary[preset][method] = entry
    return summary
