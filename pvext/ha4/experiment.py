"""HA4 experiment on IDSSE dev (design: docs/superpowers/specs/2026-11-13-pitchvision-ha4-design.md).

  cache    per dev match and frozen evaluation copy: held-out p_dead (frozen M1 lomo model),
           held-out p_home (frozen HA3 dev fold model), the target, the blocks and the exact DFL
           set-piece tags. The HA3 dev prediction hashes must be reproduced first.
  grid     every pre-registered setting on every cached copy, with simulated tags.
  select   nested leave-one-match-out over dev, then the decision rule.
Only dev matches are allowed; the lockbox is never read.
"""

import csv
import hashlib
import itertools
import json
import os

import joblib
import numpy as np

from pitchvision import config
from pvext import benchmark, convert, dfl_parse, idsse
from pvext.benchmark import pm
from pvext.ha import decode as ha_decode
from pvext.ha import features as ha_features
from pvext.ha4 import events, post
from pvext.v2 import experiment as v2x
from pvext.v2 import features as m1_features
from pvext.v2 import m1

# Params
DEV = m1.SECOND_DIVISION
FPS = 25
PRESETS = m1.EVAL_PRESETS
SEEDS = m1.EVAL_SEEDS
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
MIN_DEAD_S = (1.0, 2.0, 3.0)
LAMBDAS = (0.25, 0.5, 1.0)
SNAP_S = (0.0, 3.0, 6.0)
INSERT_S = (0.0, 2.0)
BETAS = (0.0, 1.0, 3.0)
HA3_MIN_DEAD_S = 2.0
TAG_SETTINGS = ("none", "clean", "doc", "pessimistic")
SELECT_TAGS = "doc"
KEYS = ("threshold", "min_dead_s", "lam", "snap_s", "insert_s", "beta")


def _guard(match):
    if match not in DEV:
        raise ValueError("{} is not a dev match; HA4 never reads the lockbox".format(match))


def _sha(pred):
    return hashlib.sha256("".join(pred.tolist()).encode("ascii")).hexdigest()


def spell(gt_codes, runs, pred):
    """Official spellF1 mean (abstention-filled, inside the ground truth's valid runs)."""
    filled = pm.fill_abstentions(pm.label_codes(pred), runs)
    s = pm.spell_scores(gt_codes, filled, runs)
    return s["mean_f1"], s["n_gt"], s["n_pred"]


def settings(family):
    """Pre-registered settings of one family, as dicts of KEYS."""
    out = []
    for thr, md, lam in itertools.product(THRESHOLDS, MIN_DEAD_S, LAMBDAS):
        if family == "a4":
            out.append({"threshold": thr, "min_dead_s": md, "lam": lam, "snap_s": 0.0,
                        "insert_s": 0.0, "beta": 0.0})
        elif family == "full":
            for snap_s, ins, beta in itertools.product(SNAP_S, INSERT_S, BETAS):
                if snap_s == 0 and ins > 0:
                    continue
                out.append({"threshold": thr, "min_dead_s": md, "lam": lam, "snap_s": snap_s,
                            "insert_s": ins, "beta": beta})
        else:
            raise ValueError(family)
    return out


def setting_id(s):
    return "t{threshold:.2f}_md{min_dead_s:g}_l{lam:g}_s{snap_s:g}_i{insert_s:g}_b{beta:g}".format(**s)


# -- cache -----------------------------------------------------------------------------------------

def frozen_ha3(ha3_run):
    out = {}
    with open(os.path.join(ha3_run, "dev_per_run.csv")) as handle:
        for r in csv.DictReader(handle):
            if r["method"] == "m1_ha1":
                out[(r["match"], r["preset"], int(float(r["seed"])))] = (
                    r["input_sha256"], r["prediction_sha256"], float(r["spell_f1_mean"]))
    return out


def build_cache(match, root, work, m1_models, ha3_run):
    """Cache every evaluation copy of one dev match; checks HA3 is reproduced exactly."""
    _guard(match)
    paths = idsse.local_paths(match, root)
    track, labels, meta = dfl_parse.parse(paths["meta"], paths["positions"])
    frames, target, blocks = convert.layout(track, labels, meta)
    tags = events.to_frames(events.parse_setpieces(paths["events"], meta),
                            events.frame_axis(track, blocks))
    mm = joblib.load(os.path.join(m1_models, "lomo_{}.joblib".format(match)))
    ha = joblib.load(os.path.join(ha3_run, "models", "ha1_lomo_{}.joblib".format(match)))
    v2x.check_held_out(mm, match, "m1")
    v2x.check_held_out(ha, match, "ha3")
    cfg = config.load(m1.V1_CONFIG)
    if cfg["alive"]["min_dead_s"] != HA3_MIN_DEAD_S:
        raise ValueError("V1 min_dead_s changed")
    frozen = frozen_ha3(ha3_run)
    gt = pm.label_codes(target)
    runs = pm.valid_runs(gt)
    os.makedirs(os.path.join(work, "cache"), exist_ok=True)
    out = []
    for preset in PRESETS:
        for seed in SEEDS:
            tr, input_sha = v2x._tracking(frames, meta, preset, seed, work, len(target))
            _, X = m1_features.compute(tr, meta.pitch, blocks, FPS)
            p_dead = mm["model"].predict_proba(X)[:, 1]
            names, Xh = ha_features.compute(tr, meta.pitch, blocks, FPS,
                                            feature_set=ha["feature_set"])
            if names != ha["features"]:
                raise ValueError("HA3 feature names differ")
            p_home = ha["model"].predict_proba(Xh)[:, 1]
            dead = m1.persist(p_dead, blocks, cfg["alive"], FPS, mm["threshold"])
            if not np.array_equal(dead, post.dead_mask(p_dead, blocks, mm["threshold"],
                                                       HA3_MIN_DEAD_S)):
                raise ValueError("HA4 persistence does not reproduce M1")
            pred = ha_decode.decode(p_home, dead, blocks, ha["lam"], FPS)
            want = frozen[(match, preset, seed)]
            if (input_sha, _sha(pred)) != want[:2]:
                raise ValueError("HA3 dev prediction not reproduced: {} {} {}".format(
                    match, preset, seed))
            f1, _, _ = spell(gt, runs, pred)
            if abs(f1 - want[2]) > 1e-12:
                raise ValueError("fast spellF1 differs from the official score")
            path = os.path.join(work, "cache", "{}_{}_{}.npz".format(match, preset, seed))
            np.savez(path, p_dead=p_dead, p_home=p_home, target=target,
                     blocks=json.dumps(blocks), tag_frame=tags.frame, tag_team=tags.team,
                     tag_kind=np.asarray(tags.kind, dtype="<U16"),
                     ha3_threshold=mm["threshold"], ha3_lam=ha["lam"], input_sha=input_sha)
            out.append(path)
    return out


def load_copy(work, match, preset, seed):
    _guard(match)
    d = np.load(os.path.join(work, "cache", "{}_{}_{}.npz".format(match, preset, seed)))
    return {"p_dead": d["p_dead"], "p_home": d["p_home"], "target": d["target"],
            "blocks": [tuple(b) for b in json.loads(str(d["blocks"]))],
            "tags": events.Tags(d["tag_frame"], d["tag_team"], d["tag_kind"].astype(object)),
            "ha3_threshold": float(d["ha3_threshold"]), "ha3_lam": float(d["ha3_lam"])}


def copy_tags(c, match, preset, seed, setting):
    """The tags one copy sees under a tag setting (seeded per match, preset and seed)."""
    if setting == "none":
        return events.Tags(np.zeros(0, np.int64), np.zeros(0, np.int8), np.zeros(0, object))
    sim_seed = (events.SIM_SEED_BASE + 1000 * DEV.index(match) + 100 * PRESETS.index(preset)
                + int(seed))
    return events.simulate(c["tags"], c["blocks"], setting, sim_seed)


# -- grid ------------------------------------------------------------------------------------------

def label(c, s, tags):
    dead = post.dead_mask(c["p_dead"], c["blocks"], s["threshold"], s["min_dead_s"], tags.frame,
                          s["snap_s"], s["insert_s"])
    return post.decode(c["p_home"], dead, c["blocks"], s["lam"], tags.frame, tags.team,
                       s["beta"], post.assign_seconds(s["snap_s"]))


def evaluate_copy(work, match, preset, seed, tag_setting, family_settings):
    """Rows for one copy: the HA3 baseline and every setting given (one tag setting)."""
    c = load_copy(work, match, preset, seed)
    gt = pm.label_codes(c["target"])
    runs = pm.valid_runs(gt)
    tags = copy_tags(c, match, preset, seed, tag_setting)
    base = {"threshold": c["ha3_threshold"], "min_dead_s": HA3_MIN_DEAD_S, "lam": c["ha3_lam"],
            "snap_s": 0.0, "insert_s": 0.0, "beta": 0.0}
    rows = []
    cache = {}
    for family, s in [("ha3", base)] + family_settings:
        key = (s["threshold"], s["min_dead_s"], s["snap_s"], s["insert_s"])
        if key not in cache:
            cache[key] = post.dead_mask(c["p_dead"], c["blocks"], s["threshold"],
                                        s["min_dead_s"], tags.frame, s["snap_s"], s["insert_s"])
        pred = post.decode(c["p_home"], cache[key], c["blocks"], s["lam"], tags.frame,
                           tags.team, s["beta"], post.assign_seconds(s["snap_s"]))
        f1, n_gt, n_pred = spell(gt, runs, pred)
        rows.append(dict(s, match=match, preset=preset, seed=int(seed), tags=tag_setting,
                         family=family, setting=setting_id(s), spell_f1=f1, n_true=n_gt,
                         n_pred=n_pred))
    return rows


# -- selection -------------------------------------------------------------------------------------

def _closeness(s, base_thr=0.40):
    """Tie-break: the setting closest to HA3 (fewest and smallest changes)."""
    return (abs(s["threshold"] - base_thr) + abs(s["min_dead_s"] - HA3_MIN_DEAD_S)
            + abs(np.log2(s["lam"] / 0.5)) + s["snap_s"] + s["insert_s"] + s["beta"])


def table(rows):
    """{(family, setting, match, preset): mean spellF1 over seeds} and the settings."""
    acc, specs = {}, {}
    for r in rows:
        k = (r["family"], r["setting"], r["match"], r["preset"])
        acc.setdefault(k, []).append(r["spell_f1"])
        specs[(r["family"], r["setting"])] = {key: r[key] for key in KEYS}
    out = {}
    for k, v in acc.items():
        if len(v) != len(SEEDS):
            raise ValueError("expected {} seeds for {}".format(len(SEEDS), k))
        out[k] = float(np.mean(v))
    return out, specs


def choose(tab, specs, family, matches, preset="doc"):
    best, best_key = None, None
    for (fam, sid), s in specs.items():
        if fam != family:
            continue
        value = float(np.mean([tab[(fam, sid, m, preset)] for m in matches]))
        key = (round(value, 12), -_closeness(s))
        if best_key is None or key > best_key:
            best, best_key = sid, key
    return best


def nested(tab, specs, family):
    """{held-out match: (chosen setting, {preset: gain over HA3})}."""
    out = {}
    for held in DEV:
        sid = choose(tab, specs, family, [m for m in DEV if m != held])
        gains = {}
        for preset in PRESETS:
            gains[preset] = tab[(family, sid, held, preset)] - ha3_score(tab, held, preset)
        out[held] = (sid, gains)
    return out


def ha3_score(tab, match, preset):
    v = [value for (fam, _, m, p), value in tab.items() if fam == "ha3" and m == match and p == preset]
    if len(v) != 1:
        raise ValueError("expected one HA3 baseline for {} {}".format(match, preset))
    return v[0]


def decide(full, a4, pessimistic_gain):
    """The pre-registered rule (design §6). `full`/`a4` are nested() results."""
    def mean(res, preset):
        return float(np.mean([g[preset] for _, g in res.values()]))

    def basic(res):
        doc = [g["doc"] for _, g in res.values()]
        return (mean(res, "doc") >= 0.010 and sum(v > 0 for v in doc) >= 4
                and mean(res, "mild") >= 0 and mean(res, "severe") >= 0)

    checks = {
        "full_basic": basic(full),
        "full_pessimistic_ge_0": pessimistic_gain >= 0,
        "full_beats_a4_by_0.005": mean(full, "doc") - mean(a4, "doc") >= 0.005,
        "a4_basic": basic(a4),
    }
    if checks["full_basic"] and checks["full_pessimistic_ge_0"] and checks["full_beats_a4_by_0.005"]:
        verdict = "ADOPT_HA4_FULL"
    elif checks["a4_basic"]:
        verdict = "ADOPT_HA4_A4_ONLY"
    else:
        verdict = "KEEP_HA3"
    return {"checks": checks, "verdict": verdict,
            "mean_gain": {"full": {p: mean(full, p) for p in PRESETS},
                          "a4": {p: mean(a4, p) for p in PRESETS}}}
