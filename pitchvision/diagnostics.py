"""L1 diagnostics: what a labelling claims about the match, without any ground truth.

Plausibility, not accuracy. These numbers say whether a submission looks like football -- how many
possessions, how long, how much dead ball, how often it flips -- and how stable it is when the ball
track gets worse. They cannot say whether it is right, and nothing here reads a ground truth.

Everything is computed the way the official scorer sees a submission: per graded block, with X
forward-filled inside each block (possession_metrics.fill_abstentions), so a count here means the
same as a count on the board. Only aggregates come out; no per-frame array leaves this module.
"""

import os
import sys

import numpy as np

from pitchvision import tagging
from pitchvision.atd import Tracking

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "evaluation") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


def block_runs(blocks):
    """games.half_video_blocks() -> the half-open [(start, stop)] runs possession_metrics uses."""
    return [(first, last + 1) for _, first, last in blocks]


def summarize(labels, blocks, fps):
    """Aggregate statistics per block ('half_<n>') and over all blocks ('total')."""
    codes = pm.label_codes(labels)
    runs = block_runs(blocks)
    filled = pm.fill_abstentions(codes, runs)
    out = {"half_{}".format(half): _stats(filled, [run], fps)
           for (half, _, _), run in zip(blocks, runs)}
    out["total"] = _stats(filled, runs, fps)
    return out


def agreement(labels_a, labels_b, blocks):
    """How far labels_a agrees with labels_b (the reference) on the graded frames."""
    runs = block_runs(blocks)
    a, b = pm.label_codes(labels_a), pm.label_codes(labels_b)
    graded = np.zeros(len(a), dtype=bool)
    for start, stop in runs:
        graded[start:stop] = True
    spells = pm.spell_scores(pm.fill_abstentions(b, runs), pm.fill_abstentions(a, runs), runs)
    return {"frame_agreement": float(np.mean(a[graded] == b[graded])),
            "spell_f1_mean": float(spells["mean_f1"])}


def perturb_ball(tr, frac, seed):
    """A copy of `tr` with `frac` of the observed ball frames removed, chosen with `seed`."""
    present = np.flatnonzero(tr.ball_present)
    drop = np.random.RandomState(seed).choice(present, int(round(frac * len(present))),
                                              replace=False)
    ball_xy = tr.ball_xy.copy()
    ball_xy[drop] = np.nan
    return Tracking(players_xy=tr.players_xy, players_speed=tr.players_speed,
                    team_of_slot=tr.team_of_slot, ball_xy=ball_xy,
                    ball_present=~np.isnan(ball_xy[:, 0]))


def _stats(filled, runs, fps):
    spells = pm.possession_spells(filled, runs)
    durations = np.array([stop - start for start, stop, _ in spells], dtype=float) / fps
    dead = np.array([stop - start for start, stop, code in pm.segments(filled, runs)
                     if code == pm.D], dtype=float) / fps
    n_frames = sum(stop - start for start, stop in runs)
    graded = np.concatenate([filled[start:stop] for start, stop in runs])
    home, away = int(np.sum(graded == pm.H)), int(np.sum(graded == pm.A))
    minutes = n_frames / float(fps) / 60.0
    return {
        "n_spells": len(spells),
        "duration_p10_p50_p90_s": (np.percentile(durations, [10, 50, 90]).tolist()
                                   if len(durations) else None),
        "share_spells_lt_1s": float(np.mean(durations < 1.0)) if len(durations) else None,
        "transitions_per_min": (len(pm.transitions(filled, runs)) / minutes
                                if minutes else None),
        "d_share_graded": float(np.sum(graded == pm.D)) / n_frames if n_frames else None,
        "d_run_p50_s": float(np.median(dead)) if len(dead) else None,
        "home_share": home / float(home + away) if home + away else None,
    }


def setpiece_consistency(labels, events, cfg, blocks, tmap):
    """L2: how the labelling lines up with Smart Tagging's SET PIECES, inside graded blocks.

    restart_timing  share of set-piece tags that contain a predicted DEAD -> H/A restart.
    restart_team    among those with a named team (N/A excluded), the share whose restart goes
                    to the tagged team.

    A tag counts when its interval overlaps a graded block; only restarts inside graded blocks
    can match it.

    Both figures carry `independent`, False whenever any tagging input fed the model in this run.
    Either cue can move either figure: the dead cue creates stoppages and so restarts that the team
    figure then scores, and the team cue turns unresolved restart runs (X, forward-filled to D)
    into possessions that the timing figure then counts. Agreement with an input is partly
    agreement with itself, so it is reported but must not be read as validation.
    """
    runs = block_runs(blocks)
    filled = pm.fill_abstentions(pm.label_codes(labels), runs)
    restarts = [(frame, to) for frame, frm, to in pm.transitions(filled, runs)
                if frm == pm.D and to in (pm.H, pm.A)]
    tags = [event for event in events
            if (event["code"] or "").strip().casefold() == tagging.SET_PIECES
            and any(event["start_frame"] < stop and event["end_frame"] > start
                    for start, stop in runs)]

    hits, compared, agreed = 0, 0, 0
    for event in tags:
        inside = [to for frame, to in restarts
                  if event["start_frame"] <= frame < event["end_frame"]]
        if not inside:
            continue
        hits += 1
        team = tmap.get(event["tags"].get("Team"), -1)
        if team >= 0:
            compared += 1
            agreed += int(inside[0] == (pm.H if team == 0 else pm.A))

    independent = not any(cfg["tagging"].values())
    return {
        "n_setpiece_tags": len(tags),
        "restart_timing": {
            "share_tags_with_restart": hits / float(len(tags)) if tags else None,
            "independent": independent},
        "restart_team": {
            "agreement": agreed / float(compared) if compared else None,
            "n_compared": compared,
            "independent": independent},
    }
