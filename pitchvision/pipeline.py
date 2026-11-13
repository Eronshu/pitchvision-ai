"""Compose the V1 stages into one label per video frame.

Features and the nearest team are computed once over the whole recording. Each graded block (half)
is then labelled on its own, so no state, smoothing window or dead-ball run crosses halftime.
Frames outside every block are X, which the scorer never grades.
"""

import numpy as np

from pitchvision import alive, atd, features, possession, tagging


def uses_tagging(cfg):
    """True when any Smart Tagging cue is switched on."""
    return any(cfg["tagging"].values())


def run(tr, cfg, fps, pitch, blocks, events=None, team_names=None):
    """(T,) labels in {'H', 'A', 'D', 'X'}.

    `blocks` is games.half_video_blocks() output: [(half, first_frame, last_frame)], inclusive.
    `events` (starter/load_smart_tagging.load output) and `team_names` (the ATD's (home, away))
    are needed only when a tagging flag is on; with both flags off they are ignored entirely.
    """
    tag_dead = restart = None
    if uses_tagging(cfg):
        if events is None:
            raise ValueError("a tagging flag is on but no Smart Tagging events were passed")
        if team_names is None:
            raise ValueError("a tagging flag is on but no ATD team names were passed")
        n = len(tr.ball_xy)
        if cfg["tagging"]["use_setpiece_dead"]:
            tag_dead = tagging.setpiece_dead(events, n)
        if cfg["tagging"]["use_setpiece_team"]:
            restart = tagging.restart_team(
                events, n, tagging.team_map(tagging.tag_teams(events), team_names))

    feat = features.compute(tr, cfg["features"], fps, pitch)
    nearest = atd.nearest_team(tr)

    out = np.full(len(tr.ball_xy), "X", dtype="<U1")
    for _, first, last in blocks:
        span = slice(first, last + 1)
        block_feat = features.Features(*(array[span] for array in feat))
        dead = alive.dead_mask(block_feat, cfg["alive"], fps,
                               None if tag_dead is None else tag_dead[span])
        out[span] = possession.label_block(block_feat, dead, nearest[span], cfg["possession"],
                                           fps, None if restart is None else restart[span])
    return out
