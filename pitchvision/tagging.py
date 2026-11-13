"""Optional Smart Tagging evidence, in the shape starter/load_smart_tagging.load returns.

Two cues, each behind its own config flag, so either can be switched off independently:

  setpiece_dead   1 inside every SET PIECES interval: a soft dead-ball cue (alive.py, `w_tag`).
  restart_team    the tagged team inside a SET PIECES interval: seeds who has the ball when play
                  restarts there (possession.py), instead of starting UNKNOWN.

The tagging is automatic and noisy, and its intervals are phases rather than restarts, so neither
cue is decisive: the dead cue is a nudge below `dead_on`, and the restart seed is overturned by
the usual switch rule as soon as the other team shows clear control. Which side a set piece's Team
tag names (the restarting side is assumed) is not documented, which is a limitation to check in
L2 diagnostics with the flag off.

`N/A` -- every throw-in, among others -- maps to no team. Any other name that does not match an
ATD team after case folding and whitespace collapsing is an error, not a silent "no team".
"""

import numpy as np

# Params
SET_PIECES = "set pieces"
NO_TEAM = "n/a"
NOBODY = -1


def _norm(name):
    return " ".join(name.split()).casefold()


def tag_teams(events):
    """Every Team tag value in the events."""
    return {event["tags"].get("Team") for event in events}


def team_map(tag_teams, atd_names):
    """{tag team name: 0 home / 1 away / -1 none} for every name in tag_teams (None ignored)."""
    sides = {_norm(name): side for side, name in enumerate(atd_names)}
    out = {}
    for name in tag_teams:
        if name is None:
            continue
        key = _norm(name)
        if key == NO_TEAM:
            out[name] = NOBODY
        elif key in sides:
            out[name] = sides[key]
        else:
            raise ValueError("Smart Tagging team {!r} matches neither ATD team {}".format(
                name, list(atd_names)))
    return out


def _setpieces(events, n_frames):
    for event in events:
        if (event["code"] or "").strip().casefold() == SET_PIECES:
            start, stop = max(0, event["start_frame"]), min(n_frames, event["end_frame"])
            if stop > start:
                yield start, stop, event


def setpiece_dead(events, n_frames):
    """(T,) float32: 1 inside SET PIECES intervals, 0 elsewhere."""
    out = np.zeros(n_frames, dtype=np.float32)
    for start, stop, _ in _setpieces(events, n_frames):
        out[start:stop] = 1.0
    return out


def restart_team(events, n_frames, tmap):
    """(T,) int8: the tagged team (0/1) inside SET PIECES intervals, -1 elsewhere or for N/A.
    Later intervals win where two overlap."""
    out = np.full(n_frames, NOBODY, dtype=np.int8)
    for start, stop, event in _setpieces(events, n_frames):
        team = tmap.get(event["tags"].get("Team"), NOBODY)
        if team != NOBODY:
            out[start:stop] = team
    return out
