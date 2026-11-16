"""DFL (IDSSE) set-piece events, and the simulated Smart-Tagging-like tags built from them.

HA4 (design: docs/superpowers/specs/2026-11-13-pitchvision-ha4-design.md) uses set-piece tags at
prediction time. The competition provides them as Metrica's automatic Smart Tagging (`SET PIECES`
intervals with a `Type` and a `Team` that is `N/A` for every throw-in). IDSSE has no Smart
Tagging, but its official DFL event feed records every restart: KickOff, ThrowIn, CornerKick,
GoalKick, FreeKick and Penalty, each with an execution time (`EventTime`) and the executing team.

`simulate()` turns those events into tags with the competition tags' known defects: per-type
recall, timing jitter and bias, no team on throw-ins, occasional wrong team, false positives. Every
setting is a stated assumption (SIM below) and the simulation is seeded. Only IDSSE data passes
through this module; labels (BallStatus/BallPossession) never do.
"""

import collections
import datetime
import xml.etree.ElementTree as ET

import numpy as np

# Params
FPS = 25
TYPES = ("KickOff", "ThrowIn", "CornerKick", "GoalKick", "FreeKick", "Penalty")
HOME, AWAY, NO_TEAM = 0, 1, -1

# Tag-quality settings (assumptions, design §3). recall: share of real restarts that get a tag,
# from the competition match's published tag counts (DATA.md) against typical DFL restart counts.
# jitter_s / bias_s: tag start minus the real execution time. flip: wrong team share (non-throw-ins).
# fp_per_half: false-positive tags at random alive times.
SIM = {
    "clean": {"recall": {t: 1.0 for t in TYPES}, "jitter_s": 0.0, "bias_s": 0.0, "flip": 0.0,
              "fp_per_half": 0.0, "throwin_team": True},
    "doc": {"recall": {"KickOff": 1.0, "ThrowIn": 0.65, "CornerKick": 0.9, "GoalKick": 0.6,
                       "FreeKick": 0.25, "Penalty": 0.5},
            "jitter_s": 1.0, "bias_s": 0.0, "flip": 0.05, "fp_per_half": 2.0,
            "throwin_team": False},
    "pessimistic": {"recall": {"KickOff": 0.9, "ThrowIn": 0.5, "CornerKick": 0.75,
                               "GoalKick": 0.45, "FreeKick": 0.15, "Penalty": 0.5},
                    "jitter_s": 2.0, "bias_s": -2.0, "flip": 0.15, "fp_per_half": 5.0,
                    "throwin_team": False},
}
SIM_SEED_BASE = 7000

SetPiece = collections.namedtuple("SetPiece", ["kind", "time", "team"])
Tags = collections.namedtuple("Tags", ["frame", "team", "kind"])     # arrays, sorted by frame


def parse_time(text):
    return datetime.datetime.fromisoformat(text)


def parse_setpieces(events_xml, meta):
    """[SetPiece] of one match: every restart event with its execution time and team (0 home,
    1 away, -1 unknown)."""
    root = ET.parse(events_xml).getroot()
    teams = {meta.home_id: HOME, meta.away_id: AWAY}
    out = []
    for event in root.iter("Event"):
        for child in event:
            if child.tag not in TYPES:
                continue
            team_id = child.get("Team")
            if team_id is None:
                play = child.find("Play")
                team_id = play.get("Team") if play is not None else None
            out.append(SetPiece(child.tag, parse_time(event.get("EventTime")),
                                teams.get(team_id, NO_TEAM)))
    return sorted(out, key=lambda s: s.time)


def frame_axis(track, blocks, fps=FPS):
    """Per section: (block first frame, block last frame, time of the ball's first frame, frame
    offset of the ball's first frame inside the section). blocks come from convert.layout."""
    out = []
    for section, data, (_, first, last) in zip(track.sections, track.data, blocks):
        n_first_ball = track.ball_n_range[section][0]
        offset = int(n_first_ball - int(data.n[0]))
        out.append((first, last, parse_time(track.t_range[section][0]), offset))
    return out


def to_frames(setpieces, axis, fps=FPS):
    """Tags with exact frames: (frame index on the layout axis, team, kind) for every set piece
    that falls inside a block."""
    frames, teams, kinds = [], [], []
    for s in setpieces:
        for first, last, t0, offset in axis:
            f = first + offset + int(round((s.time - t0).total_seconds() * fps))
            if first <= f <= last:
                frames.append(f)
                teams.append(s.team)
                kinds.append(s.kind)
                break
    order = np.argsort(np.asarray(frames, dtype=np.int64), kind="stable")
    return Tags(np.asarray(frames, dtype=np.int64)[order], np.asarray(teams, dtype=np.int8)[order],
                np.asarray(kinds, dtype=object)[order])


def simulate(exact, blocks, setting, seed, fps=FPS):
    """Degrade exact tags the way the competition's automatic tags are degraded (SIM[setting])."""
    cfg = SIM[setting]
    rng = np.random.default_rng(seed)
    frames, teams, kinds = [], [], []
    for f, team, kind in zip(exact.frame, exact.team, exact.kind):
        if rng.random() >= cfg["recall"][kind]:
            continue
        shift = cfg["bias_s"] + (rng.normal(0.0, cfg["jitter_s"]) if cfg["jitter_s"] else 0.0)
        g = int(f) + int(round(shift * fps))
        t = int(team)
        if kind == "ThrowIn" and not cfg["throwin_team"]:
            t = NO_TEAM
        elif t != NO_TEAM and rng.random() < cfg["flip"]:
            t = 1 - t
        frames.append(g)
        teams.append(t)
        kinds.append(kind)
    for _, first, last in blocks:
        for _ in range(rng.poisson(cfg["fp_per_half"])):
            frames.append(int(rng.integers(first, last + 1)))
            teams.append(int(rng.integers(0, 2)) if rng.random() < 0.5 else NO_TEAM)
            kinds.append("FalsePositive")
    inside = [i for i, g in enumerate(frames) if any(a <= g <= b for _, a, b in blocks)]
    frames = np.asarray([frames[i] for i in inside], dtype=np.int64)
    order = np.argsort(frames, kind="stable")
    return Tags(frames[order], np.asarray([teams[i] for i in inside], dtype=np.int8)[order],
                np.asarray([kinds[i] for i in inside], dtype=object)[order])
