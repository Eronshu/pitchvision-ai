"""Raw ATD -> numpy arrays on the video-frame index.

Same line format and the same slot-to-team rule as starter/baseline.py, which this module imports
rather than re-derives: "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>", NaN for a missing observation,
video_frame = frame_id - 1. What it adds is the whole position array instead of only the nearest
team, and loud failures where the baseline is silent: a malformed line, a slot count that
disagrees with the metadata, or a slot that belongs to neither team are errors here, not skipped
lines or silent away players.
"""

import collections
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from starter.baseline import FIELD_SEP, GROUP_SEP, NOBODY, team_of_each_track  # noqa: E402

Tracking = collections.namedtuple(
    "Tracking", ["players_xy", "players_speed", "team_of_slot", "ball_xy", "ball_present"])


def check_teams(meta_path):
    """Number of slots in the metadata, after checking every slot names one of the two teams.

    team_of_each_track maps anything that is not the home id to away, so a referee or a third
    team id would quietly become an away player. Checked here, before that mapping runs.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    strangers = sorted(set(slots) - set(team_ids))
    if strangers:
        raise ValueError("{}: Player teamId {} is neither team {}".format(
            meta_path, strangers, team_ids))
    return len(slots)


def load_arrays(meta_path, raw_path, n_frames, pitch):
    """The whole ATD as a Tracking, positions in metres, indexed by video frame."""
    n_slots = check_teams(meta_path)
    team_of_slot = team_of_each_track(meta_path)
    length, width = pitch

    players_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    players_speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)

    line_no = 0
    with open(raw_path) as handle:
        for line_no, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                raise ValueError("{} line {}: expected 3 '{}'-separated fields, got {}".format(
                    raw_path, line_no, FIELD_SEP, len(parts)))
            frame_id = int(parts[0])
            _check_sequence(raw_path, line_no, frame_id, n_frames)
            frame = frame_id - 1  # frame_id counts from 1, video frames from 0

            groups = parts[1].split(GROUP_SEP)
            if len(groups) != n_slots:
                raise ValueError("{} line {}: {} slot groups, metadata {} has {}".format(
                    raw_path, line_no, len(groups), meta_path, n_slots))
            for slot, group in enumerate(groups):
                values = group.split(",")
                if len(values) < 3:
                    raise ValueError("{} line {}: slot {} is {!r}, expected x,y,speed".format(
                        raw_path, line_no, slot + 1, group))
                players_xy[frame, slot, 0] = float(values[0]) * length
                players_xy[frame, slot, 1] = float(values[1]) * width
                players_speed[frame, slot] = float(values[2])

            ball = parts[2].split(",")
            if len(ball) < 2:
                raise ValueError("{} line {}: ball is {!r}, expected x,y,...".format(
                    raw_path, line_no, parts[2]))
            ball_xy[frame, 0] = float(ball[0]) * length
            ball_xy[frame, 1] = float(ball[1]) * width

    if line_no != n_frames:
        raise ValueError("{}: {} records, expected {} (one per video frame)".format(
            raw_path, line_no, n_frames))
    return Tracking(players_xy=players_xy, players_speed=players_speed,
                    team_of_slot=team_of_slot, ball_xy=ball_xy,
                    ball_present=~np.isnan(ball_xy[:, 0]))


def _check_sequence(raw_path, line_no, frame_id, n_frames):
    """Line k must carry frame_id k: one record per video frame, in order, none missing.

    A duplicated, missing or reordered record would otherwise shift or blank frames silently --
    the off-by-one the submission's frame column exists to catch, one step earlier.
    """
    if frame_id > n_frames:
        raise ValueError("{} line {}: frame_id {} outside 1..{}".format(
            raw_path, line_no, frame_id, n_frames))
    if frame_id < line_no:
        raise ValueError("{}: duplicate or out-of-order frame_id {} at line {}".format(
            raw_path, frame_id, line_no))
    if frame_id > line_no:
        missing = str(line_no) if frame_id == line_no + 1 else "{}..{}".format(
            line_no, frame_id - 1)
        raise ValueError("{}: missing frame_id {} before line {}".format(
            raw_path, missing, line_no))


def nearest_team(tr):
    """Per-frame team of the slot nearest the ball (0/1), NOBODY (-1) with no ball or no player.

    The baseline's rule on arrays: squared distance in metres, ties to the lower slot index.
    """
    d2 = ((tr.players_xy - tr.ball_xy[:, None, :]) ** 2).sum(axis=2)
    seen = ~np.isnan(d2)
    out = np.full(len(d2), NOBODY, dtype=np.int8)
    rows = seen.any(axis=1)
    if rows.any():
        best = np.argmin(np.where(seen[rows], d2[rows], np.inf), axis=1)
        out[rows] = tr.team_of_slot[best]
    return out
