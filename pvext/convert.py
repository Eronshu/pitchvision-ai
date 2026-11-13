"""DFL (IDSSE) tracking -> the competition's ATD text format, read by the frozen V1 unchanged.

Two separate products:

  layout(...)    -> (FramesTrack, target, blocks). FramesTrack is tracking only. `target` holds
                    H / A / D / X built from BallStatus and BallPossession, for scoring only.
  write_atd(...) -> (meta_path, raw_path). It takes a FramesTrack and the match information.
                    It has no parameter through which a label could reach the written files.

Conventions, as measured in P0 on J03WPY (docs/EXTERNAL_DATA.md):
  DFL positions are metres from the centre spot (x along the length); ATD is a corner origin,
  normalised to 0-1 by the pitch. So x_atd = (x + L/2) / L and y_atd = (y + W/2) / W. Orientation
  does not matter to V1, which only uses distances and out-of-bounds tests.
  DFL S is km/h; the ATD speed field and V1 are m/s. Conversion goes only through
  dfl_parse.kmh_to_mps().
  Game sections are laid end to end on one frame axis, with `gap_frames` empty frames between
  them, and each section becomes one graded block. The home team is written first, which is
  what V1's slot-to-team rule (starter/baseline.team_of_each_track) assumes.
"""

import collections
import os
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from pvext import dfl_parse

FramesTrack = collections.namedtuple("FramesTrack", [
    "players_xy", "players_speed_mps", "team_of_slot", "ball_xy", "ball_z", "pitch",
    "speed_unit"])

MISSING_GROUP = "NaN,NaN,NaN"


def target_label(status, possession):
    """The scoring label of one frame: D when dead, H/A when alive with a known side, else X."""
    if status == 0:
        return "D"
    if status == 1 and possession in (1, 2):
        return "H" if possession == 1 else "A"
    return "X"


def layout(track, labels, meta, gap_frames=250):
    """Concatenate the sections onto one frame axis. Returns (FramesTrack, target, blocks).

    blocks are [(section_number, first_frame, last_frame)], inclusive, like
    games.half_video_blocks(). Gap frames have no tracking and target X.
    """
    length, width = meta.pitch
    pieces, targets, blocks, start = [], [], [], 0
    for number, (data, lab) in enumerate(zip(track.data, labels.data), 1):
        if number > 1:
            pieces.append(_empty(gap_frames, len(track.person_ids)))
            targets.append(np.full(gap_frames, "X", dtype="<U1"))
            start += gap_frames
        T = len(data.n)
        shift = np.array([length / 2.0, width / 2.0], dtype=np.float32)
        pieces.append((data.players_xy + shift, dfl_parse.kmh_to_mps(data.players_speed_kmh),
                       data.ball_xy + shift, data.ball_z))
        targets.append(_targets(lab.ball_status, lab.possession))
        blocks.append((number, start, start + T - 1))
        start += T
    players_xy, speed, ball_xy, ball_z = (np.concatenate([p[k] for p in pieces])
                                          for k in range(4))
    frames = FramesTrack(players_xy=players_xy.astype(np.float32),
                         players_speed_mps=speed.astype(np.float32),
                         team_of_slot=np.asarray(track.team_of_person, dtype=np.int8),
                         ball_xy=ball_xy.astype(np.float32), ball_z=ball_z.astype(np.float32),
                         pitch=meta.pitch, speed_unit="m/s")
    return frames, np.concatenate(targets), blocks


def write_atd(frames, meta, dirpath):
    """Write <dirpath>/atd.xml and <dirpath>/atd.txt in the competition ATD format.

    Slot k belongs to team frames.team_of_slot[k] (0 home, 1 away). Coordinates are written
    normalised with 6 decimals (about 0.1 mm), and speed in m/s.
    """
    if frames.speed_unit != "m/s":
        raise ValueError("ATD speed is m/s; got a FramesTrack in {}".format(frames.speed_unit))
    os.makedirs(dirpath, exist_ok=True)
    meta_path = os.path.join(dirpath, "atd.xml")
    raw_path = os.path.join(dirpath, "atd.txt")
    team_ids = ("HOME-" + meta.home_id, "AWAY-" + meta.away_id)

    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<Root>", "  <Teams>"]
    for team_id, name in zip(team_ids, meta.names):
        lines.append("    <Team id={}><Name>{}</Name></Team>".format(quoteattr(team_id),
                                                                     escape(name or "")))
    lines += ["  </Teams>", "  <Players>"]
    for slot, side in enumerate(frames.team_of_slot):
        lines.append('    <Player id="S{0}" teamId={1}><Name>Track {0}</Name></Player>'.format(
            slot + 1, quoteattr(team_ids[int(side)])))
    lines += ["  </Players>", "</Root>"]
    with open(meta_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    length, width = frames.pitch
    px = frames.players_xy[..., 0] / length
    py = frames.players_xy[..., 1] / width
    speed = frames.players_speed_mps
    bx, by = frames.ball_xy[:, 0] / length, frames.ball_xy[:, 1] / width
    with open(raw_path, "w", encoding="ascii") as handle:
        for t in range(len(bx)):
            groups = [MISSING_GROUP if np.isnan(px[t, k]) else
                      "{:.6f},{:.6f},{}".format(px[t, k], py[t, k],
                                                "NaN" if np.isnan(speed[t, k])
                                                else "{:.4f}".format(speed[t, k]))
                      for k in range(px.shape[1])]
            ball = MISSING_GROUP if np.isnan(bx[t]) else "{:.6f},{:.6f},0".format(bx[t], by[t])
            handle.write("{}:{}:{}\n".format(t + 1, ";".join(groups), ball))
    return meta_path, raw_path


def _targets(status, possession):
    out = np.full(len(status), "X", dtype="<U1")
    out[status == 0] = "D"
    alive = status == 1
    out[alive & (possession == 1)] = "H"
    out[alive & (possession == 2)] = "A"
    return out


def _empty(n, persons):
    return (np.full((n, persons, 2), np.nan, dtype=np.float32),
            np.full((n, persons), np.nan, dtype=np.float32),
            np.full((n, 2), np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32))
