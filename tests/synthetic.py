"""Synthetic ATD fixtures. Nothing here touches real match data.

Two ways to get a Tracking for a test:

  write_fixture()  writes a tiny raw ATD txt + meta XML in the real export's line format, for the
                   loader and end-to-end tests.
  make_tracking()  builds a Tracking in memory, in metres, for feature and state-machine tests.
"""

import os

import numpy as np

from pitchvision.atd import Tracking

MISSING_GROUP = "NaN,NaN,NaN"


def write_meta(path, teams=("T1", "T2"), names=("Home FC", "Away FC"), slot_teams=()):
    """A meta XML with exactly what team_of_each_track and team_names read."""
    lines = ["<Root>", "  <Teams>"]
    for team_id, name in zip(teams, names):
        lines.append('    <Team id="{}"><Name>{}</Name></Team>'.format(team_id, name))
    lines += ["  </Teams>", "  <Players>"]
    for slot, team_id in enumerate(slot_teams):
        lines.append('    <Player id="P{}" teamId="{}"><Name>Track {}</Name></Player>'.format(
            slot + 1, team_id, slot + 1))
    lines += ["  </Players>", "</Root>"]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def raw_line(frame_id, players, ball):
    """One raw ATD line: '<frame_id>:<x,y,s>;...:<x,y,s>'. None means a missing observation."""
    groups = [MISSING_GROUP if p is None else "{},{},{}".format(*p) for p in players]
    ball_group = MISSING_GROUP if ball is None else "{},{},0".format(*ball)
    return "{}:{}:{}\n".format(frame_id, ";".join(groups), ball_group)


def write_fixture(dirpath, frames, teams=("T1", "T2"), slot_teams=None):
    """Write <dirpath>/atd.xml and <dirpath>/atd.txt. Returns (meta_path, raw_path).

    `frames` is a list of {"players": [(x, y, s) | None, ...], "ball": (x, y) | None} in
    normalised 0-1 coordinates; frame_id counts from 1 like the real export. `slot_teams` defaults
    to alternating home/away slots.
    """
    n_slots = len(frames[0]["players"])
    if slot_teams is None:
        slot_teams = [teams[slot % 2] for slot in range(n_slots)]
    meta_path = write_meta(os.path.join(str(dirpath), "atd.xml"), teams=teams,
                           slot_teams=slot_teams)
    raw_path = os.path.join(str(dirpath), "atd.txt")
    with open(raw_path, "w") as handle:
        for index, frame in enumerate(frames):
            handle.write(raw_line(index + 1, frame["players"], frame["ball"]))
    return meta_path, raw_path


def make_tracking(players_xy_m, ball_xy_m, team_of_slot, players_speed=None):
    """A Tracking from metre arrays. NaN marks a missing observation."""
    players_xy = np.asarray(players_xy_m, dtype=np.float32)
    ball_xy = np.asarray(ball_xy_m, dtype=np.float32)
    if players_speed is None:
        players_speed = np.where(np.isnan(players_xy[..., 0]), np.nan, 0.0)
    return Tracking(
        players_xy=players_xy,
        players_speed=np.asarray(players_speed, dtype=np.float32),
        team_of_slot=np.asarray(team_of_slot, dtype=np.int8),
        ball_xy=ball_xy,
        ball_present=~np.isnan(ball_xy[:, 0]),
    )


def write_smart_tagging(path, instances):
    """A SportsCode XML in the shape starter/load_smart_tagging.load reads.

    `instances` is a list of (code, start_s, end_s, {group: text}).
    """
    from xml.sax.saxutils import escape
    lines = ["<file>", "  <ALL_INSTANCES>"]
    for number, (code, start, end, tags) in enumerate(instances, 1):
        lines.append("    <instance><ID>{}</ID><start>{}</start><end>{}</end>"
                     "<code>{}</code>".format(number, start, end, escape(code)))
        for group, text in tags.items():
            lines.append("      <label><group>{}</group><text>{}</text></label>".format(
                escape(group), escape(text)))
        lines.append("    </instance>")
    lines += ["  </ALL_INSTANCES>", "</file>"]
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(str(path), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return str(path)
