"""P0 feasibility probe: what one IDSSE match actually contains, as aggregates only.

    python scripts/inspect_idsse.py --match J03WPY --out /path/outside/repo/J03WPY_probe.json
    python scripts/inspect_idsse.py --meta META.xml --positions POSITIONS.xml --out OUT.json

Per game section it reports:

- the timeline (N range, frames, ball gaps) and the frame rate measured from the ball's own
  first/last N and T, next to the half duration the match information states;
- the coordinate ranges, the ball at the section's first frame, and the share of observations
  outside a centred pitch vs a corner-origin pitch, so the origin can be inferred rather than
  assumed;
- players visible per frame, ball presence and height;
- the targets, BallStatus and BallPossession: value counts, dead-run lengths, alive possession
  spells (split by dead runs, as spellF1 counts them), switches per alive minute, what possession
  says while the ball is dead, and ball speed in dead vs alive frames.

Nothing per-frame leaves this script. The targets are only summarised; they feed nothing.
"""

import argparse
import datetime
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pvext import dfl_parse, idsse  # noqa: E402

# Params
FPS_NOMINAL = 25.0
OUTSIDE_MARGIN_M = 1.0
TOTAL_TIME_KEYS = {"firstHalf": "TotalTimeFirstHalf", "secondHalf": "TotalTimeSecondHalf",
                   "firstHalfExtra": "TotalTimeFirstHalfExtra",
                   "secondHalfExtra": "TotalTimeSecondHalfExtra"}


def _runs(mask):
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _p(values, q):
    return [round(float(v), 3) for v in np.percentile(values, q)] if len(values) else None


def _counts(values, keys):
    return {str(k): int(np.sum(values == k)) for k in keys}


def _seconds(t):
    return datetime.datetime.fromisoformat(t)


def _total_times(meta_xml):
    other = ET.parse(meta_xml).getroot().find(".//OtherGameInformation")
    return {} if other is None else dict(other.attrib)


def section_report(name, data, lab, pitch, total_times):
    length, width = pitch
    status, possession = lab.ball_status, lab.possession
    present = ~np.isnan(data.ball_xy[:, 0])
    alive, dead = status == 1, status == 0

    dead_runs = [b - a for a, b in _runs(dead)]
    spells, switches = [], 0
    for a, b in _runs(alive):
        seg = possession[a:b]
        breaks = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        switches += len(breaks)
        bounds = np.concatenate(([0], breaks, [len(seg)]))
        spells += [int(bounds[i + 1] - bounds[i]) for i in range(len(bounds) - 1)
                   if seg[bounds[i]] in (1, 2)]

    velocity = np.full(len(present), np.nan)
    step = np.linalg.norm(np.diff(data.ball_xy.astype(np.float64), axis=0), axis=1)
    velocity[1:] = step * FPS_NOMINAL

    pts_x = np.concatenate([data.ball_xy[:, 0], data.players_xy[..., 0].ravel()])
    pts_y = np.concatenate([data.ball_xy[:, 1], data.players_xy[..., 1].ravel()])
    seen = ~np.isnan(pts_x)
    x, y = pts_x[seen], pts_y[seen]
    m = OUTSIDE_MARGIN_M
    outside_centred = (np.abs(x) > length / 2 + m) | (np.abs(y) > width / 2 + m)
    outside_corner = (x < -m) | (x > length + m) | (y < -m) | (y > width + m)

    players_seen = (~np.isnan(data.players_xy[..., 0])).sum(axis=1)
    alive_minutes = alive.sum() / FPS_NOMINAL / 60.0
    return {
        "frames": int(len(data.n)),
        "n_range": [int(data.n[0]), int(data.n[-1])],
        "ball_gaps": int(len(_runs(~present))),
        "ball_present_share": round(float(present.mean()), 6),
        "fps_from_timestamps": None,
        "total_time_ms_meta": (int(float(total_times[TOTAL_TIME_KEYS[name]]))
                               if TOTAL_TIME_KEYS.get(name) in total_times else None),
        "coordinates": {
            "ball_x_min_max": [round(float(np.nanmin(data.ball_xy[:, 0])), 3),
                               round(float(np.nanmax(data.ball_xy[:, 0])), 3)],
            "ball_y_min_max": [round(float(np.nanmin(data.ball_xy[:, 1])), 3),
                               round(float(np.nanmax(data.ball_xy[:, 1])), 3)],
            "player_x_p1_p99": _p(data.players_xy[..., 0][~np.isnan(data.players_xy[..., 0])],
                                  [1, 99]),
            "player_y_p1_p99": _p(data.players_xy[..., 1][~np.isnan(data.players_xy[..., 1])],
                                  [1, 99]),
            "kickoff_ball_xy": [round(float(v), 3) for v in data.ball_xy[0]],
            "share_outside_centred_pitch": round(float(outside_centred.mean()), 6),
            "share_outside_corner_pitch": round(float(outside_corner.mean()), 6),
        },
        "players_per_frame": {"min_mean_max": [int(players_seen.min()),
                                               round(float(players_seen.mean()), 3),
                                               int(players_seen.max())]},
        "ball_z_min_p99_max": [round(float(np.nanmin(data.ball_z)), 3),
                               round(float(np.nanpercentile(data.ball_z, 99)), 3),
                               round(float(np.nanmax(data.ball_z)), 3)],
        "labels": {
            "ball_status": _counts(status, (0, 1, -1)),
            "possession": _counts(possession, (1, 2, -1)),
            "dead_share": round(float(dead.mean()), 6),
            "dead_runs": {"count": len(dead_runs),
                          "duration_s_p10_p50_p90": _p(np.array(dead_runs) / FPS_NOMINAL,
                                                       [10, 50, 90])},
            "alive_spells": {"count": len(spells),
                             "duration_s_p10_p50_p90": _p(np.array(spells) / FPS_NOMINAL,
                                                          [10, 50, 90]),
                             "mean_s": (round(float(np.mean(spells)) / FPS_NOMINAL, 3)
                                        if spells else None),
                             "switches_per_alive_min": (round(switches / alive_minutes, 3)
                                                        if alive_minutes else None)},
            "possession_during_dead": _counts(possession[dead], (1, 2, -1)),
            "ball_speed_median_dead_alive_mps": [
                round(float(np.nanmedian(velocity[dead])), 3) if dead.any() else None,
                round(float(np.nanmedian(velocity[alive])), 3) if alive.any() else None],
        },
    }


def report(meta_xml, positions_xml):
    track, labels, meta = dfl_parse.parse(meta_xml, positions_xml)
    total_times = _total_times(meta_xml)
    sections = {}
    for name, data, lab in zip(track.sections, track.data, labels.data):
        entry = section_report(name, data, lab, meta.pitch, total_times)
        # Pair the ball's first/last N with the ball's first/last T: the union timeline can be
        # longer than the ball stream when player frames extend beyond it.
        first, last = track.t_range[name]
        n_first, n_last = track.ball_n_range[name]
        entry["ball_n_range"] = [n_first, n_last]
        if first and last and n_last > n_first:
            span = (_seconds(last) - _seconds(first)).total_seconds()
            entry["fps_from_timestamps"] = round((n_last - n_first) / span, 4) if span else None
        sections[name] = entry
    return {
        "match_id": meta.match_id,
        "pitch_match_information": list(meta.pitch),
        "pitch_positions_metadata": (list(track.census["pitch_size"])
                                     if track.census["pitch_size"] else None),
        "persons": {"home": int(np.sum(track.team_of_person == 0)),
                    "away": int(np.sum(track.team_of_person == 1))},
        "skipped_framesets": track.skipped_framesets,
        "label_value_problems": labels.counts,
        "frame_attributes": {"ball": len(track.census["ball_attrs"]),
                             "player": len(track.census["player_attrs"])},
        "sections": sections,
        "attribution": idsse.ATTRIBUTION,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", help="an IDSSE match id; files read from --root")
    parser.add_argument("--root", default=idsse.DEFAULT_ROOT)
    parser.add_argument("--meta")
    parser.add_argument("--positions")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.match:
        paths = idsse.local_paths(args.match, args.root)
        meta_xml, positions_xml = paths["meta"], paths["positions"]
    else:
        meta_xml, positions_xml = args.meta, args.positions
    result = report(meta_xml, positions_xml)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
