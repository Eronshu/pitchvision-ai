"""Independent cross-check: kloppy's Sportec reader against pvext.dfl_parse, frame by frame.

Both read the same XML. They must agree on every frame and player observation: ball and player
positions, BallStatus (alive/dead) and BallPossession (home/away). kloppy is loaded with
only_alive=False, because its default (only_alive=True) silently drops every dead-ball frame.
That default is measured too, so the report shows what it would have hidden.

kloppy is a development cross-check only, not a runtime dependency. It needs several GB of memory
for a full match (about 4 GB for J03WPY), against about 0.25 GB for our parser. The two loads run
one after the other, never at the same time.
"""

import gc
import math

import numpy as np

from pvext import dfl_parse


def compare(meta_xml, positions_xml):
    """Aggregate agreement figures; result["agrees"] is True only when nothing differs."""
    from kloppy import sportec

    default = sportec.load_tracking(meta_data=meta_xml, raw_data=positions_xml,
                                    coordinates="sportec")
    alive_only_frames = len(default.records)
    del default
    gc.collect()

    dataset = sportec.load_tracking(meta_data=meta_xml, raw_data=positions_xml,
                                    coordinates="sportec", only_alive=False)
    track, labels, _ = dfl_parse.parse(meta_xml, positions_xml)
    index = {}
    for data, lab in zip(track.data, labels.data):
        for i, n in enumerate(data.n):
            index[int(n)] = (data, lab, i)
    column = {pid: c for c, pid in enumerate(track.person_ids)}
    home_id = dataset.metadata.teams[0].team_id

    missing = status_bad = possession_bad = player_obs = player_missing = 0
    unknown_labels = non_finite = 0
    ball_diff = player_diff = 0.0

    def gap(theirs, ours):
        """|theirs - ours|, or None when either side is not finite. max() would silently drop
        a NaN, so a non-finite pair is counted and can never pass as agreement."""
        ours = float(ours)
        if not (math.isfinite(theirs) and math.isfinite(ours)):
            return None
        return abs(theirs - ours)

    for record in dataset.records:
        entry = index.get(int(record.frame_id))
        if entry is None:
            missing += 1
            continue
        data, lab, i = entry
        for theirs, ours in ((record.ball_coordinates.x, data.ball_xy[i, 0]),
                             (record.ball_coordinates.y, data.ball_xy[i, 1])):
            diff = gap(theirs, ours)
            if diff is None:
                non_finite += 1
            else:
                ball_diff = max(ball_diff, diff)
        # Unknown (-1) on our side is never an agreement, even where kloppy's defaults
        # (missing status -> dead, missing possession -> away) would happen to line up.
        if lab.ball_status[i] not in (0, 1) or lab.possession[i] not in (1, 2):
            unknown_labels += 1
        status_bad += (record.ball_state.value == "alive") != (lab.ball_status[i] == 1)
        possession_bad += (record.ball_owning_team.team_id == home_id) != (lab.possession[i] == 1)
        for player, point in record.players_coordinates.items():
            player_obs += 1
            j = column.get(player.player_id)
            if j is None or np.isnan(data.players_xy[i, j, 0]):
                player_missing += 1
                continue
            for theirs, ours in ((point.x, data.players_xy[i, j, 0]),
                                 (point.y, data.players_xy[i, j, 1])):
                diff = gap(theirs, ours)
                if diff is None:
                    non_finite += 1
                else:
                    player_diff = max(player_diff, diff)

    ours_frames = sum(len(d.n) for d in track.data)
    ours_players = sum(int((~np.isnan(d.players_xy[..., 0])).sum()) for d in track.data)
    result = {
        "frames": {"kloppy": len(dataset.records), "ours": ours_frames,
                   "kloppy_missing_in_ours": missing},
        "ball_max_abs_diff_m": round(float(ball_diff), 6),
        "status_mismatches": int(status_bad),
        "unknown_labels": int(unknown_labels),
        "non_finite_coordinates": int(non_finite),
        "possession_mismatches": int(possession_bad),
        "player_observations": {"kloppy": player_obs, "ours": ours_players,
                                "kloppy_missing_in_ours": player_missing},
        "player_max_abs_diff_m": round(float(player_diff), 6),
        "kloppy_only_alive_default_frames": alive_only_frames,
        "kloppy_coordinate_system": {
            "origin": str(dataset.metadata.coordinate_system.origin),
            "vertical": str(dataset.metadata.coordinate_system.vertical_orientation)},
    }
    result["agrees"] = bool(missing == 0 and ours_frames == len(dataset.records)
                        and unknown_labels == 0 and non_finite == 0
                        and ball_diff < 1e-4 and status_bad == 0 and possession_bad == 0
                        and player_missing == 0 and ours_players == player_obs
                        and player_diff < 1e-4)
    return result
