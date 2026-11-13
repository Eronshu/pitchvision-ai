"""Streaming parser for DFL match-information and positions XML (the open IDSSE files).

The positions file holds one FrameSet per person, or the ball, per GameSection. Each FrameSet holds
Frame elements carrying N (frame number), T (time), X, Y and S, and on the ball also Z,
BallPossession (1 home, 2 away) and BallStatus (0 inactive, 1 active). The file is around 370 MB,
so it is read with iterparse and each element is cleared as soon as it has been read.

What comes out is deliberately split in two:

  DFLTrack   positions only: what a labeller may read. Units are in UNITS: metres from the
             centre spot, and speed in km/h (players_speed_kmh), which must be converted
             with kmh_to_mps() before it meets anything that works in m/s.
  Labels     BallStatus and BallPossession: targets for evaluation, never inputs.

A missing or unexpected label value becomes -1 (unknown) and is counted. It is never defaulted to
"away" or "dead" the way kloppy does. Officials, and any FrameSet belonging to neither team nor
the ball, are skipped and counted. Each section's frame axis is the union of every stream's N range
(ball and players), so no frame is ever dropped; within it, each stream's gaps are reported.
"""

import array
import collections
import xml.etree.ElementTree as ET

import numpy as np

UNKNOWN = -1
STATUS_VALUES = {"0": 0, "1": 1}
POSSESSION_VALUES = {"1": 1, "2": 2}

Meta = collections.namedtuple("Meta", ["match_id", "home_id", "away_id", "names", "pitch",
                                       "officials", "players"])
PITCH_TOLERANCE_M = 0.01
SectionTrack = collections.namedtuple("SectionTrack", ["n", "players_xy", "players_speed_kmh",
                                                       "ball_xy", "ball_z"])

# Physical units of the Frame attributes. "documented": stated by Bassek et al. (2025);
# "measured": confirmed on J03WPY; "inferred": read off J03WPY, undocumented, never used.
# S is km/h -- confirmed on J03WPY (median S / position-derived speed = 3.57 players,
# 3.75 ball). The competition ATD and V1 work in m/s: convert with kmh_to_mps(), never implicitly.
UNITS = {
    "N": {"unit": "frame (25 fps)", "status": "measured", "used": True},
    "T": {"unit": "ISO 8601 timestamp", "status": "documented", "used": True},
    "X": {"unit": "m", "status": "documented", "used": True,
          "note": "origin at the centre spot, x along the pitch length (measured)"},
    "Y": {"unit": "m", "status": "documented", "used": True,
          "note": "origin at the centre spot, bottom to top (kloppy: BOTTOM_TO_TOP)"},
    "Z": {"unit": "m", "status": "documented", "used": True, "note": "ball height"},
    "S": {"unit": "km/h", "status": "documented", "used": True,
          "note": "confirmed on J03WPY against position-derived speed"},
    "D": {"unit": "cm per frame", "status": "inferred", "used": False,
          "note": "about 100 x the per-frame displacement on J03WPY"},
    "A": {"unit": "m/s^2", "status": "inferred", "used": False,
          "note": "within +-8 for players, 0 for the ball on J03WPY"},
    "M": {"unit": "match minute", "status": "inferred", "used": False,
          "note": "1..47 per half on J03WPY"},
}
KMH_PER_MPS = 3.6


def kmh_to_mps(speed_kmh):
    """DFL speed (km/h) to the m/s that V1 and the competition ATD use. NaN stays NaN.

    Floating input keeps its precision (float32 stays float32). Integer or boolean input is
    promoted to float64 first, so the 3.6 factor is never truncated (36 km/h -> 10.0 m/s).
    """
    values = np.asarray(speed_kmh)
    dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.dtype(np.float64)
    out = values.astype(dtype) / dtype.type(KMH_PER_MPS)
    return out if values.ndim else out[()]
DFLTrack = collections.namedtuple("DFLTrack", ["sections", "data", "person_ids", "team_of_person",
                                               "skipped_framesets", "gaps", "frame_counts",
                                               "census", "t_range", "ball_n_range"])
SectionLabels = collections.namedtuple("SectionLabels", ["ball_status", "possession"])
Labels = collections.namedtuple("Labels", ["data", "counts"])


def parse_meta(meta_xml):
    """Teams, players, officials and pitch size from the match-information XML."""
    try:
        root = ET.parse(meta_xml).getroot()
    except ET.ParseError as error:
        raise ValueError(_xml_error(meta_xml, error)) from None
    general = root.find(".//General")
    teams = {team.get("Role", "").lower(): team for team in root.iter("Team")}
    home = teams.get("home")
    away = teams.get("guest", teams.get("away"))
    if home is None or away is None:
        raise ValueError("{}: expected a home and a guest Team, found roles {}".format(
            meta_xml, sorted(teams)))
    home_id = (general.get("HomeTeamId") if general is not None else None) or home.get("TeamId")
    away_id = (general.get("GuestTeamId") if general is not None else None) or away.get("TeamId")
    if home.get("TeamId") != home_id or away.get("TeamId") != away_id:
        raise ValueError("{}: General team ids disagree with the Team elements".format(meta_xml))

    players = {}
    for side, team in ((0, home), (1, away)):
        for player in team.iter("Player"):
            pid = player.get("PersonId")
            if pid in players:
                raise ValueError("{}: PersonId {} is listed {}".format(
                    meta_xml, pid, "twice" if players[pid] == side else "in both teams"))
            players[pid] = side
    environment = root.find(".//Environment")
    if environment is None or environment.get("PitchX") is None:
        raise ValueError("{}: no Environment PitchX/PitchY".format(meta_xml))
    officials = {ref.get("PersonId") for ref in root.iter("Referee")}
    for pid in sorted(officials & set(players)):
        raise ValueError("{}: PersonId {} is listed as a player and as an official".format(
            meta_xml, pid))
    return Meta(match_id=general.get("MatchId") if general is not None else None,
                home_id=home_id, away_id=away_id,
                names=(home.get("TeamName"), away.get("TeamName")),
                pitch=(_pitch_length(environment.get("PitchX"), "PitchX", meta_xml),
                       _pitch_length(environment.get("PitchY"), "PitchY", meta_xml)),
                officials=officials,
                players=players)


def parse(meta_xml, positions_xml):
    """(DFLTrack, Labels, Meta) for one match."""
    meta = parse_meta(meta_xml)
    sections, ball, people = [], {}, collections.defaultdict(dict)
    skipped = collections.Counter()
    census = {"ball_attrs": set(), "player_attrs": set()}
    t_range = {}
    counts = {"ball_status": {"missing": 0, "unexpected": {}},
              "possession": {"missing": 0, "unexpected": {}}}

    try:
        with open(positions_xml, "rb") as handle:
            _read_positions(handle, meta, positions_xml, skipped, sections, ball, people,
                            t_range, census, counts)
    except ET.ParseError as error:
        raise ValueError(_xml_error(positions_xml, error)) from None

    if not ball:
        raise ValueError("{}: no ball FrameSet with frames: nothing to parse".format(positions_xml))
    person_ids = [pid for pid in meta.players if any(pid in people[s] for s in sections)]
    data, label_data, gaps, frame_counts = [], [], {}, {}
    for section in sections:
        if section not in ball:
            raise ValueError("{}: section {} has no ball FrameSet".format(positions_xml, section))
        track, section_labels, section_gaps = _assemble(ball[section], people[section],
                                                        person_ids)
        data.append(track)
        label_data.append(section_labels)
        if section_gaps:
            gaps[section] = section_gaps
        frame_counts[section] = {"timeline": len(track.n), "ball": len(ball[section]["n"]),
                                 "dropped": 0}

    return (DFLTrack(sections=sections, data=data, person_ids=person_ids,
                     team_of_person=np.array([meta.players[p] for p in person_ids], dtype=np.int8),
                     skipped_framesets=dict(skipped), gaps=gaps, frame_counts=frame_counts,
                     census=census, t_range=t_range,
                     # The ball's own first and last N: t_range belongs to these frames, so
                     # frame-rate estimates must pair them, not the union timeline.
                     ball_n_range={section: (int(ball[section]["n"][0]),
                                             int(ball[section]["n"][-1]))
                                   for section in sections}),
            Labels(data=label_data, counts=counts), meta)


# Where each element may appear: the positions file is PutDataRequest > Positions > FrameSet > Frame.
# The real IDSSE files also carry one <MetaData Type="pitch-size"><PitchSize X Y/></MetaData>
# directly under Positions; it is validated against the match information, not ignored.
PARENT = {"PutDataRequest": None, "Positions": "PutDataRequest", "MetaData": "Positions",
          "PitchSize": "MetaData", "FrameSet": "Positions", "Frame": "FrameSet"}


def _read_positions(handle, meta, positions_xml, skipped, sections, ball, people, t_range,
                    census, counts):
    """One streaming pass over the positions file, filling the collectors in place.

    The structure is enforced, not assumed: an unknown root, a namespaced or unexpected element,
    or an element in the wrong place is an error, so a file we do not understand can never come
    back as an empty, plausible-looking result.
    """
    current, stack = None, []
    pitch = {"metadata": 0, "sizes": []}
    census["pitch_size"] = None
    for event, elem in ET.iterparse(handle, events=("start", "end")):
        tag = elem.tag
        if event == "start":
            _check_place(tag, stack, positions_xml)
            stack.append(tag)
            if tag in ("MetaData", "PitchSize"):
                _open_metadata(tag, elem, meta, positions_xml, pitch)
            elif tag == "FrameSet":
                current = _open_frameset(elem, meta, positions_xml, skipped, counts)
                if current is not None and current["section"] not in sections:
                    sections.append(current["section"])
            continue
        stack.pop()
        if tag == "MetaData":
            census["pitch_size"] = _close_metadata(meta, positions_xml, pitch)
        elif tag == "Frame":
            if current is not None:
                _read_frame(current, elem.attrib, census)
            elem.clear()
        elif tag == "FrameSet":
            if current is not None:
                _close_frameset(current, ball, people, t_range)
            current = None
            elem.clear()


def _open_metadata(tag, elem, meta, path, pitch):
    if tag == "MetaData":
        if pitch["metadata"]:
            raise ValueError("{}: more than one MetaData under Positions".format(path))
        pitch["metadata"] += 1
        if elem.get("Type") != "pitch-size":
            raise ValueError("{}: MetaData Type={!r}, expected 'pitch-size'".format(
                path, elem.get("Type")))
        if elem.get("MatchId") is None:
            raise ValueError("{}: MetaData without MatchId".format(path))
        if not meta.match_id:
            raise ValueError("{}: MetaData MatchId {} cannot be checked: the match information "
                             "has no General MatchId".format(path, elem.get("MatchId")))
        if elem.get("MatchId") != meta.match_id:
            raise ValueError("{}: MetaData MatchId {} does not match the match information "
                             "({})".format(path, elem.get("MatchId"), meta.match_id))
        return
    if pitch["sizes"]:
        raise ValueError("{}: more than one PitchSize in MetaData".format(path))
    size = []
    for axis in ("X", "Y"):
        value = elem.get(axis)
        if value is None:
            raise ValueError("{}: PitchSize without {}".format(path, axis))
        try:
            number = float(value)
        except ValueError:
            raise ValueError("{}: PitchSize {}={!r} is not a number".format(
                path, axis, value)) from None
        size.append(_pitch_length(number, "PitchSize " + axis, path))
    pitch["sizes"].append(tuple(size))


def _close_metadata(meta, path, pitch):
    """Cross-check the positions file's own pitch size against the match information."""
    if not pitch["sizes"]:
        raise ValueError("{}: MetaData without a PitchSize".format(path))
    size = pitch["sizes"][0]
    for axis, value, expected, name in (("X", size[0], meta.pitch[0], "PitchX"),
                                        ("Y", size[1], meta.pitch[1], "PitchY")):
        if abs(value - expected) > PITCH_TOLERANCE_M:
            raise ValueError("{}: PitchSize {}={} conflicts with the match information {}={} "
                             "(tolerance {} m)".format(path, axis, value, name, expected,
                                                       PITCH_TOLERANCE_M))
    return size


def _check_place(tag, stack, path):
    if tag.startswith("{"):
        raise ValueError("{}: element <{}> is in namespace {}; expected un-namespaced DFL "
                         "positions XML".format(path, tag.split("}", 1)[1], tag[1:].split("}")[0]))
    parent = stack[-1] if stack else None
    if not stack:
        if tag != "PutDataRequest":
            raise ValueError("{}: root element <{}>, expected <PutDataRequest>".format(path, tag))
        return
    if tag not in PARENT:
        raise ValueError("{}: unexpected element <{}> inside <{}>".format(path, tag, parent))
    if tag == "Frame" and parent != "FrameSet":
        raise ValueError("{}: Frame outside a FrameSet (inside <{}>)".format(path, parent))
    if tag == "FrameSet" and parent == "FrameSet":
        raise ValueError("{}: nested FrameSet inside another FrameSet".format(path))
    if tag == "FrameSet" and parent != "Positions":
        raise ValueError("{}: FrameSet outside <Positions> (inside <{}>)".format(path, parent))
    if PARENT[tag] != parent:
        raise ValueError("{}: unexpected element <{}> inside <{}>".format(path, tag, parent))


def _open_frameset(elem, meta, path, skipped, counts):
    section, team, person = elem.get("GameSection"), elem.get("TeamId"), elem.get("PersonId")
    if not section:
        raise ValueError("{}: FrameSet without GameSection".format(path))
    clubs = (meta.home_id, meta.away_id)
    if (team or "").upper() == "BALL":
        if person in meta.players or person in meta.officials:
            raise ValueError("{}: ball FrameSet carries PersonId {}".format(path, person))
        kind = "ball"
    elif person in meta.players:
        side = meta.players[person]
        if team != clubs[side]:
            raise ValueError("{}: PersonId {} is {} player but its FrameSet has TeamId {}".format(
                path, person, "a home" if side == 0 else "an away", team))
        kind = "player"
    elif person in meta.officials:
        if team in clubs:
            raise ValueError("{}: PersonId {} is an official but its FrameSet has TeamId {}".format(
                path, person, team))
        skipped[team or "?"] += 1
        return None
    elif team in clubs:
        raise ValueError("{}: FrameSet for PersonId {} of team {} is not in the match "
                         "information".format(path, person, team))
    else:
        skipped[team or "?"] += 1
        return None
    # Typed arrays, not lists of Python floats: a real match has ~4 million Frame elements.
    return {"path": path, "section": section, "kind": kind, "person": person, "counts": counts,
            "n": array.array("q"), "x": array.array("d"), "y": array.array("d"),
            "s": array.array("d"), "z": array.array("d"), "status": array.array("b"),
            "possession": array.array("b"), "t_first": None, "t_last": None}


def _read_frame(current, attrib, census):
    where = "{}: {} in {}".format(current["path"], current["person"], current["section"])
    if "N" not in attrib:
        raise ValueError("{}: Frame without N".format(where))
    try:
        n = int(attrib["N"])
    except ValueError:
        raise ValueError("{}: N={!r} is not an integer frame number".format(
            where, attrib["N"])) from None
    census["ball_attrs" if current["kind"] == "ball" else "player_attrs"].update(attrib)
    at = "{} at N={}".format(where, n)
    current["n"].append(n)
    current["x"].append(_float(attrib.get("X"), "X", at))
    current["y"].append(_float(attrib.get("Y"), "Y", at))
    current["s"].append(_float(attrib.get("S"), "S", at))
    if current["kind"] == "ball":
        counts = current["counts"]
        current["z"].append(_float(attrib.get("Z"), "Z", at))
        current["status"].append(_code(attrib.get("BallStatus"), STATUS_VALUES,
                                       counts["ball_status"]))
        current["possession"].append(_code(attrib.get("BallPossession"), POSSESSION_VALUES,
                                           counts["possession"]))
        # Times belong to the smallest and largest N, whatever order the frames arrive in.
        if current["t_first"] is None or n < current["t_first"][0]:
            current["t_first"] = (n, attrib.get("T"))
        if current["t_last"] is None or n > current["t_last"][0]:
            current["t_last"] = (n, attrib.get("T"))


def _close_frameset(current, ball, people, t_range):
    """Freeze one FrameSet's arrays: duplicates are an error, out-of-order frames are sorted."""
    n = np.frombuffer(current["n"], dtype=np.int64)
    fields = [key for key in ("x", "y", "s", "z", "status", "possession") if len(current[key])]
    if len(n) > 1 and not (np.diff(n) > 0).all():
        values, repeats = np.unique(n, return_counts=True)
        if (repeats > 1).any():
            raise ValueError("{}: {} in {}: duplicate frame N={}".format(
                current["path"], current["person"], current["section"],
                int(values[repeats > 1][0])))
        order = np.argsort(n, kind="stable")
        n = n[order]
        for key in fields:
            current[key] = np.asarray(current[key])[order]
    current["n"] = n.copy()
    for key in fields:
        current[key] = np.asarray(current[key])

    section = current["section"]
    if current["kind"] == "ball":
        if not len(n):
            raise ValueError("{}: ball FrameSet with no frames in section {}".format(
                current["path"], section))
        if section in ball:
            raise ValueError("two ball FrameSets in section {}".format(section))
        ball[section] = current
        t_range[section] = (current["t_first"][1], current["t_last"][1])
    else:
        if current["person"] in people[section]:
            raise ValueError("two FrameSets for {} in section {}".format(current["person"], section))
        people[section][current["person"]] = current


def _assemble(ball, people, person_ids):
    streams = [np.asarray(ball["n"], dtype=np.int64)]
    streams += [np.asarray(people[pid]["n"], dtype=np.int64) for pid in person_ids if pid in people]
    n = np.arange(min(int(x.min()) for x in streams if len(x)),
                  max(int(x.max()) for x in streams if len(x)) + 1)
    T, P = len(n), len(person_ids)
    gaps = {}

    index = np.asarray(ball["n"], dtype=np.int64) - n[0]
    ball_xy = np.full((T, 2), np.nan, dtype=np.float32)
    ball_z = np.full(T, np.nan, dtype=np.float32)
    ball_xy[index, 0], ball_xy[index, 1] = ball["x"], ball["y"]
    ball_z[index] = ball["z"]
    _record_gaps(gaps, "ball", n, index)

    status = np.full(T, UNKNOWN, dtype=np.int8)
    possession = np.full(T, UNKNOWN, dtype=np.int8)
    status[index] = ball["status"]
    possession[index] = ball["possession"]

    players_xy = np.full((T, P, 2), np.nan, dtype=np.float32)
    players_speed_kmh = np.full((T, P), np.nan, dtype=np.float32)
    for column, pid in enumerate(person_ids):
        if pid not in people:
            continue
        frames = people[pid]
        rows = np.asarray(frames["n"], dtype=np.int64) - n[0]
        players_xy[rows, column, 0] = frames["x"]
        players_xy[rows, column, 1] = frames["y"]
        players_speed_kmh[rows, column] = frames["s"]
        _record_gaps(gaps, pid, n, rows)

    return (SectionTrack(n=n, players_xy=players_xy, players_speed_kmh=players_speed_kmh,
                         ball_xy=ball_xy, ball_z=ball_z),
            SectionLabels(ball_status=status, possession=possession), gaps)


def _record_gaps(gaps, stream, n, rows):
    present = np.zeros(len(n), dtype=bool)
    present[rows] = True
    spans = [(int(n[a]), int(n[b - 1])) for a, b in _runs(~present)]
    if spans:
        gaps[stream] = spans


def _code(value, allowed, count):
    if value is None:
        count["missing"] += 1
        return UNKNOWN
    if value not in allowed:
        count["unexpected"][value] = count["unexpected"].get(value, 0) + 1
        return UNKNOWN
    return allowed[value]


def _float(value, name, where):
    """A coordinate or speed; NaN when absent; ValueError naming the frame when not a number."""
    if value in (None, ""):
        return np.nan
    try:
        return float(value)
    except ValueError:
        raise ValueError("{}: {}={!r} is not a number".format(where, name, value)) from None


def _number(value, name, path):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("{}: {}={!r} is not a number".format(path, name, value)) from None


def _pitch_length(value, name, path):
    """A pitch dimension: a number that is finite and positive (NaN would pass any comparison)."""
    number = value if isinstance(value, float) else _number(value, name, path)
    if not np.isfinite(number) or number <= 0:
        raise ValueError("{}: {}={} must be a finite positive number".format(path, name, number))
    return number


def _xml_error(path, error):
    line, column = getattr(error, "position", (None, None))
    return "{}: not well-formed XML at line {}, column {} ({})".format(path, line, column, error)


def _runs(mask):
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))
