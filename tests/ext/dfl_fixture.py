"""Synthetic DFL match-information and positions XML, in the shape the open IDSSE files use.

Reconstructed from the paper (Bassek et al. 2025) and kloppy's Sportec deserializer: one FrameSet
per person (or the ball) per GameSection, Frame elements with N, T, X, Y, S, and on the ball also
Z, BallPossession and BallStatus. No real data is copied here.
"""

import datetime
import os
from xml.sax.saxutils import quoteattr

HOME, AWAY, BALL = "DFL-CLU-HOME", "DFL-CLU-AWAY", "BALL"
REFEREE = "DFL-OBJ-REF001"


def write_meta(path, home_players, away_players, pitch=(105.0, 68.0), referees=(REFEREE,),
               half_ms=(4000, 4000)):
    """Match information complete enough for kloppy's Sportec reader as well as ours: coaches,
    player names and line-up flags, the result, half durations and referee names."""
    def players(ids):
        return "".join(
            '<Player PersonId="{p}" ShirtNumber="{n}" Shortname="P{p}" FirstName="F{p}" '
            'LastName="L{p}" Starting="true" PlayingPosition="MZ" />'.format(p=p, n=i + 1)
            for i, p in enumerate(ids))

    def team(team_id, role, name, ids):
        return ('<Team TeamId="{}" Role="{}" TeamName="{}"><TrainerStaff>'
                '<Trainer PersonId="C-{}" Shortname="Coach {}" Role="headcoach" /></TrainerStaff>'
                "<Players>{}</Players></Team>").format(team_id, role, name, role, name,
                                                         players(ids))
    xml = ('<?xml version="1.0" encoding="utf-8"?><PutDataRequest><MatchInformation>'
           '<General MatchId="DFL-MAT-TEST01" HomeTeamId="{h}" GuestTeamId="{a}" '
           'HomeTeamName="Home FC" GuestTeamName="Away FC" Result="1:0" '
           'KickoffTime="2023-01-01T13:30:00.000+01:00" MatchDay="1" />'
           '<Environment PitchX="{L}" PitchY="{W}" />'
           '<OtherGameInformation TotalTimeFirstHalf="{t1}" TotalTimeSecondHalf="{t2}" />'
           "<Teams>{home}{away}</Teams><Referees>{refs}</Referees>"
           "</MatchInformation></PutDataRequest>").format(
        h=HOME, a=AWAY, L=pitch[0], W=pitch[1], t1=half_ms[0], t2=half_ms[1],
        home=team(HOME, "home", "Home FC", home_players),
        away=team(AWAY, "guest", "Away FC", away_players),
        refs="".join('<Referee PersonId="{r}" Shortname="R{r}" FirstName="F" LastName="L" '
                     'Role="referee" />'.format(r=r) for r in referees))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(xml)
    return str(path)


def _t(start, n0, n):
    return (start + datetime.timedelta(seconds=(n - n0) / 25.0)).isoformat(timespec="milliseconds")


def pitch_metadata(x="105.00", y="68.00", match_id="DFL-MAT-TEST01"):
    """The <MetaData Type="pitch-size"> block the real IDSSE positions files carry under
    <Positions>, ahead of the FrameSets."""
    return ('<MetaData MatchId="{}" Type="pitch-size"><PitchSize X="{}" Y="{}" /></MetaData>'
            .format(match_id, x, y))


def write_positions(path, framesets, metadata=None):
    """framesets: [(section, team_id, person_id, [frame_attr_dict, ...])], written in order.
    metadata: raw XML placed under <Positions> before the FrameSets (default: none)."""
    start = datetime.datetime(2023, 1, 1, 13, 30, 0)
    parts = ['<?xml version="1.0" encoding="utf-8"?><PutDataRequest><Positions>',
             metadata or ""]
    for section, team, person, frames in framesets:
        parts.append('<FrameSet GameSection="{}" MatchId="DFL-MAT-TEST01" TeamId="{}" '
                     'PersonId="{}">'.format(section, team, person))
        n0 = frames[0]["N"] if frames else 0
        for frame in frames:
            attrs = dict(frame)
            attrs.setdefault("T", _t(start, n0, attrs["N"]))
            parts.append("<Frame " + " ".join("{}={}".format(k, quoteattr(str(v)))
                                              for k, v in attrs.items()) + " />")
        parts.append("</FrameSet>")
    parts.append("</Positions></PutDataRequest>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(parts))
    return str(path)


def frames(n_start, count, x, y, s=1.0, **extra):
    """count frames from N = n_start; x, y, s may be callables of the offset i."""
    out = []
    for i in range(count):
        frame = {"N": n_start + i,
                 "X": x(i) if callable(x) else x, "Y": y(i) if callable(y) else y,
                 "S": s(i) if callable(s) else s}
        for key, value in extra.items():
            frame[key] = value(i) if callable(value) else value
        out.append(frame)
    return out


def standard_match(dirpath):
    """Two sections, 2 home + 2 away players, a referee, and the ball. Returns (meta, positions)."""
    os.makedirs(str(dirpath), exist_ok=True)
    meta = write_meta(os.path.join(str(dirpath), "meta.xml"), ["H1", "H2"], ["A1", "A2"])
    sets = []
    for section, n0 in (("firstHalf", 10000), ("secondHalf", 100000)):
        sets.append((section, BALL, "DFL-OBJ-BALL", frames(
            n0, 100, x=lambda i: -20.0 + 0.4 * i, y=5.0, Z=0.1,
            BallPossession=lambda i: 1 if i < 60 else 2,
            BallStatus=lambda i: 0 if 40 <= i < 50 else 1)))
        sets.append((section, HOME, "H1", frames(n0, 100, x=lambda i: -20.5 + 0.4 * i, y=5.0)))
        sets.append((section, HOME, "H2", frames(n0, 100, x=-30.0, y=-10.0)))
        sets.append((section, AWAY, "A1", frames(n0, 100, x=10.0, y=5.0)))
        sets.append((section, AWAY, "A2", frames(n0, 100, x=30.0, y=20.0)))
        sets.append((section, "referee", REFEREE, frames(n0, 100, x=0.0, y=0.0)))
    positions = write_positions(os.path.join(str(dirpath), "positions.xml"), sets,
                                metadata=pitch_metadata())
    return meta, positions
