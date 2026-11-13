import os

import numpy as np
import pytest

from pvext import dfl_parse
from tests.ext import dfl_fixture as fx


@pytest.fixture
def parsed(tmp_path):
    return dfl_parse.parse(*fx.standard_match(tmp_path))


def test_meta(parsed):
    _, _, meta = parsed
    assert meta.home_id == fx.HOME and meta.away_id == fx.AWAY
    assert meta.names == ("Home FC", "Away FC")
    assert meta.pitch == (105.0, 68.0)
    assert fx.REFEREE in meta.officials


def test_sections_and_frames(parsed):
    track, labels, _ = parsed
    assert track.sections == ["firstHalf", "secondHalf"]
    first = track.data[0]
    assert first.n.tolist() == list(range(10000, 10100))
    assert first.ball_xy.shape == (100, 2) and first.players_xy.shape == (100, 4, 2)
    assert first.ball_xy[10].tolist() == pytest.approx([-16.0, 5.0])
    assert labels.data[1].ball_status.shape == (100,)


def test_targets_kept_separate(parsed):
    track, labels, _ = parsed
    for name in track.data[0]._fields:
        assert "status" not in name.lower() and "possession" not in name.lower()
    status, possession = labels.data[0].ball_status, labels.data[0].possession
    assert (status[40:50] == 0).all() and (status[:40] == 1).all()
    assert (possession[:60] == 1).all() and (possession[60:] == 2).all()


def test_officials_excluded_and_teams(parsed):
    track, _, _ = parsed
    assert track.person_ids == ["H1", "H2", "A1", "A2"]
    assert track.team_of_person.tolist() == [0, 0, 1, 1]
    assert track.skipped_framesets == {"referee": 2}


@pytest.mark.parametrize("dropped", ["BallStatus", "BallPossession"])
def test_missing_label_is_unknown_not_default(tmp_path, dropped):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    ball = fx.frames(1, 10, x=0.0, y=0.0, Z=0.0, BallPossession=1, BallStatus=1)
    del ball[3][dropped]
    pos = fx.write_positions(str(tmp_path / "p.xml"), [
        ("firstHalf", fx.BALL, "B", ball),
        ("firstHalf", fx.HOME, "H1", fx.frames(1, 10, 1.0, 1.0)),
        ("firstHalf", fx.AWAY, "A1", fx.frames(1, 10, 5.0, 5.0))])
    _, labels, _ = dfl_parse.parse(meta, pos)
    field = "ball_status" if dropped == "BallStatus" else "possession"
    assert getattr(labels.data[0], field)[3] == -1
    assert labels.counts[field]["missing"] == 1


def test_unexpected_label_value_is_unknown_and_counted(tmp_path):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    ball = fx.frames(1, 5, x=0.0, y=0.0, Z=0.0, BallPossession=1, BallStatus=1)
    ball[2]["BallPossession"] = 0
    pos = fx.write_positions(str(tmp_path / "p.xml"), [("firstHalf", fx.BALL, "B", ball),
                                                        ("firstHalf", fx.HOME, "H1", fx.frames(1, 5, 1, 1))])
    _, labels, _ = dfl_parse.parse(meta, pos)
    assert labels.data[0].possession[2] == -1
    assert labels.counts["possession"]["unexpected"] == {"0": 1}


def test_substitute_aligned_by_n(tmp_path):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1", "H9"], ["A1"])
    pos = fx.write_positions(str(tmp_path / "p.xml"), [
        ("firstHalf", fx.BALL, "B", fx.frames(1, 100, 0.0, 0.0, Z=0, BallPossession=1, BallStatus=1)),
        ("firstHalf", fx.HOME, "H1", fx.frames(1, 60, 1.0, 1.0)),
        ("firstHalf", fx.HOME, "H9", fx.frames(51, 50, lambda i: 2.0 + i, 3.0)),
        ("firstHalf", fx.AWAY, "A1", fx.frames(1, 100, 5.0, 5.0))])
    track, _, _ = dfl_parse.parse(meta, pos)
    sub = track.person_ids.index("H9")
    xy = track.data[0].players_xy[:, sub]
    assert np.isnan(xy[:50]).all()
    assert xy[50].tolist() == [2.0, 3.0] and xy[99].tolist() == [51.0, 3.0]
    assert np.isnan(track.data[0].players_xy[60:, track.person_ids.index("H1")]).all()


def test_ball_n_gap_reported(tmp_path):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    ball = [f for f in fx.frames(1, 20, 0.0, 0.0, Z=0, BallPossession=1, BallStatus=1)
            if not 8 <= f["N"] < 11]
    pos = fx.write_positions(str(tmp_path / "p.xml"), [("firstHalf", fx.BALL, "B", ball)])
    track, labels, _ = dfl_parse.parse(meta, pos)
    assert track.data[0].n.tolist() == list(range(1, 21))      # gap kept as frames, missing ball
    assert np.isnan(track.data[0].ball_xy[7:10]).all()
    assert (labels.data[0].ball_status[7:10] == -1).all()
    assert track.gaps == {"firstHalf": {"ball": [(8, 10)]}}


# -- review fix 2: the timeline is the union of all streams; no frame is dropped ---------------

def test_player_frames_outside_ball_range_are_kept(tmp_path):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    pos = fx.write_positions(str(tmp_path / "p.xml"), [
        ("firstHalf", fx.BALL, "B", fx.frames(11, 20, 0.0, 0.0, Z=0, BallPossession=1,
                                                BallStatus=1)),            # N 11..30
        ("firstHalf", fx.HOME, "H1", fx.frames(1, 40, lambda i: float(i), 1.0)),   # N 1..40
        ("firstHalf", fx.AWAY, "A1", fx.frames(5, 10, 5.0, 5.0))])         # N 5..14
    track, labels, _ = dfl_parse.parse(meta, pos)
    section = track.data[0]
    assert section.n.tolist() == list(range(1, 41))
    h1 = track.person_ids.index("H1")
    assert section.players_xy[:, h1, 0].tolist() == [float(i) for i in range(40)]
    assert np.isnan(section.ball_xy[:10]).all() and np.isnan(section.ball_xy[30:]).all()
    assert (labels.data[0].ball_status[:10] == -1).all()
    assert track.gaps["firstHalf"]["ball"] == [(1, 10), (31, 40)]
    assert track.gaps["firstHalf"]["A1"] == [(1, 4), (15, 40)]
    assert "H1" not in track.gaps["firstHalf"]
    assert track.frame_counts["firstHalf"] == {"timeline": 40, "ball": 20, "dropped": 0}


def test_section_without_ball_is_an_error(tmp_path):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    pos = fx.write_positions(str(tmp_path / "p.xml"), [
        ("firstHalf", fx.HOME, "H1", fx.frames(1, 5, 1.0, 1.0))])
    with pytest.raises(ValueError, match="no ball FrameSet"):
        dfl_parse.parse(meta, pos)


def test_duplicate_n_raises(tmp_path):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    ball = fx.frames(1, 5, 0.0, 0.0, Z=0, BallPossession=1, BallStatus=1)
    ball.insert(3, dict(ball[2]))
    pos = fx.write_positions(str(tmp_path / "p.xml"), [("firstHalf", fx.BALL, "B", ball)])
    with pytest.raises(ValueError, match="duplicate frame N=3"):
        dfl_parse.parse(meta, pos)


def test_unknown_person_raises(tmp_path):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    pos = fx.write_positions(str(tmp_path / "p.xml"), [
        ("firstHalf", fx.BALL, "B", fx.frames(1, 5, 0, 0, Z=0, BallPossession=1, BallStatus=1)),
        ("firstHalf", fx.HOME, "STRANGER", fx.frames(1, 5, 0, 0))])
    with pytest.raises(ValueError, match="STRANGER"):
        dfl_parse.parse(meta, pos)


def test_structure_census(parsed):
    track, _, _ = parsed
    assert track.census["ball_attrs"] >= {"N", "T", "X", "Y", "Z", "S", "BallStatus",
                                          "BallPossession"}
    assert track.census["player_attrs"] >= {"N", "T", "X", "Y", "S"}
    assert track.t_range["firstHalf"][0].startswith("2023-01-01T13:30:00")


def test_agrees_with_kloppy_on_fixture(tmp_path):
    """Review fix 5: an independent parser must agree frame by frame. Skips only when kloppy is
    not installed, never because it cannot read the fixture."""
    pytest.importorskip("kloppy")
    from kloppy import sportec
    meta, positions = fx.standard_match(tmp_path)
    # kloppy's default only_alive=True silently drops every dead-ball frame: with it, an
    # alive/dead benchmark would contain no D at all. Pin that, then compare all frames.
    alive_only = sportec.load_tracking(raw_data=positions, meta_data=meta, coordinates="sportec")
    dataset = sportec.load_tracking(raw_data=positions, meta_data=meta, coordinates="sportec",
                                    only_alive=False)
    track, labels, _ = dfl_parse.parse(meta, positions)
    assert len(alive_only.records) == sum(int((lab.ball_status == 1).sum()) for lab in labels.data)

    ours = {}
    for section, data, lab in zip(track.sections, track.data, labels.data):
        for i, n in enumerate(data.n):
            ours[int(n)] = (data, lab, i)
    home_id = dataset.metadata.teams[0].team_id
    assert len(dataset.records) == sum(len(d.n) for d in track.data)
    for record in dataset.records:
        data, lab, i = ours[int(record.frame_id)]
        assert (record.ball_coordinates.x, record.ball_coordinates.y) == pytest.approx(
            tuple(map(float, data.ball_xy[i])), abs=1e-4)
        assert (record.ball_state.value == "alive") == bool(lab.ball_status[i] == 1)
        assert (record.ball_owning_team.team_id == home_id) == bool(lab.possession[i] == 1)
        for player, coordinates in record.players_coordinates.items():
            column = track.person_ids.index(player.player_id)
            assert (coordinates.x, coordinates.y) == pytest.approx(
                tuple(map(float, data.players_xy[i, column])), abs=1e-4)


# -- review fix 3: invalid XML fails loudly, naming file, section, person and frame ------------

GOOD_BALL = '<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="B">' \
            '<Frame N="1" X="0" Y="0" Z="0" S="0" BallStatus="1" BallPossession="1" /></FrameSet>'


def _positions(tmp_path, body):
    path = tmp_path / "p.xml"
    path.write_text('<?xml version="1.0"?><PutDataRequest><Positions>' + body
                    + "</Positions></PutDataRequest>")
    return str(path)


@pytest.mark.parametrize("body, fragment", [
    ('<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="B"><Frame X="0" Y="0" />'
     "</FrameSet>", "without N"),
    ('<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="B"><Frame N="1.5" X="0" Y="0" />'
     "</FrameSet>", "N='1.5'"),
    ('<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="B"><Frame N="1" X="abc" Y="0" />'
     "</FrameSet>", "X='abc'"),
    ('<Frame N="1" X="0" Y="0" />' + GOOD_BALL, "outside a FrameSet"),
    ('<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="B">' + GOOD_BALL + "</FrameSet>",
     "nested FrameSet"),
    ('<FrameSet TeamId="BALL" PersonId="B"><Frame N="1" X="0" Y="0" /></FrameSet>',
     "without GameSection"),
])
def test_invalid_positions_content(tmp_path, body, fragment):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    with pytest.raises(ValueError, match=fragment) as error:
        dfl_parse.parse(meta, _positions(tmp_path, body))
    assert "p.xml" in str(error.value)


def test_frame_error_names_person_and_frame(tmp_path):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    body = GOOD_BALL + ('<FrameSet GameSection="firstHalf" TeamId="{}" PersonId="H1">'
                        '<Frame N="1" X="0" Y="0" /><Frame N="2" X="0" Y="NaNx" /></FrameSet>'
                        ).format(fx.HOME)
    with pytest.raises(ValueError, match=r"H1.*firstHalf.*N=2.*Y='NaNx'"):
        dfl_parse.parse(meta, _positions(tmp_path, body))


@pytest.mark.parametrize("content", ["", "<PutDataRequest><Positions><FrameSet",
                                      "not xml at all"])
def test_malformed_positions_xml(tmp_path, content):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    path = tmp_path / "p.xml"
    path.write_text(content)
    with pytest.raises(ValueError, match=r"p\.xml.*not well-formed XML|p\.xml.*no element"):
        dfl_parse.parse(meta, str(path))


def test_truncated_real_shape_file(tmp_path):
    meta, positions = fx.standard_match(tmp_path)
    with open(positions, "rb") as handle:
        data = handle.read()
    cut = tmp_path / "cut.xml"
    cut.write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError, match=r"cut\.xml.*line \d+"):
        dfl_parse.parse(meta, str(cut))


@pytest.mark.parametrize("mutate, fragment", [
    (lambda s: s.replace('PitchX="105.0"', 'PitchX="wide"'), "PitchX"),
    (lambda s: s[: len(s) // 2], "not well-formed XML"),
    (lambda s: s.replace('Role="guest"', 'Role="visitor"'), "home and a guest"),
])
def test_invalid_meta(tmp_path, mutate, fragment):
    meta, positions = fx.standard_match(tmp_path)
    with open(meta, encoding="utf-8") as handle:
        text = handle.read()
    with open(meta, "w", encoding="utf-8") as handle:
        handle.write(mutate(text))
    with pytest.raises(ValueError, match=fragment):
        dfl_parse.parse(meta, positions)


# -- review fix 4: PersonId and TeamId must agree, and every person has exactly one role -------

def _one_frameset(tmp_path, team, person, home=("H1",), away=("A1",), referees=(fx.REFEREE,)):
    meta = fx.write_meta(str(tmp_path / "m.xml"), list(home), list(away), referees=referees)
    pos = fx.write_positions(str(tmp_path / "p.xml"), [
        ("firstHalf", fx.BALL, "B", fx.frames(1, 5, 0, 0, Z=0, BallPossession=1, BallStatus=1)),
        ("firstHalf", team, person, fx.frames(1, 5, 1.0, 1.0))])
    return meta, pos


@pytest.mark.parametrize("team, person, fragment", [
    (fx.AWAY, "H1", "H1 is a home player but its FrameSet has TeamId " + fx.AWAY),
    (fx.HOME, "A1", "A1 is an away player but its FrameSet has TeamId " + fx.HOME),
    ("referee", "H1", "H1 is a home player but its FrameSet has TeamId referee"),
    (fx.HOME, fx.REFEREE, fx.REFEREE + " is an official but its FrameSet has TeamId " + fx.HOME),
    (fx.BALL, "H1", "ball FrameSet carries PersonId H1"),
])
def test_team_person_mismatch_raises(tmp_path, team, person, fragment):
    with pytest.raises(ValueError, match=fragment):
        dfl_parse.parse(*_one_frameset(tmp_path, team, person))


def test_official_under_non_club_team_is_skipped(tmp_path):
    track, _, _ = dfl_parse.parse(*_one_frameset(tmp_path, "referee", fx.REFEREE))
    assert track.skipped_framesets == {"referee": 1} and track.person_ids == []


@pytest.mark.parametrize("home, away, referees, fragment", [
    (("H1", "X"), ("A1", "X"), (fx.REFEREE,), "X is listed in both teams"),
    (("H1",), ("A1",), ("H1",), "H1 is listed as a player and as an official"),
    (("H1", "H1"), ("A1",), (fx.REFEREE,), "H1 is listed twice"),
])
def test_meta_roles_must_be_unique(tmp_path, home, away, referees, fragment):
    with pytest.raises(ValueError, match=fragment):
        dfl_parse.parse(*_one_frameset(tmp_path, fx.HOME, "H1", home, away, referees))


def test_unknown_non_club_frameset_is_counted(tmp_path):
    track, _, _ = dfl_parse.parse(*_one_frameset(tmp_path, "DFL-CLU-OTHER", "Z9"))
    assert track.skipped_framesets == {"DFL-CLU-OTHER": 1}


# -- memory: the collectors must stay compact (a real match has ~4 million Frame elements) -----

MEMORY_PROBE = r"""
import sys, tracemalloc
sys.path.insert(0, sys.argv[1])
from pvext import dfl_parse
tracemalloc.start()
track, labels, meta = dfl_parse.parse(sys.argv[2], sys.argv[3])
current, peak = tracemalloc.get_traced_memory()
frames = sum(len(d.n) for d in track.data) * (1 + len(track.person_ids))
dense = sum(d.players_xy.nbytes + d.players_speed_kmh.nbytes + d.ball_xy.nbytes + d.ball_z.nbytes
            for d in track.data) + sum(l.ball_status.nbytes + l.possession.nbytes
                                       for l in labels.data)
print(frames, peak, dense)
"""


def test_parser_memory_per_frame_is_bounded(tmp_path):
    """Peak traced memory beyond the dense output arrays must stay under 64 bytes per Frame
    element. Measured in a fresh interpreter so other tests cannot distort it."""
    import subprocess
    import sys
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H{}".format(i) for i in range(11)],
                         ["A{}".format(i) for i in range(11)])
    sets = []
    for section, n0 in (("firstHalf", 10000), ("secondHalf", 100000)):
        sets.append((section, fx.BALL, "B", fx.frames(n0, 3000, lambda i: i * 0.01, 1.0, Z=0.1,
                                                         BallPossession=1, BallStatus=1)))
        for team, prefix in ((fx.HOME, "H"), (fx.AWAY, "A")):
            for k in range(11):
                sets.append((section, team, "{}{}".format(prefix, k),
                             fx.frames(n0, 3000, float(k), 2.0)))
    positions = fx.write_positions(str(tmp_path / "p.xml"), sets)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = subprocess.run([sys.executable, "-c", MEMORY_PROBE, root, meta, positions],
                         capture_output=True, text=True, check=True).stdout.split()
    frames, peak, dense = map(int, out)
    assert frames == 2 * 3000 * 23
    assert (peak - dense) / frames < 64, "{:.0f} bytes per frame".format((peak - dense) / frames)


def test_out_of_order_frames_are_sorted(tmp_path):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    ball = fx.frames(1, 6, lambda i: float(i), 0.0, Z=0, BallPossession=1,
                     BallStatus=lambda i: i % 2)
    ball = [ball[i] for i in (0, 2, 1, 3, 5, 4)]
    pos = fx.write_positions(str(tmp_path / "p.xml"), [("firstHalf", fx.BALL, "B", ball)])
    track, labels, _ = dfl_parse.parse(meta, pos)
    assert track.data[0].ball_xy[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert labels.data[0].ball_status.tolist() == [0, 1, 0, 1, 0, 1]


# -- review 2, fix 1: well-formed but structurally invalid positions are rejected ---------------

PLAYER_ONLY = ('<FrameSet GameSection="firstHalf" TeamId="{}" PersonId="H1">'
               '<Frame N="1" X="0" Y="0" S="0" /></FrameSet>').format(fx.HOME)


@pytest.mark.parametrize("document, fragment", [
    ('<?xml version="1.0"?><PutDataRequest><Positions /></PutDataRequest>',
     "no ball FrameSet with frames"),
    ('<?xml version="1.0"?><PutDataRequest><Positions></Positions></PutDataRequest>',
     "no ball FrameSet with frames"),
    ('<?xml version="1.0"?><PutDataRequest xmlns="urn:other"><Positions>' + GOOD_BALL
     + "</Positions></PutDataRequest>", "namespace urn:other"),
    ('<?xml version="1.0"?><x:PutDataRequest xmlns:x="urn:x"><x:Positions>' + GOOD_BALL
     + "</x:Positions></x:PutDataRequest>", "namespace urn:x"),
    ('<?xml version="1.0"?><Something>' + GOOD_BALL + "</Something>",
     "root element <Something>"),
    ('<?xml version="1.0"?><PutDataRequest>' + GOOD_BALL + "</PutDataRequest>",
     "FrameSet outside <Positions>"),
    ('<?xml version="1.0"?><PutDataRequest><Positions><Frames>' + GOOD_BALL
     + "</Frames></Positions></PutDataRequest>", "unexpected element <Frames>"),
    ('<?xml version="1.0"?><PutDataRequest><Positions>' + PLAYER_ONLY
     + "</Positions></PutDataRequest>", "no ball FrameSet"),
    ('<?xml version="1.0"?><PutDataRequest><Positions>'
     '<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="B"></FrameSet>'
     "</Positions></PutDataRequest>", "ball FrameSet with no frames"),
])
def test_structurally_invalid_positions_rejected(tmp_path, document, fragment):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    path = tmp_path / "p.xml"
    path.write_text(document)
    with pytest.raises(ValueError, match=fragment) as error:
        dfl_parse.parse(meta, str(path))
    assert "p.xml" in str(error.value)


def test_minimal_valid_positions_accepted(tmp_path):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    track, labels, _ = dfl_parse.parse(meta, _positions(tmp_path, GOOD_BALL))
    assert track.sections == ["firstHalf"] and len(track.data[0].n) == 1


@pytest.mark.parametrize("order", [(0, 1, 2, 3, 4), (4, 3, 2, 1, 0), (2, 0, 4, 1, 3)])
def test_t_range_follows_frame_numbers_not_file_order(tmp_path, order):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    ball = fx.frames(10000, 5, 0.0, 0.0, Z=0, BallPossession=1, BallStatus=1,
                     T=lambda i: "2023-01-01T13:30:00.{:03d}".format(40 * i))
    pos = fx.write_positions(str(tmp_path / "p.xml"),
                             [("firstHalf", fx.BALL, "B", [ball[i] for i in order])])
    track, _, _ = dfl_parse.parse(meta, pos)
    assert track.t_range["firstHalf"] == ("2023-01-01T13:30:00.000", "2023-01-01T13:30:00.160")


# -- T3 finding: the real files carry <Positions><MetaData Type="pitch-size"><PitchSize/> ------

def _with_metadata(tmp_path, metadata, body=None):
    path = tmp_path / "p.xml"
    path.write_text('<?xml version="1.0"?><PutDataRequest><Positions>' + metadata
                    + (GOOD_BALL if body is None else body) + "</Positions></PutDataRequest>")
    return str(path)


@pytest.mark.parametrize("x, y", [("105.00", "68.00"), ("105.004", "67.995"), ("105", "68")])
def test_pitch_metadata_accepted_and_recorded(tmp_path, x, y):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    track, labels, _ = dfl_parse.parse(meta, _with_metadata(tmp_path, fx.pitch_metadata(x, y)))
    assert track.census["pitch_size"] == (float(x), float(y))
    for name in labels.data[0]._fields:              # metadata never reaches the targets
        assert "pitch" not in name


def test_pitch_metadata_absent_is_recorded_not_assumed(tmp_path):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    track, _, _ = dfl_parse.parse(meta, _positions(tmp_path, GOOD_BALL))
    assert track.census["pitch_size"] is None


def test_standard_fixture_carries_metadata(tmp_path):
    track, _, _ = dfl_parse.parse(*fx.standard_match(tmp_path))
    assert track.census["pitch_size"] == (105.0, 68.0)


@pytest.mark.parametrize("metadata, fragment", [
    (fx.pitch_metadata(x="104.00"), r"PitchSize X=104\.0 .*PitchX=105\.0"),
    (fx.pitch_metadata(y="68.02"), r"PitchSize Y=68\.02 .*PitchY=68\.0"),
    (fx.pitch_metadata(match_id="DFL-MAT-OTHER"), "MatchId DFL-MAT-OTHER"),
    (fx.pitch_metadata(x="wide"), "PitchSize X='wide' is not a number"),
    ('<MetaData MatchId="DFL-MAT-TEST01" Type="pitch-size"><PitchSize X="105" /></MetaData>',
     "PitchSize without Y"),
    ('<MetaData MatchId="DFL-MAT-TEST01" Type="pitch-size" />', "MetaData without a PitchSize"),
    ('<MetaData MatchId="DFL-MAT-TEST01" Type="pitch-size"><PitchSize X="105" Y="68" />'
     '<PitchSize X="105" Y="68" /></MetaData>', "more than one PitchSize"),
    ('<MetaData MatchId="DFL-MAT-TEST01" Type="weather"><PitchSize X="105" Y="68" /></MetaData>',
     "MetaData Type='weather'"),
    ('<MetaData MatchId="DFL-MAT-TEST01"><PitchSize X="105" Y="68" /></MetaData>',
     "MetaData Type=None"),
    ('<MetaData MatchId="DFL-MAT-TEST01" Type="pitch-size"><PitchSize X="105" Y="68" />'
     "<Other /></MetaData>", "unexpected element <Other> inside <MetaData>"),
    ('<MetaData MatchId="DFL-MAT-TEST01" Type="pitch-size"><PitchSize X="105" Y="68">'
     "<Inner /></PitchSize></MetaData>", "unexpected element <Inner> inside <PitchSize>"),
    (fx.pitch_metadata() + fx.pitch_metadata(), "more than one MetaData"),
    ('<PitchSize X="105" Y="68" />', "unexpected element <PitchSize> inside <Positions>"),
])
def test_invalid_pitch_metadata_rejected(tmp_path, metadata, fragment):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    with pytest.raises(ValueError, match=fragment) as error:
        dfl_parse.parse(meta, _with_metadata(tmp_path, metadata))
    assert "p.xml" in str(error.value)


@pytest.mark.parametrize("document, fragment", [
    ('<?xml version="1.0"?><PutDataRequest>' + fx.pitch_metadata() + "<Positions>" + GOOD_BALL
     + "</Positions></PutDataRequest>", "unexpected element <MetaData> inside <PutDataRequest>"),
    ('<?xml version="1.0"?><PutDataRequest><Positions>'
     '<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="B">' + fx.pitch_metadata()
     + '<Frame N="1" X="0" Y="0" Z="0" S="0" BallStatus="1" BallPossession="1" /></FrameSet>'
     "</Positions></PutDataRequest>", "unexpected element <MetaData> inside <FrameSet>"),
])
def test_metadata_in_wrong_place_rejected(tmp_path, document, fragment):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    path = tmp_path / "p.xml"
    path.write_text(document)
    with pytest.raises(ValueError, match=fragment):
        dfl_parse.parse(meta, str(path))


# -- units: DFL S is km/h (Bassek et al.; confirmed on J03WPY against position-derived speed) ---

def test_speed_field_names_its_unit(parsed):
    track, _, _ = parsed
    section = track.data[0]
    assert "players_speed_kmh" in section._fields
    assert "players_speed" not in section._fields       # no unit-less speed to misread as m/s
    assert section.players_speed_kmh[0, 0] == pytest.approx(1.0)   # fixture S="1.0", raw km/h


def test_units_table_documents_every_used_field():
    units = dfl_parse.UNITS
    assert units["X"]["unit"] == "m" and units["Y"]["unit"] == "m" and units["Z"]["unit"] == "m"
    assert units["S"]["unit"] == "km/h" and units["S"]["status"] == "documented"
    assert units["N"]["unit"] == "frame (25 fps)"
    for unverified in ("D", "A", "M"):
        assert units[unverified]["status"] == "inferred" and units[unverified]["used"] is False


@pytest.mark.parametrize("kmh, mps", [(0.0, 0.0), (3.6, 1.0), (36.0, 10.0), (125.28, 34.8)])
def test_kmh_to_mps(kmh, mps):
    assert dfl_parse.kmh_to_mps(kmh) == pytest.approx(mps)


def test_kmh_to_mps_keeps_nan_and_arrays():
    out = dfl_parse.kmh_to_mps(np.array([7.2, np.nan], dtype=np.float32))
    assert out.dtype == np.float32 and out[0] == pytest.approx(2.0) and np.isnan(out[1])


# -- PR #2 review 3, fix 1: kmh_to_mps must not truncate the factor for integer input ----------

@pytest.mark.parametrize("kmh, expected", [
    (36, 10.0), (0, 0.0), (125, 125 / 3.6), (np.int32(36), 10.0), (np.int64(72), 20.0),
    (True, 1 / 3.6),
])
def test_kmh_to_mps_integer_scalars(kmh, expected):
    out = dfl_parse.kmh_to_mps(kmh)
    assert out == pytest.approx(expected)
    assert np.issubdtype(np.asarray(out).dtype, np.floating)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8])
def test_kmh_to_mps_integer_arrays(dtype):
    out = dfl_parse.kmh_to_mps(np.array([0, 36, 72], dtype=dtype))
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [0.0, 10.0, 20.0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmh_to_mps_keeps_float_precision(dtype):
    out = dfl_parse.kmh_to_mps(np.array([36.0, np.nan], dtype=dtype))
    assert out.dtype == dtype and out[0] == pytest.approx(10.0) and np.isnan(out[1])


# -- PR #2 review 3, fix 2: present pitch metadata is validated strictly --------------------------

@pytest.mark.parametrize("x, y, fragment", [
    ("NaN", "68.00", r"PitchSize X=nan must be a finite positive number"),
    ("105.00", "nan", r"PitchSize Y=nan must be a finite positive number"),
    ("inf", "68.00", r"PitchSize X=inf must be a finite positive number"),
    ("105.00", "-inf", r"PitchSize Y=-inf must be a finite positive number"),
    ("0", "68.00", r"PitchSize X=0.0 must be a finite positive number"),
    ("-105.00", "68.00", r"PitchSize X=-105.0 must be a finite positive number"),
])
def test_pitch_metadata_values_must_be_finite_positive(tmp_path, x, y, fragment):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    with pytest.raises(ValueError, match=fragment):
        dfl_parse.parse(meta, _with_metadata(tmp_path, fx.pitch_metadata(x, y)))


def test_pitch_metadata_requires_match_id(tmp_path):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    metadata = '<MetaData Type="pitch-size"><PitchSize X="105" Y="68" /></MetaData>'
    with pytest.raises(ValueError, match="MetaData without MatchId"):
        dfl_parse.parse(meta, _with_metadata(tmp_path, metadata))


def test_pitch_metadata_needs_match_information_match_id(tmp_path):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    with open(meta, encoding="utf-8") as handle:
        text = handle.read()
    with open(meta, "w", encoding="utf-8") as handle:
        handle.write(text.replace('MatchId="DFL-MAT-TEST01" ', "", 1))
    with pytest.raises(ValueError, match="no General MatchId"):
        dfl_parse.parse(meta, _with_metadata(tmp_path, fx.pitch_metadata()))


@pytest.mark.parametrize("pitch_x, fragment", [
    ("NaN", "PitchX=nan must be a finite positive number"),
    ("inf", "PitchX=inf must be a finite positive number"),
    ("0", "PitchX=0.0 must be a finite positive number"),
    ("-105.0", "PitchX=-105.0 must be a finite positive number"),
])
def test_match_information_pitch_must_be_finite_positive(tmp_path, pitch_x, fragment):
    meta, positions = fx.standard_match(tmp_path)
    with open(meta, encoding="utf-8") as handle:
        text = handle.read()
    with open(meta, "w", encoding="utf-8") as handle:
        handle.write(text.replace('PitchX="105.0"', 'PitchX="{}"'.format(pitch_x)))
    with pytest.raises(ValueError, match=fragment):
        dfl_parse.parse(meta, positions)


def test_ball_n_range_recorded_per_section(tmp_path):
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    pos = fx.write_positions(str(tmp_path / "p.xml"), [
        ("firstHalf", fx.BALL, "B", fx.frames(11, 20, 0.0, 0.0, Z=0, BallPossession=1,
                                                BallStatus=1)),
        ("firstHalf", fx.HOME, "H1", fx.frames(1, 40, 1.0, 1.0))])
    track, _, _ = dfl_parse.parse(meta, pos)
    assert track.ball_n_range == {"firstHalf": (11, 30)}
    assert track.data[0].n[0] == 1 and track.data[0].n[-1] == 40
