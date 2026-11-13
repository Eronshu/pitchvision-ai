import numpy as np
import pytest

from pitchvision import tagging
from starter import load_smart_tagging
from tests.synthetic import write_smart_tagging

ATD_NAMES = ("Barcelona", "Atletic Club")


def event(code, start, end, team=None):
    tags = {} if team is None else {"Team": team}
    return {"code": code, "start_frame": start, "end_frame": end, "tags": tags}


def test_team_map_casefold_whitespace_and_na():
    tmap = tagging.team_map({"Barcelona", "atletic  club", "N/A"}, ATD_NAMES)
    assert tmap == {"Barcelona": 0, "atletic  club": 1, "N/A": -1}


@pytest.mark.parametrize("bad", ["Athletic Club", "Real Madrid", ""])
def test_team_map_unknown_name_raises(bad):
    with pytest.raises(ValueError, match="Smart Tagging team"):
        tagging.team_map({"Barcelona", bad}, ATD_NAMES)


def test_team_map_ignores_none():
    assert tagging.team_map({"Barcelona", None}, ATD_NAMES) == {"Barcelona": 0}


def test_setpiece_dead_marks_interval_exactly():
    events = [event("SET PIECES", 10, 20, "Barcelona"), event("BUILD UP", 30, 40, "Barcelona"),
              event("set pieces", 45, 48)]
    cue = tagging.setpiece_dead(events, 50)
    assert cue.dtype == np.float32
    assert np.flatnonzero(cue).tolist() == list(range(10, 20)) + [45, 46, 47]


def test_setpiece_dead_clips_to_recording():
    cue = tagging.setpiece_dead([event("SET PIECES", -5, 3), event("SET PIECES", 48, 60)], 50)
    assert np.flatnonzero(cue).tolist() == [0, 1, 2, 48, 49]


def test_restart_team_ignores_na_and_untagged():
    tmap = tagging.team_map({"Barcelona", "Atletic Club", "N/A"}, ATD_NAMES)
    events = [event("SET PIECES", 0, 5, "Atletic Club"), event("SET PIECES", 10, 15, "N/A"),
              event("SET PIECES", 20, 25), event("PROGRESSION", 30, 35, "Barcelona"),
              event("SET PIECES", 40, 45, "Barcelona")]
    team = tagging.restart_team(events, 50, tmap)
    assert team.dtype == np.int8
    assert (team[0:5] == 1).all() and (team[40:45] == 0).all()
    assert (np.delete(team, list(range(0, 5)) + list(range(40, 45))) == -1).all()


def test_events_from_official_loader(tmp_path):
    """The synthetic XML goes through starter/load_smart_tagging.load unchanged."""
    write_smart_tagging(tmp_path / "tactical" / "BAR-ATH_tactical_smart_tagging.xml",
                        [("SET PIECES", 4.0, 6.0, {"Team": "N/A", "Type": "Throw-in"}),
                         ("SET PIECES", 8.0, 9.0, {"Team": "Barcelona"})])
    events = load_smart_tagging.load(data_dir=str(tmp_path))
    tmap = tagging.team_map(tagging.tag_teams(events), ATD_NAMES)
    assert tmap == {"N/A": -1, "Barcelona": 0}
    assert np.flatnonzero(tagging.setpiece_dead(events, 300))[[0, -1]].tolist() == [100, 224]
    assert (tagging.restart_team(events, 300, tmap)[200:225] == 0).all()
    assert (tagging.restart_team(events, 300, tmap)[100:150] == -1).all()


# -- pipeline effects: each flag on its own, and nothing changes with both off -----------------

from pitchvision import config, pipeline  # noqa: E402
from tests.synthetic import make_tracking  # noqa: E402

FPS = 25
PITCH = (105.3, 68.0)
BLOCK = [(1, 0, 399)]
TEAMS = ("Barcelona", "Atletic Club")


def scripted_tracking(segments, n=400):
    """segments: [(start, stop, ball_fn(t) -> (x, y) | None, holder_team | None, field_speed)].
    Slot 0 is home, slot 1 away; the holder stands 0.5 m behind the ball, the others are far."""
    players = np.full((n, 4, 2), np.nan)
    ball = np.full((n, 2), np.nan)
    speed = np.full((n, 4), 3.0)
    for start, stop, ball_fn, holder, field in segments:
        for t in range(start, stop):
            b = ball_fn(t)
            anchor = b if b is not None else (50.0, 34.0)
            players[t] = [(anchor[0] + 20, 60.0), (anchor[0] - 20, 60.0),
                          (anchor[0] + 30, 8.0), (anchor[0] - 30, 8.0)]
            if holder is not None:
                players[t, holder] = (anchor[0] - 0.5, anchor[1])
            if b is not None:
                ball[t] = b
            speed[t] = field
    return make_tracking(players, ball, [0, 1, 0, 1], speed)


def roll(x0, y, t0):
    return lambda t: (x0 + 0.2 * (t - t0), y)


def tagged(cfg_tagging, tr, events):
    cfg = config.from_dict({"tagging": cfg_tagging})
    return pipeline.run(tr, cfg, FPS, PITCH, BLOCK, events=events, team_names=TEAMS)


@pytest.fixture
def restart_scene():
    """H dribbles; ball sits out of play 4 s; a restart rolls loose for 2.4 s, then H controls.
    Smart Tagging says Atletic Club took the set piece."""
    tr = scripted_tracking([
        (0, 100, roll(30.0, 34.0, 0), 0, 3.0),
        (100, 200, lambda t: (50.0, -3.0), None, 2.0),
        (200, 260, roll(50.0, 0.5, 200), None, 3.0),
        (260, 400, roll(62.0, 0.5, 260), 0, 3.0),
    ])
    events = [event("SET PIECES", 190, 215, "Atletic Club")]
    return tr, events


@pytest.fixture
def setpiece_scene():
    """A free kick: ball still 5 s, nobody within reach, field ambling at 1.3 m/s -- below the
    soft threshold on tracking alone. Smart Tagging marks the set piece."""
    tr = scripted_tracking([
        (0, 100, roll(30.0, 34.0, 0), 0, 3.0),
        (100, 225, lambda t: (50.0, 34.0), None, 1.3),
        (225, 400, roll(50.0, 34.0, 224), 0, 3.0),
    ])
    events = [event("SET PIECES", 100, 225, "Barcelona")]
    return tr, events


def test_both_flags_off_ignores_events(restart_scene):
    tr, events = restart_scene
    with_events = tagged({}, tr, events)
    without = pipeline.run(tr, config.defaults(), FPS, PITCH, BLOCK)
    assert np.array_equal(with_events, without)


def test_setpiece_team_seeds_the_restart(restart_scene):
    tr, events = restart_scene
    off = tagged({}, tr, events)
    on = tagged({"use_setpiece_team": True}, tr, events)
    assert set(off[200:258]) == {"H"}            # UNKNOWN back-filled from H's later control
    assert set(on[200:258]) == {"A"}             # seeded by the tag ...
    assert set(on[270:]) == {"H"}                # ... and overturned by clear H control
    assert np.array_equal(off[:200], on[:200])


def test_setpiece_dead_adds_a_stoppage(setpiece_scene):
    tr, events = setpiece_scene
    off = tagged({}, tr, events)
    on = tagged({"use_setpiece_dead": True}, tr, events)
    assert "D" not in set(off[100:225])
    assert set(on[115:220]) == {"D"}


def test_flags_are_independent(restart_scene, setpiece_scene):
    tr, events = restart_scene
    assert set(tagged({"use_setpiece_dead": True}, tr, events)[200:258]) == {"H"}
    tr, events = setpiece_scene
    assert "D" not in set(tagged({"use_setpiece_team": True}, tr, events)[100:225])


@pytest.mark.parametrize("events, names, fragment", [
    (None, TEAMS, "events"), ([], None, "team names")])
def test_flag_on_needs_inputs(restart_scene, events, names, fragment):
    tr, _ = restart_scene
    cfg = config.from_dict({"tagging": {"use_setpiece_team": True}})
    with pytest.raises(ValueError, match=fragment):
        pipeline.run(tr, cfg, FPS, PITCH, BLOCK, events=events, team_names=names)


def test_unknown_tag_team_fails_loudly(restart_scene):
    tr, _ = restart_scene
    with pytest.raises(ValueError, match="Smart Tagging team"):
        tagged({"use_setpiece_team": True}, tr, [event("SET PIECES", 190, 215, "Athletic")])


# -- L2: consistency with Smart Tagging, and whether it is independent evidence ----------------

from pitchvision import diagnostics  # noqa: E402

TMAP = {"Barcelona": 0, "Atletic Club": 1, "N/A": -1}
L2_BLOCK = [(1, 0, 299)]
# H, stoppage, A restarts at 150 ... stoppage, H restarts at 260.
L2_LABELS = np.array(list("H" * 100 + "D" * 50 + "A" * 80 + "D" * 30 + "H" * 40), dtype="<U1")


def l2(events, **flags):
    cfg = config.from_dict({"tagging": flags})
    return diagnostics.setpiece_consistency(L2_LABELS, events, cfg, L2_BLOCK, TMAP)


def test_l2_restart_share_and_team_agreement():
    report = l2([event("SET PIECES", 140, 160, "Atletic Club"),
                 event("SET PIECES", 250, 270, "Atletic Club"),
                 event("SET PIECES", 20, 40, "Barcelona")])   # no restart inside: missed
    assert report["n_setpiece_tags"] == 3
    assert report["restart_timing"]["share_tags_with_restart"] == pytest.approx(2 / 3)
    assert report["restart_team"]["n_compared"] == 2
    assert report["restart_team"]["agreement"] == pytest.approx(0.5)


def test_l2_na_tags_not_compared_and_outside_blocks_ignored():
    report = l2([event("SET PIECES", 140, 160, "N/A"), event("SET PIECES", 400, 420, "N/A"),
                 event("BUILD UP", 140, 160, "Barcelona")])
    assert report["n_setpiece_tags"] == 1
    assert report["restart_timing"]["share_tags_with_restart"] == 1.0
    assert report["restart_team"] == {"agreement": None, "n_compared": 0, "independent": True}


@pytest.mark.parametrize("flags, independent", [
    ({}, True),
    ({"use_setpiece_dead": True}, False),
    ({"use_setpiece_team": True}, False),
    ({"use_setpiece_dead": True, "use_setpiece_team": True}, False),
])
def test_l2_any_tagging_input_makes_both_figures_non_independent(flags, independent):
    """Either cue can move either figure (see the cross-effect tests below), so any enabled
    tagging input marks both as not independent."""
    report = l2([event("SET PIECES", 140, 160, "Atletic Club")], **flags)
    assert report["restart_timing"]["independent"] is independent
    assert report["restart_team"]["independent"] is independent


def test_l2_interval_overlap_with_blocks():
    """Tags crossing a block boundary count; restarts outside graded blocks never do."""
    blocks = [(1, 100, 299)]
    labels = np.array(list("D" * 120 + "H" * 80 + "D" * 50 + "A" * 50), dtype="<U1")
    # restarts inside the block: D->H at 120, D->A at 250. Frame 100 is the block start.
    events = [event("SET PIECES", 90, 130, "Barcelona"),     # starts before the block: counts
              event("SET PIECES", 280, 320, "Barcelona"),    # crosses the block end, no restart
              event("SET PIECES", 0, 50, "Barcelona"),       # entirely outside: ignored
              event("SET PIECES", 300, 330, "Barcelona")]    # touches only frame 300: outside
    cfg = config.defaults()
    labels_ext = np.concatenate([labels, np.array(list("D" * 10 + "H" * 40), dtype="<U1")])
    report = diagnostics.setpiece_consistency(labels_ext, events, cfg, blocks, TMAP)
    assert report["n_setpiece_tags"] == 2
    assert report["restart_timing"]["share_tags_with_restart"] == pytest.approx(0.5)
    assert report["restart_team"] == {"agreement": 1.0, "n_compared": 1, "independent": True}


# -- cross-effects: each cue can move the L2 figure that looks like it belongs to the other ----

def test_cross_effect_dead_cue_changes_restart_team(setpiece_scene):
    """The dead cue creates a stoppage, and with it a restart that restart_team then scores."""
    tr, _ = setpiece_scene
    events = [event("SET PIECES", 100, 240, "Barcelona")]
    reports = {}
    for name, flags in (("off", {}), ("on", {"use_setpiece_dead": True})):
        labels = tagged(flags, tr, events)
        cfg = config.from_dict({"tagging": flags})
        reports[name] = diagnostics.setpiece_consistency(labels, events, cfg, BLOCK, TMAP)
    assert reports["off"]["restart_team"]["n_compared"] == 0
    assert reports["on"]["restart_team"]["n_compared"] == 1
    assert reports["on"]["restart_team"]["independent"] is False


def test_cross_effect_team_cue_changes_restart_timing():
    """With nobody tracked after a restart, the stretch is X, and the scorer's forward fill
    turns it into D: no restart. Seeding the tagged team makes that restart appear."""
    tr = scripted_tracking([
        (0, 100, roll(30.0, 34.0, 0), 0, 3.0),
        (100, 200, lambda t: (50.0, -3.0), None, 2.0),
        (200, 300, roll(50.0, 0.5, 200), None, 3.0),
        (300, 400, lambda t: (90.0, -3.0), None, 2.0),
    ])
    tr.players_xy[200:300] = np.nan
    tr.players_speed[200:300] = np.nan
    events = [event("SET PIECES", 190, 215, "Atletic Club")]
    shares = {}
    for name, flags in (("off", {}), ("on", {"use_setpiece_team": True})):
        labels = tagged(flags, tr, events)
        cfg = config.from_dict({"tagging": flags})
        shares[name] = diagnostics.setpiece_consistency(
            labels, events, cfg, BLOCK, TMAP)["restart_timing"]
    assert shares["off"]["share_tags_with_restart"] == 0.0
    assert shares["on"]["share_tags_with_restart"] == 1.0
    assert shares["on"]["independent"] is False
