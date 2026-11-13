import inspect

import numpy as np
import pytest

from pvext import convert, degrade

FPS = 25


def synthetic_frames(T=30000, persons=22, seed=7, gap=(10000, 10500)):
    """Clean, DFL-like tracking: every player and the ball present on every frame (except one
    empty half-time gap). Random-walk players, a ball that is sometimes high."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.12, size=(T, persons, 2))
    players = np.cumsum(steps, axis=0) + rng.uniform([10, 10], [95, 58], size=(persons, 2))
    players = np.clip(players, 1, [104, 67]).astype(np.float32)
    ball = players[:, 0, :] + 0.5
    z = np.where((np.arange(T) // 200) % 7 == 0, 3.0, 0.1).astype(np.float32)
    speed = np.full((T, persons), 3.0, dtype=np.float32)
    a, b = gap
    players[a:b], ball[a:b], z[a:b], speed[a:b] = np.nan, np.nan, np.nan, np.nan
    return convert.FramesTrack(players_xy=players, players_speed_mps=speed,
                               team_of_slot=np.array([0] * (persons // 2) + [1] * (persons // 2),
                                                     dtype=np.int8),
                               ball_xy=ball.astype(np.float32), ball_z=z, pitch=(105.0, 68.0),
                               speed_unit="m/s")


@pytest.fixture(scope="module")
def frames():
    return synthetic_frames()


def _same(a, b):
    return all(np.array_equal(x, y, equal_nan=True) for x, y in
               ((a.players_xy, b.players_xy), (a.ball_xy, b.ball_xy),
                (a.players_speed_mps, b.players_speed_mps), (a.team_of_slot, b.team_of_slot)))


def test_presets_and_monotone_severity():
    assert list(degrade.PRESETS) == ["clean", "mild", "doc", "severe"]
    mild, doc, severe = (degrade.PRESETS[k] for k in ("mild", "doc", "severe"))
    assert mild["ball_present"] > doc["ball_present"] > severe["ball_present"]
    assert mild["player_missing"] < doc["player_missing"] < severe["player_missing"]
    assert mild["tracklet_mean_s"] > doc["tracklet_mean_s"] > severe["tracklet_mean_s"]
    assert doc["ball_present"] == pytest.approx(0.593) and doc["n_slots"] == 46
    for name, preset in degrade.PRESETS.items():
        assert preset["source"], name                       # every preset states its basis


def test_apply_takes_no_labels():
    params = inspect.signature(degrade.apply).parameters
    assert list(params) == ["frames", "preset", "seed"]


def test_clean_is_identity(frames):
    assert _same(degrade.apply(frames, degrade.PRESETS["clean"], seed=0), frames)


@pytest.mark.parametrize("preset", ["mild", "doc", "severe"])
def test_deterministic_and_seed_dependent(frames, preset):
    a = degrade.apply(frames, degrade.PRESETS[preset], seed=3)
    b = degrade.apply(frames, degrade.PRESETS[preset], seed=3)
    c = degrade.apply(frames, degrade.PRESETS[preset], seed=4)
    assert _same(a, b) and not _same(a, c)


def test_input_not_mutated(frames):
    before = frames.ball_xy.copy(), frames.players_xy.copy()
    degrade.apply(frames, degrade.PRESETS["severe"], seed=1)
    assert np.array_equal(before[0], frames.ball_xy, equal_nan=True)
    assert np.array_equal(before[1], frames.players_xy, equal_nan=True)


@pytest.mark.parametrize("target", [0.9, 0.593, 0.45])
def test_ball_present_rate(frames, target):
    preset = dict(degrade.PRESETS["doc"], ball_present=target, false_ball_rate=0.0)
    out = degrade.apply(frames, preset, seed=11)
    playable = ~np.isnan(frames.ball_xy[:, 0])
    achieved = (~np.isnan(out.ball_xy[playable, 0])).mean()
    assert achieved == pytest.approx(target, abs=0.03)


def test_ball_dropout_bursty_and_harder_when_high(frames):
    preset = degrade.PRESETS["doc"]
    out = degrade.apply(frames, preset, seed=5)
    playable = ~np.isnan(frames.ball_xy[:, 0])
    missing = np.isnan(out.ball_xy[:, 0]) & playable
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    assert runs.mean() >= 0.8 * preset["ball_gap_s"] * FPS
    high = playable & (frames.ball_z > 1.5)
    assert missing[high].mean() > missing[playable & ~high].mean()


def test_slots_bounded_team_fixed_and_fragmented(frames):
    preset = degrade.PRESETS["doc"]
    out = degrade.apply(frames, preset, seed=2)
    assert out.players_xy.shape[1] == preset["n_slots"] == len(out.team_of_slot)
    assert (out.team_of_slot[: preset["n_slots"] // 2] == 0).all()
    assert (out.team_of_slot[preset["n_slots"] // 2:] == 1).all()
    occupied = ~np.isnan(out.players_xy[..., 0])
    segments = sum(int((np.diff(np.concatenate(([0], col.astype(np.int8), [0]))) == 1).sum())
                   for col in occupied.T)
    assert segments > 3 * frames.players_xy.shape[1]      # identities do not persist


def test_visible_players_match_missing_share(frames):
    preset = degrade.PRESETS["doc"]
    out = degrade.apply(frames, preset, seed=8)
    playable = ~np.isnan(frames.ball_xy[:, 0])
    visible = (~np.isnan(out.players_xy[playable, :, 0])).sum(axis=1).mean()
    expected = 22 * (1 - preset["player_missing"])
    assert visible == pytest.approx(expected, abs=1.0)


def test_noise_is_smooth_not_per_frame(frames):
    preset = dict(degrade.PRESETS["doc"], player_missing=0.0, tracklet_mean_s=1e6)
    out = degrade.apply(frames, preset, seed=9)
    # compare speeds of person 0, slot-agnostic: find the slot holding it on frame 0
    clean_v = np.diff(frames.players_xy[:9000, 0, 0]) * FPS
    slot = int(np.nanargmin(np.abs(out.players_xy[0, :, 0] - frames.players_xy[0, 0, 0])))
    noisy_v = np.diff(out.players_xy[:9000, slot, 0]) * FPS
    extra = np.nanstd(noisy_v - clean_v)
    assert extra < 1.0          # iid noise with sigma 0.2 m would add about 7 m/s


def test_false_ball_detections(frames):
    preset = dict(degrade.PRESETS["doc"], ball_present=1.0, false_ball_rate=0.02,
                  noise_ball_m=0.0)
    out = degrade.apply(frames, preset, seed=4)
    playable = ~np.isnan(frames.ball_xy[:, 0])
    jump = np.linalg.norm(out.ball_xy[playable] - frames.ball_xy[playable], axis=1)
    assert (jump > 4.0).mean() == pytest.approx(0.02, abs=0.01)


def test_gap_frames_stay_empty(frames):
    out = degrade.apply(frames, degrade.PRESETS["severe"], seed=0)
    assert np.isnan(out.ball_xy[10000:10500]).all()
    assert np.isnan(out.players_xy[10000:10500]).all()


# -- known issue 1 (T4-T6 review): _crowd() must honour the preset's crowd_radius_m -------------

FROZEN_HASHES = {   # apply() output with seed 0 on synthetic_frames(T=6000), pinned before the fix
    "mild": "e36aa8356fd14ea2004301fa50133d560e4b8bdfaeec4d9915cb03b5facaa662",
    "doc": "ae7cd326aaa84ccac8456e068782be0293e2eceaecc244d7910a1070eacdd047",
    "severe": "65230a225ff848e64d6b22e449b54d3d1500b85614ae6ecdd76165ca704b8952",
}


def _digest(out):
    import hashlib
    h = hashlib.sha256()
    for a in (out.players_xy, out.players_speed_mps, out.team_of_slot, out.ball_xy):
        h.update(a.tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("name", ["mild", "doc", "severe"])
def test_frozen_presets_byte_identical(name):
    frames = synthetic_frames(T=6000, gap=(2000, 2100))
    assert _digest(degrade.apply(frames, degrade.PRESETS[name], seed=0)) == FROZEN_HASHES[name]


def test_crowd_radius_parameter_is_used():
    frames = synthetic_frames(T=6000, gap=(2000, 2100))
    assert (degrade._crowd(frames, 9.0) >= degrade._crowd(frames, 3.0)).all()
    assert (degrade._crowd(frames, 9.0) > degrade._crowd(frames, 3.0)).any()
    wide = dict(degrade.PRESETS["doc"], crowd_radius_m=9.0)
    assert _digest(degrade.apply(frames, wide, seed=0)) != FROZEN_HASHES["doc"]
