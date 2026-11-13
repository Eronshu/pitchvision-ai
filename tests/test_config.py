import json
import os

import pytest

from pitchvision import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(ROOT, "configs", "v1.json")


def test_defaults_match_file():
    assert config.load(V1) == config.defaults()


@pytest.mark.parametrize("override, fragment", [
    ({"alive": {"foo": 1}}, "alive.foo"),
    ({"bogus": {}}, "bogus"),
    ({"possession": {"boundary": "first_touch"}}, "boundary"),
    ({"alive": {"motion_low_mps": 2.0, "motion_high_mps": 1.0}}, "motion_low_mps"),
    ({"alive": {"dead_off": 0.4}}, "alive.dead_off"),
    ({"possession": {"r_ctrl_m": -1.0}}, "r_ctrl_m"),
])
def test_invalid_config_rejected(override, fragment):
    with pytest.raises(ValueError, match=fragment):
        config.from_dict(override)


@pytest.mark.parametrize("boundary", ["evidence_start", "confirmed"])
def test_valid_boundaries(boundary):
    cfg = config.from_dict({"possession": {"boundary": boundary}})
    assert cfg["possession"]["boundary"] == boundary


def test_partial_override_keeps_other_defaults():
    cfg = config.from_dict({"alive": {"min_dead_s": 3.0}})
    assert cfg["alive"]["min_dead_s"] == 3.0
    assert cfg["alive"]["dead_on"] == config.defaults()["alive"]["dead_on"]


def test_hash_stable():
    assert config.sha256(config.defaults()) == \
        config.sha256(json.loads(json.dumps(config.defaults())))
