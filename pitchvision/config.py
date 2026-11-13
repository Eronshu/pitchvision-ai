"""Load and validate the V1 configuration.

Every threshold and every uncertain convention lives in one JSON file, so changing a boundary rule
once the organisers answer (issues #17, #31) is an edit to configs/*.json and nothing else. A file
only needs the keys it changes; everything else comes from defaults(). Unknown keys are an error
rather than ignored, because a misspelt threshold that silently falls back to its default is the
kind of mistake nobody finds.
"""

import copy
import hashlib
import json

# Params
BOUNDARIES = ("evidence_start", "confirmed")

# Keys that may be negative: boundary offsets shift a dead-ball run either way.
SIGNED = {("alive", "dead_start_offset_frames"), ("alive", "dead_end_offset_frames")}

_DEFAULTS = {
    "features": {"ball_speed_smooth_frames": 5, "out_margin_m": 1.0,
                 "exit_near_m": 3.0, "exit_lookback_s": 0.4, "max_ball_speed_mps": 40.0},
    "alive": {"w_static": 0.35, "w_motion": 0.35, "w_gap": 0.3, "w_tag": 0.2,
              "static_speed_mps": 0.5, "static_min_s": 1.0,
              "motion_low_mps": 0.8, "motion_high_mps": 1.8, "gap_scale_s": 4.0,
              "smooth_s": 1.0, "dead_on": 0.6, "r_hold_m": 1.0, "hold_max_s": 6.0,
              "out_confirm_frames": 3,
              "min_dead_s": 2.0, "min_alive_s": 1.0, "max_dead_s": 120.0,
              "restart_speed_mps": 3.0, "restart_min_s": 0.2,
              "dead_start_offset_frames": 0, "dead_end_offset_frames": 0},
    "possession": {"r_ctrl_m": 1.5, "r_decay_m": 1.5, "margin_m": 1.0,
                   "v_ref_mps": 15.0, "speed_floor": 0.3,
                   "win_s": 0.4, "evidence_on": 0.2, "switch_on": 0.5, "switch_margin": 0.2,
                   "init_on": 0.3, "boundary": "evidence_start"},
    "tagging": {"use_setpiece_dead": False, "use_setpiece_team": False},
}


def defaults():
    """A fresh copy of the default configuration."""
    return copy.deepcopy(_DEFAULTS)


def from_dict(overrides):
    """Defaults with `overrides` merged in key by key, validated. Raises ValueError."""
    cfg = defaults()
    for section, values in overrides.items():
        if section not in cfg:
            raise ValueError("unknown config section {!r}; have {}".format(section, sorted(cfg)))
        if not isinstance(values, dict):
            raise ValueError("config section {!r} must be an object".format(section))
        for key, value in values.items():
            if key not in cfg[section]:
                raise ValueError("unknown config key {}.{}".format(section, key))
            cfg[section][key] = value
    _validate(cfg)
    return cfg


def load(path):
    """from_dict() of a JSON file."""
    with open(path) as handle:
        return from_dict(json.load(handle))


def sha256(cfg):
    """Hash of the canonical JSON form, recorded in every run manifest."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()


def _validate(cfg):
    for section, values in cfg.items():
        for key, value in values.items():
            expected = type(_DEFAULTS[section][key])
            if expected is bool:
                if not isinstance(value, bool):
                    raise ValueError("{}.{} must be true or false".format(section, key))
            elif expected in (int, float):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("{}.{} must be a number".format(section, key))
                if expected is int and value != int(value):
                    raise ValueError("{}.{} must be an integer".format(section, key))
                if (section, key) not in SIGNED and value < 0:
                    raise ValueError("{}.{} must be >= 0, got {}".format(section, key, value))

    if cfg["possession"]["boundary"] not in BOUNDARIES:
        raise ValueError("possession.boundary must be one of {}, got {!r}".format(
            BOUNDARIES, cfg["possession"]["boundary"]))
    alive = cfg["alive"]
    if not alive["motion_low_mps"] < alive["motion_high_mps"]:
        raise ValueError("alive.motion_low_mps must be below alive.motion_high_mps")
