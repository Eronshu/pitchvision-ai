# Notices: what is whose, and under which terms

## 1. Competition organisers' files

The following files are unmodified from the official More than a Hack 2027 repository
(https://github.com/metrica-sports/more-than-a-hack-2027):

- the top-level competition documents: `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `FAQ.md`,
  `OVERVIEW.md`, `RULES.md`, `SUBMISSION.md` and `CITATION.cff`;
- `games.py`, `starter/`, `evaluation/` and `data/README.md`;
- `requirements.txt` and `requirements-starter.txt`;
- the public 5-minute ground truth and the baseline files in `evaluation/`.

Exceptions:
- `README.md` has a PitchVision AI section added at the top. The organisers' text follows it,
  unchanged.
- `.gitignore` has lines appended at the end.

These files are licensed under `LICENSE` (MIT, Copyright (c) 2026-2027 Metrica Sports), including
its scope statement.

## 2. PitchVision AI team contributions

Everything else is the team's original work, licensed under `LICENSE-PITCHVISION-AI` (MIT,
Copyright (c) 2026 PitchVision AI team). That includes:
- `pitchvision/`, `pvext/`, `astra5/`, `configs/`, `models/` and `candidate/`;
- `scripts/` (none of it came from the organisers);
- `tests/` and `docs/`;
- `requirements-pv.txt`, `requirements-dev.txt`, `requirements-v2.txt`, `requirements-ha3.txt`,
  `requirements-release.txt`;
- `pytest.ini`;
- `NOTICE.md`, `LICENSE-PITCHVISION-AI` and the PitchVision AI section of `README.md`.

## 3. Third-party data used to train the models in `models/` and `candidate/models/`

All five model files were trained only on the IDSSE open dataset. SkillCorner Open Data
(https://github.com/SkillCorner/opendata, MIT licence) was used for evaluation only and is not
redistributed.

**Attribution** (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/):

> IDSSE open data: DFL Deutsche Fußball Liga, collected by Sportec Solutions (TRACAB), published by
> Bassek, M. et al. (2025), "An integrated dataset of spatiotemporal and event data in elite
> soccer", Scientific Data 12:195, doi:10.1038/s41597-025-04505-y; dataset
> doi:10.6084/m9.figshare.28196177 (v1).

**Changes we made:**
- converted from DFL XML to arrays and to the competition's ATD text format;
- synthetically degraded (ball dropout, tracklets, noise);
- used to fit the model parameters shipped in `models/` and `candidate/models/`.

The raw IDSSE files are **not** redistributed here. No endorsement by the DFL, Sportec Solutions or
the authors is implied.

## 4. Competition match data

The match videos, tracking, Smart Tagging and homography are licensed for the competition only, and
**none of them is included** in this repository:
- `data/` ships empty;
- no predictions, extracts or score files derived from the match are committed. The technical
  document of release C quotes a few aggregate statistics of the competition tracking and states
  how they were obtained (its §3 and §10.4);
- the full-match ground truth was never available to the team.

The submitted possession file is delivered only through the organisers' submission form.
