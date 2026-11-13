# PitchVision AI — HA3 release, M1 fallback, reproduction and submission checklist

**Who runs what.** Everything below that reads the competition match (BAR-ATH) is run by the
team, locally. It is never run from an AI-assistant session: the match data must not reach a
hosted LLM (OVERVIEW.md, "Use of Large Language Models").

**Scope.** Nothing here measures competition accuracy, and nothing here uses the public
five-minute labels. Do not use them to choose between HA3 and M1; that choice has already been
made: HA3 is primary, by project-lead override (`docs/HA3_TECHNICAL_DOCUMENT.md` §5).

## 1. Environment

Python 3.13, with the dependencies pinned exactly. The releases refuse to load under other
versions.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-ha3.txt
.venv/bin/python -m pytest -q -W error          # synthetic only, no data needed
```

## 2. Release artifacts (tracked in git)

| file | sha256 |
|---|---|
| `configs/ha3.json` | HA3 release: λ, feature schema, frozen-code hashes, dependency pins, dead-ball model |
| `models/ha3_final.joblib` | db988179bdf367c8556f884d5a16636296e5b7297d4b6d195d8013ffc7ba955b |
| `models/ha3_dead_m1_league.joblib` | 850c4daad35b4db3e0d862bbccfc9f49faff17110c247c93734fad32a52d1290 |
| `configs/v2.json` | M1 release, also loaded and checked by HA3 |
| `models/m1_final.joblib` | 4b2b59d0879f484eba334ccb5a1ceac1531736d34f7624dba2513a1a64788bf8 |

The frozen HA3 prediction code is pinned in both `configs/ha3.json` and
`pvext/ha/release_check.FROZEN_HA3`:

| file | sha256 |
|---|---|
| `pvext/ha/features.py` | c078865d… |
| `pvext/ha/decode.py` | 400b1e3e… |
| `pvext/ha/experiment.py` | 61a0eb99… |

## 3. One-command local check (recommended)

```bash
export MTAH_DATA_DIR=/path/to/mtah2027      # or unpack the match into ./data
bash scripts/local_release_e2e.sh /path/outside/repo/pv_release
```

The script runs every step of §4 and §5. It prints only PASS/FAIL lines, sha256 values, run
time and peak memory, and writes them to `<out>/SUMMARY.txt`. It ends with `ALL PASS` or names
the failing check. The candidates are written to:

- HA3, primary: `<out>/ha3/PitchVision AI.csv`
- M1, fallback: `<out>/m1/PitchVision AI.csv`

## 4. HA3 (primary), step by step

```bash
python scripts/run_ha3.py --config configs/ha3.json --out /path/outside/repo/pv_ha3
python scripts/verify_ha3_release.py /path/outside/repo/pv_ha3/manifest.json
python scripts/verify_submission.py /path/outside/repo/pv_ha3/manifest.json --strict
python evaluation/validate_submission.py "/path/outside/repo/pv_ha3/PitchVision AI.csv"
```

Exit codes of `run_ha3.py`:

| code | meaning |
|---|---|
| 0 | success |
| 1 | the file fails the official validation |
| 2 | bad output path |
| 3 | a release check failed before any data was read |
| 4 | a check failed after feature extraction |

`verify_ha3_release.py` re-checks all of the following:
- the commit and a clean tree;
- every M1 check;
- the HA3 config, the HA3 model and the dead-ball model (tracked in git, inside the repository,
  hashes match);
- the frozen HA3 code;
- the CSV hash and the official validation at 199,319 rows.

## 5. M1 fallback, step by step

```bash
python scripts/run_v2.py --config configs/v2.json --out /path/outside/repo/pv_m1
python scripts/verify_v2_release.py /path/outside/repo/pv_m1/manifest.json
python scripts/verify_submission.py /path/outside/repo/pv_m1/manifest.json --strict
```

How the M1 fallback CSV differs from HA3:
- Its H/A labels differ from HA3's.
- Its D frames differ slightly. M1 uses the all-seven M1 final model; HA3 uses the dev-trained
  league model it was evaluated with.

The last-resort fallback is V1: `scripts/run_v1.py` (see `docs/PITCHVISION.md`).

## 6. Reproduction of the research (IDSSE only, not needed for the submission)

- The HA research driver is `scripts/ha_experiment.py`, with a dev stage and a once-only lockbox
  stage.
- Its outputs, the research journal and the decision log are kept locally, outside git, under
  `External/idsse/v3/ha/`.
- The lockbox stage refuses to run again once its directory exists. Re-running it is not part
  of the release.
- `fit_gb_sym` is bit-identical across runs and at 1, 4 and 8 OpenMP threads (synthetic check).

## 7. Local BAR-ATH release record (2026-11-07)

Run by the project lead on the team's own machine, not from an AI-assistant session, with
`scripts/local_release_e2e.sh` at commit `10435695a71bc19fa15ef614bd59aa0d949d46d8`. Result:
**ALL PASS**.

| | HA3 (primary) | M1 (fallback) |
|---|---|---|
| data rows | 199,319 | 199,319 |
| official `validate_submission.py` | OK | OK |
| `verify_*_release.py` and `verify_submission.py --strict` | OK | OK |
| byte-identical: run A, run B, clean clone | yes | yes |
| sha256 of `PitchVision AI.csv` | `a512018d99108e8e97b8a1d53d66bebd4fd6c4c86f7478170335721e81d34d29` | `3f6fb3bb38ebd259cf8e741f7c2e6fa536812a510941830bc2b5e83648cf1bb2` |
| wall time per run | 8–9 s | 6–7 s |
| peak memory | about 1.67 GB | about 1.55 GB |

What this record shows and does not show:
- It shows that the repository generates, validates and reproduces both files deterministically
  on macOS.
- It is **not** an accuracy measurement. No competition spellF1, macroF1 or ranking has been
  measured, and none is claimed.
- The release verifiers require HEAD to equal the commit recorded in the manifest. Commits after
  `1043569` change documentation only, so regenerating at the final commit must reproduce the two
  sha256 values above; rerun the script there before uploading (checklist item 3).
- Linux reproduction has not been tested.

## 8. Final submission checklist

Status as of 2026-11-07. Items 3–5 were completed at `1043569` and must be repeated once at the
final approved commit.

1. [ ] The release has been reviewed and the project lead approves it.
2. [x] Clean checkout at the approved commit (`git status` empty), with `requirements-ha3.txt`
   installed. *(done at `1043569`)*
3. [x] `bash scripts/local_release_e2e.sh <out>` ends with `ALL PASS`. Record `SUMMARY.txt`
   (sha256 values, runtime, peak memory). *(done at `1043569`, §7; repeat at the final commit)*
4. [x] HA3 and M1 CSVs: 199,319 data rows plus the header; official validation OK;
   `verify_*_release.py` OK; `verify_submission.py --strict` OK. *(done at `1043569`)*
5. [x] HA3 regenerates byte-identically: twice in the checkout and once from a clean clone.
   *(done at `1043569`)*
6. [ ] The code repository is **public** (SUBMISSION.md, deliverable 2) and contains the approved
   commit, with no match data in its history. This is a decision for the lead.
7. [ ] The technical document `docs/submission/PitchVision AI.md` is reviewed, its repository URL
   filled in, and it is exported as `PitchVision AI.pdf` (or uploaded as `.md`).
8. [ ] Upload `PitchVision AI.csv` from `<out>/ha3/` with the technical document, the team name and
   the public repository URL, through the organisers' form, before 2026-11-17. The sha256 of the
   uploaded file must equal the HA3 value in §7. This is a manual decision by the lead.
9. [ ] Keep `<out>/m1/` (M1) and the V1 output as local fallbacks, with their manifests.
10. [ ] Do not tune or choose anything on the public five-minute labels.
