#!/usr/bin/env bash
# PitchVision end-to-end release check on the competition match -- run LOCALLY by the team, never
# from an AI-assistant session (the match data must not reach a hosted LLM).
#
#   MTAH_DATA_DIR=/path/to/mtah2027 bash scripts/local_release_e2e.sh /path/outside/repo/pv_release
#
# What it does, for HA3 (configs/ha3.json, primary) and M1 (configs/v2.json, fallback):
#   1. checks the checkout is clean and at a commit (the release verifiers require it)
#   2. generates each submission twice in this checkout, timing each run and its peak memory
#   3. clones this commit into <out>/clone and regenerates both submissions there
#   4. byte-compares all three copies of each CSV
#   5. runs verify_ha3_release.py / verify_v2_release.py (all hashes, tracked artifacts, frozen
#      code, official validation at 199,319 rows), verify_submission.py --strict and
#      evaluation/validate_submission.py, and counts the CSV lines
# It never reads any ground truth and prints no labels: only PASS/FAIL lines, sha256, seconds and
# peak memory, collected in <out>/SUMMARY.txt. Uploading a file is a separate, manual decision.
set -u
OUT=${1:?usage: bash scripts/local_release_e2e.sh <output dir outside the repository>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-$ROOT/.venv/bin/python}
ROWS=${PV_E2E_ROWS:-199319}          # PV_E2E_* are test hooks (synthetic ATD); leave them unset
RUN_ARGS=${PV_E2E_RUN_ARGS:-}
VERIFY_ARGS=""
[ -n "${PV_E2E_ROWS:-}" ] && VERIFY_ARGS="--rows $ROWS"
CSV="PitchVision AI.csv"
mkdir -p "$OUT" || exit 2
OUT=$(cd "$OUT" && pwd)
case "$OUT/" in "$ROOT"/*) echo "output must be outside the repository" >&2; exit 2;; esac
if [ -z "${MTAH_DATA_DIR:-}" ]; then export MTAH_DATA_DIR="$ROOT/data"; fi
SUMMARY="$OUT/SUMMARY.txt"
: > "$SUMMARY"
FAILED=0
say() { echo "$*" | tee -a "$SUMMARY"; }
check() { if [ "$1" = 0 ]; then say "PASS $2"; else say "FAIL $2"; FAILED=1; fi; }
sha() { shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1 || sha256sum "$1" | cut -d' ' -f1; }

timed() {   # timed <log> <cmd...>: run, recording wall seconds and peak RSS (MB) in <log>.time
  local log=$1; shift
  local start=$(date +%s)
  if [ "$(uname)" = Darwin ]; then
    /usr/bin/time -l "$@" > "$log" 2> "$log.err"; local rc=$?
    local rss=$(awk '/maximum resident set size/ {printf "%.0f", $1/1048576}' "$log.err")
  else
    /usr/bin/time -v "$@" > "$log" 2> "$log.err"; local rc=$?
    local rss=$(awk -F: '/Maximum resident set size/ {printf "%.0f", $2/1024}' "$log.err")
  fi
  echo "$(( $(date +%s) - start )) ${rss:-?}" > "$log.time"
  return $rc
}

cd "$ROOT" || exit 2
HEAD=$(git rev-parse HEAD)
say "commit $HEAD  python $($PY -c 'import platform;print(platform.python_version(), platform.platform())')"
[ -z "$(git status --porcelain)" ]; check $? "checkout is clean"

run_release() {   # run_release <tree> <name> <script> <config> <dest>
  local tree=$1 name=$2 script=$3 config=$4 dest=$5
  rm -rf "$dest"
  (cd "$tree" && timed "$dest.log" "$PY" "$script" --config "$config" --out "$dest" $RUN_ARGS < /dev/null)
  local rc=$?
  read -r secs rss < "$dest.log.time"
  check $rc "$name generated ($dest) in ${secs}s, peak ${rss} MB"
}

for spec in "ha3 scripts/run_ha3.py configs/ha3.json scripts/verify_ha3_release.py" \
            "m1 scripts/run_v2.py configs/v2.json scripts/verify_v2_release.py"; do
  set -- $spec
  name=$1 script=$2 config=$3 verify=$4
  run_release "$ROOT" "$name run A" "$script" "$config" "$OUT/$name"
  run_release "$ROOT" "$name run B" "$script" "$config" "$OUT/${name}_again"
done

rm -rf "$OUT/clone"
git clone -q "$ROOT" "$OUT/clone" && (cd "$OUT/clone" && git checkout -q "$HEAD")
check $? "clean clone of $HEAD"
for spec in "ha3 scripts/run_ha3.py configs/ha3.json" "m1 scripts/run_v2.py configs/v2.json"; do
  set -- $spec
  run_release "$OUT/clone" "$1 run from clean clone" "$2" "$3" "$OUT/${1}_clone"
done

for spec in "ha3 scripts/verify_ha3_release.py" "m1 scripts/verify_v2_release.py"; do
  set -- $spec
  name=$1 verify=$2
  f="$OUT/$name/$CSV"
  cmp -s "$f" "$OUT/${name}_again/$CSV" && cmp -s "$f" "$OUT/${name}_clone/$CSV"
  check $? "$name: run A, run B and clean-clone CSVs are byte-identical"
  "$PY" "$verify" "$OUT/$name/manifest.json" $VERIFY_ARGS > "$OUT/$name.verify_release.txt" 2>&1 < /dev/null
  check $? "$name: $verify (hashes, tracked artifacts, official validation)"
  (cd "$OUT/clone" && "$PY" "$verify" "$OUT/${name}_clone/manifest.json" $VERIFY_ARGS > "$OUT/${name}_clone.verify_release.txt" 2>&1 < /dev/null)
  check $? "$name: $verify on the clean-clone run"
  "$PY" scripts/verify_submission.py "$OUT/$name/manifest.json" --strict --rows $ROWS > "$OUT/$name.verify_submission.txt" 2>&1 < /dev/null
  check $? "$name: verify_submission.py --strict"
  "$PY" evaluation/validate_submission.py "$f" --rows $ROWS > "$OUT/$name.validate.txt" 2>&1 < /dev/null
  check $? "$name: official validate_submission.py at $ROWS rows"
  [ "$(wc -l < "$f" | tr -d ' ')" = $((ROWS + 1)) ]
  check $? "$name: $ROWS data rows + header"
  say "     $name sha256 $(sha "$f")"
done

"$PY" - "$OUT" >> "$SUMMARY" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1] + "/ha3/manifest.json"))
print("     ha3 model {}  dead-ball model {}  M1 fallback model {}".format(
    m["ha3_model_sha256"][:16], m["dead_model_sha256"][:16], m["model_sha256"][:16]))
EOF
if [ $FAILED = 0 ]; then say "ALL PASS"; else say "SOME CHECKS FAILED -- see $OUT/*.txt and *.err"; fi
exit $FAILED
