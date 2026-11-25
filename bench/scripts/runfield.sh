#!/usr/bin/env bash
# Copyright 2026 Inworld AI — Apache-2.0
#
# Renders the board's external systems over the eval set, one venv each
# (installed by setupfield.sh).
#
#   VOICE_ISOLATION_EVALS=<set> BENCH_OUT=<outputs> BENCH_RESULTS=<results> \
#   BENCH_VENVS=<venvs> BENCH_SRC=<sources> [DEVICE=cuda:0] runfield.sh [system ...]
#
# Every wrapper process writes a receipt before it renders (wrappers/receipt.py:
# its code, the release's source and weights, packages, device, arguments and
# inputs). Before a system renders, this script takes the receipt its venv
# would write now and compares it with the receipts in the system's output
# folder: outputs of this identity are resumed (wrappers skip the files they
# hold); outputs of another identity, or without receipts, stop the system
# unless DISCARD=1, which removes them first. After the wrapper,
# stamprender.py stamps $BENCH_RESULTS/<name>.meta.json only if the wrapper
# exited zero, every receipt carries this identity and the inventory is exact.
# STAMP_ONLY=1 verifies and stamps outputs that wrapper processes rendered
# elsewhere (shards: <wrapper> --shard i/k), the same way. DEVICE is what
# every GPU system computes on; CUDA_VISIBLE_DEVICES chooses the card.
#
# PROBE_IN=<dir> renders another folder of inputs instead, the streaming probe
# (scorestream.py --probe) or the same-room pair (scoreevents.py --pair), into
# $BENCH_OUT/<name>, only in the environment that rendered the board (its
# receipt's environment digest must equal the board's stamp), and stamps
# nothing: the scorer that cut the inputs reads the receipts.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
IN=${PROBE_IN:-${VOICE_ISOLATION_EVALS:-$ROOT/data/voice-isolation-evals}/mix}
OUT=${BENCH_OUT:-$ROOT/data/outputs}
RES=${BENCH_RESULTS:-$ROOT/bench/results}
VENVS=${BENCH_VENVS:-$HOME/.cache/voice-isolation-bench/venvs}
DEVICE=${DEVICE:-cuda:0}
W=$HERE/../wrappers
export BENCH_SRC=${BENCH_SRC:-$HOME/.cache/voice-isolation-bench/src}
[ -n "$(ls "$IN"/*.wav 2>/dev/null | head -1)" ] || {
  echo "no inputs under $IN (set VOICE_ISOLATION_EVALS or PROBE_IN)"; exit 1; }
SET_ID=""
if [ -z "${PROBE_IN:-}" ]; then
  SET_ID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['set_id'])" \
    "$(dirname "$IN")/checksums.json" 2>/dev/null) || {
    echo "no checksums.json with a set_id beside $IN"; exit 1; }
fi
mkdir -p "$OUT" "$RES"

field () {  # field <key>: one field of the JSON document on stdin
  python3 -c "import json,sys; print(json.load(sys.stdin).get(sys.argv[1]) or '')" "$1"
}

held () {  # held <out dir>: the identity digests of the receipts there
  python3 - "$1" <<'PY'
import glob, json, os, sys
paths = glob.glob(os.path.join(sys.argv[1], ".receipt-*.json"))
print(" ".join(sorted({json.load(open(p)).get("digest", "") for p in paths})))
PY
}

FAILS=0
fail () { echo "  !! $*"; FAILS=$((FAILS + 1)); }

run () {  # run <system> <venv> <wrapper> <device> [wrapper arguments]
  local name=$1 venv=$2 wrap=$3 dev=$4; shift 4
  local py="$VENVS/${venv}_venv/bin/python" out="$OUT/$name" probe digest
  echo "===== $name"
  # each venv carries its own CUDA libraries: the node's must not shadow them
  probe=$(env -u LD_LIBRARY_PATH "$py" "$W/receipt.py" probe "$name" "$W/$wrap" \
    --in-dir "$IN" --device "$dev" -- "$@") || {
    fail "$name: no receipt from this environment (venv, release or weights missing)"; return; }
  digest=$(field digest <<< "$probe")
  if [ -n "${PROBE_IN:-}" ]; then
    local board
    board=$(field render_environment_sha256 < "$RES/$name.meta.json" 2>/dev/null)
    [ "$(field environment_digest <<< "$probe")" = "$board" ] || {
      fail "$name: this environment did not render the board's stamp ($RES/$name.meta.json)"; return; }
  fi
  if [ -z "${STAMP_ONLY:-}" ]; then
    local holds foreign=""
    holds=$(held "$out")
    if [ -n "$holds" ]; then
      [ "$holds" = "$digest" ] || foreign=1
    elif [ -n "$(ls "$out"/*.wav 2>/dev/null | head -1)" ]; then
      foreign=1  # outputs without a receipt: of unknown origin
    fi
    if [ -n "$foreign" ]; then
      [ -n "${DISCARD:-}" ] || {
        fail "$name: $out holds outputs of another identity; DISCARD=1 removes them"; return; }
      echo "  discarding the outputs of another identity in $out"
      rm -rf "$out"
    fi
    env -u LD_LIBRARY_PATH "$py" "$W/$wrap" --in-dir "$IN" --out-dir "$out" \
      --device "$dev" "$@" 2>&1 | tail -3
    local code=${PIPESTATUS[0]}
    [ "$code" -eq 0 ] || { fail "$name: the wrapper exited $code"; return; }
  fi
  [ -n "${PROBE_IN:-}" ] && return
  "${BENCH_PY:-python3}" "$HERE/stamprender.py" "$name" --wrapper "$W/$wrap" \
    --inputs "$IN" --outputs "$out" --results "$RES" --set-id "$SET_ID" \
    --wrapper-exit 0 --probe-digest "$digest" || fail "$name: not stamped"
}

ALL="reuse_nemotron reuse_nemotron_laughter gap_urgenet clearvoice_mf2_48k urgent26_bsrnn_flow resemble_dn \
fastenhancer_l_48k dfn3 urgent26_bsrnn unipase smule"
for s in ${*:-$ALL}; do
  case $s in
    reuse_nemotron)      run reuse_nemotron reuse runreusenemotron.py "$DEVICE" ;;
    reuse_nemotron_laughter) run reuse_nemotron_laughter reuse runreusenemotronlaughter.py "$DEVICE" ;;
    gap_urgenet)         run gap_urgenet gap_urgenet rungapurgenet.py "$DEVICE" ;;
    clearvoice_mf2_48k)  run clearvoice_mf2_48k clearvoice runclearvoice.py "$DEVICE" ;;
    urgent26_bsrnn_flow) run urgent26_bsrnn_flow urgent26 runurgent26bsrnnflow.py "$DEVICE" ;;
    resemble_dn)         run resemble_dn resemble runresembledn.py "$DEVICE" ;;
    fastenhancer_l_48k)  run fastenhancer_l_48k fastenhancer runfastenhancer.py cpu ;;
    dfn3)                run dfn3 dfn3 rundfn3.py "$DEVICE" ;;
    urgent26_bsrnn)      run urgent26_bsrnn urgent26 runurgent26bsrnn.py "$DEVICE" ;;
    unipase)             run unipase unipase rununipase.py "$DEVICE" ;;
    smule)               run smule smule runsmule.py "$DEVICE" ;;
    *) fail "unknown system $s" ;;
  esac
done
echo "FIELD DONE ($FAILS problem(s))"
[ "$FAILS" -eq 0 ]
