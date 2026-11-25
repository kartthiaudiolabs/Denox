#!/usr/bin/env bash
# Copyright 2026 Inworld AI — Apache-2.0
#
# Installs the board's external systems from upstream, each into its own venv.
# Nothing third-party is vendored in this repository: code and weights are
# fetched here, each under its own licence (bench/README.md, "Licenses of the systems compared").
#
#   BENCH_VENVS=<venvs> BENCH_SRC=<sources> setupfield.sh [system ...]
#
# Every venv is synced to its lock (bench/wrappers/locks/<venv>.txt: the exact
# packages that rendered the board), every checkout must be its pinned commit,
# and every weight file must match its pinned sha256: any difference stops the
# script. Each render's receipt records all of it again (wrappers/receipt.py).
# Needs uv, git with git-lfs, curl, unzip and sha256sum.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOCKS=$HERE/../wrappers/locks
UV=${UV:-uv}
VENVS=${BENCH_VENVS:-$HOME/.cache/voice-isolation-bench/venvs}
SRC=${BENCH_SRC:-$HOME/.cache/voice-isolation-bench/src}
HF=https://huggingface.co
GH=https://github.com
mkdir -p "$VENVS" "$SRC"

die () { echo "  !! $*" >&2; exit 1; }

pinned () {  # a wheel named by its URL reads as name==version, whichever uv printed it
  sed -E 's#^([A-Za-z0-9_.-]+) @ https://[^ ]+/[A-Za-z0-9_.]+-([^-]+)-cp[^ ]+\.whl.*#\1==\2#; s#%2B#+#g' | sort
}

locked () {  # locked <lock>: its packages as uv pip freeze prints them (index options aside)
  grep -v -E '^(#|-)' "$1" | pinned
}

frozen () {  # frozen <venv>: its packages, read the same way
  "$UV" pip freeze --python "$1/bin/python" | pinned
}

venv () {  # venv <name> <python> [package built against the installed torch]
  local name=$1 py=$2 late=${3:-} v="$VENVS/${1}_venv" lock="$LOCKS/$1.txt"
  [ -x "$v/bin/python" ] || "$UV" venv --quiet --python "$py" "$v"
  if [ "$(locked "$lock")" = "$(frozen "$v")" ]; then
    echo "  venv $name: at its lock"; return
  fi
  local first; first=$(mktemp --suffix=.txt)
  grep -v "^$late==" "$lock" > "$first"
  "$UV" pip sync --python "$v/bin/python" "$first"
  rm -f "$first"
  if [ -n "$late" ]; then
    "$UV" pip install --python "$v/bin/python" --no-build-isolation --no-deps \
      "$(grep "^$late==" "$lock")"
  fi
  [ "$(locked "$lock")" = "$(frozen "$v")" ] \
    || die "venv $name differs from $lock"
}

clone () {  # clone <dir> <repository> <commit>: a clean checkout at the pinned commit
  local d="$SRC/$1"
  [ -d "$d/.git" ] || git clone --quiet "$2" "$d"
  if [ "$(git -C "$d" rev-parse HEAD)" != "$3" ]; then
    git -C "$d" fetch --quiet origin
    git -C "$d" -c advice.detachedHead=false checkout --quiet "$3"
  fi
  [ "$(git -C "$d" rev-parse HEAD)" = "$3" ] || die "$d is not at $3"
  # upstream tracks some __pycache__ files, which running the code rewrites
  [ -z "$(git -C "$d" status --porcelain --untracked-files=no -- . ':(exclude)*.pyc')" ] \
    || die "$d has local changes"
}

sha () { sha256sum "$1" | cut -d' ' -f1; }

check () {  # check <file> <sha256>
  [ -f "$1" ] || die "$1 is missing"
  [ "$(sha "$1")" = "$2" ] || die "$1: sha256 $(sha "$1"), pinned $2"
}

fetch () {  # fetch <file> <url> <sha256>: downloaded unless present at its pin
  if [ ! -f "$1" ] || [ "$(sha "$1")" != "$3" ]; then
    mkdir -p "$(dirname "$1")"
    curl -fsSL --retry 3 -o "$1.partial" "$2"
    mv "$1.partial" "$1"
  fi
  check "$1" "$3"
}

SYSTEMS=${*:-"reuse_nemotron reuse_nemotron_laughter gap_urgenet clearvoice urgent26 resemble fastenhancer dfn3 unipase smule"}
for s in $SYSTEMS; do
  echo "===== $s"
  case $s in
    reuse_nemotron)  # NVIDIA NemotronLabs RE-USE: OpenMDW-1.1
      venv reuse 3.11
      clone reuse_nemotron "$HF/nvidia/NVIDIA-NemotronLabs-RE-USE" 31f82f2efad1095034492f236e8dac9239ba713b
      check "$SRC/reuse_nemotron/model.safetensors" 06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1
      ;;
    reuse_nemotron_laughter)  # NVIDIA NemotronLabs RE-USE-laughter: OpenMDW-1.1
      venv reuse 3.11
      clone reuse_nemotron_laughter "$HF/nvidia/NVIDIA-NemotronLabs-RE-USE-laughter" 1b1521dd659608e1e063ada83e0acfe1da473a8b
      check "$SRC/reuse_nemotron_laughter/model.safetensors" 35c2a4e284515275d2e49a8e8cb5cd392ae67a1f1d345de1c12a0202c7b6b610
      ;;
    gap_urgenet)  # MIT code, Apache-2.0 weights
      venv gap_urgenet 3.10
      clone gap-urgenet "$GH/Xiaobin-Rong/gap-urgenet" fbf467378fbfd27dd8fd8d0318c05a98eb926705
      w="$HF/Xiaobin-Rong/gap-urgenet/resolve/f1916ff86b83777e2c3e7ecae8752a04b8d88d41"
      while read -r f h; do fetch "$SRC/gap-urgenet/ckpts/$f" "$w/$f" "$h"; done <<'PINS'
config.json faba0c6f14ef6066eebf93568c4914e70d03ab66fc9c969b796aed5067bc8371
DeWavLM-Omni.pt 117a0ced06a2ccc41f5766bcc8da6aa43f98fdc336feb5997a5e5dfea1426738
Adapter.pt 63473a9f764e4a97971530e4d7b987f9472d7b431aee6a982f25f91bcb4d48c6
Vocoder.pt 6881d02168698de3d1083becb6b0c30d4468323f8cfed61f02afb675f4fd81b0
Predictor.pt 1e58c270c12ab953e4df32a34e4b580f780d016bee9bf5580684d6c982f77866
PostNet.pt d401c7572451341d1482bf35a1bc2c590b9fe8c21d17f0b4b1268e0d88d1c066
PINS
      ;;
    unipase)  # MIT code, Apache-2.0 weights
      venv unipase 3.10
      clone unipase "$GH/Xiaobin-Rong/unipase" 857b60ad05d37a2cf6d7a89883ec9fc4fc164b45
      w="$HF/Xiaobin-Rong/unipase/resolve/f0b4d4c4411fe08fc2dddbf2d9f33260c27ac4a0"
      while read -r f h; do fetch "$SRC/unipase/ckpts/$f" "$w/$f" "$h"; done <<'PINS'
config.json 4c11a811af333dd8e93db110b08d1277623e906df549c8c8d5ecac9ffc633136
DeWavLM-Omni.pt 16c94d05ff6ef2bdf08b0be2853826afda687c2d63c833382198db7682476689
Adapter.pt 3f3f9f8076e304bc740fef5496f91528ca9ea6be7f5a63deec71cdc33471c8c7
Vocoder_DWO-L1.pt 7595d48962c296bab2f7cb307f299acf5a02ec6a972d9ab7da3ed71b4a54a3d5
PostNet.pt 1bf950dd2432e3cd2ef7055226b259bb3ac3fba8d7ce1f5f2424581a3d19cf63
PINS
      ;;
    clearvoice)  # Apache-2.0 (the clearvoice package and the checkpoint)
      venv clearvoice 3.11
      w="$HF/alibabasglab/MossFormer2_SE_48K/resolve/eff8c97925c8bec812af707814b3e5d777fd4503"
      d="$SRC/clearvoice/checkpoints/MossFormer2_SE_48K"
      fetch "$d/last_best_checkpoint" "$w/last_best_checkpoint" 315744c841441f8831cb2f896e06102b4d864776bf272febaa30c639c903e1c0
      fetch "$d/last_best_checkpoint.pt" "$w/last_best_checkpoint.pt" 03692b9f773bbd6bb43b9c5a41f96b1e28affd66e13796b7bec66ad3d8b227c6
      ;;
    urgent26)  # the URGENT 2026 Track-1 baselines and the challenge's evaluation toolkit (Apache-2.0)
      venv urgent26 3.10
      clone urgent26 "$GH/urgent-challenge/urgent2026_challenge_track1" b1dc3ad1e86419ff0bd666f455bda7936bff0e9a
      git -C "$SRC/urgent26" submodule update --quiet --init --recursive  # lib/NISQA, lib/scoreq
      w="$HF/lichenda/icassp_2026_urgent_baseline/resolve/d4add2435a74b3f2dd54a9bbd417a058c68983b1"
      fetch "$SRC/urgent26/ckpts/bsrnn.ckpt" "$w/bsrnn.ckpt" 5d6b24eb0ba387428f3490a36238d17902cdc96da534fd2707a8e44f0d2431c8
      fetch "$SRC/urgent26/ckpts/flow_bsrnn.ckpt" "$w/flow_bsrnn.ckpt" f9201821243797fd5f9b852779040057b6f204267935712f96ccf0353cd9d438
      ;;
    resemble)  # MIT (the resemble-enhance package and its weights)
      venv resemble 3.10
      clone resemble-enhance "$HF/ResembleAI/resemble-enhance" 4e3510ce4a8391159f665903544c5150bee7b2cb
      check "$SRC/resemble-enhance/enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt" f9d035f318de3e6d919bc70cf7ad7d32b4fe92ec5cbe0b30029a27f5db07d9d6
      ;;
    fastenhancer)  # MIT
      venv fastenhancer 3.11
      fetch "$SRC/fastenhancer_onnx/fastenhancer_l.onnx" \
        "$GH/aask1357/fastenhancer/releases/download/onnx-48khz-v1/fastenhancer_l.onnx" \
        d7138309ec98266c668b2f86658fcc8f4e82ded67cd7fc8d6598534104b4bf89
      ;;
    dfn3)  # MIT or Apache-2.0 (the deepfilternet package and the model of tag v0.5.6)
      venv dfn3 3.11
      fetch "$SRC/deepfilternet/DeepFilterNet3.zip" \
        "$GH/Rikorose/DeepFilterNet/raw/v0.5.6/models/DeepFilterNet3.zip" \
        49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2
      unzip -qo "$SRC/deepfilternet/DeepFilterNet3.zip" -d "$SRC/deepfilternet"
      check "$SRC/deepfilternet/DeepFilterNet3/config.ini" 415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290
      check "$SRC/deepfilternet/DeepFilterNet3/checkpoints/model_120.ckpt.best" 23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003
      ;;
    smule)  # MIT code and weights
      venv smule 3.10
      clone smule-renaissance "$GH/smulelabs/smule-renaissance" b80e3a69c347f1290e62dddcb8e069654825aa7b
      fetch "$SRC/smule-renaissance/smule-renaissance-small.pt" \
        "$HF/smulelabs/Smule-Renaissance-Small/resolve/12024e3250ace99e566662a2963e6158d104e69c/smule-renaissance-small.pt" \
        dd73b487ed058fdea66a25915f9f5a50b3d2dd97c03480f268a1d50a00fe2b06
      ;;
    *) die "unknown system $s" ;;
  esac
done
echo "setup done -> $VENVS, $SRC"
