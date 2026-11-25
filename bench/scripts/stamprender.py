# Copyright 2026 Inworld AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stamps a complete, verified render of one system, or refuses.

    python bench/scripts/stamprender.py <system> --wrapper path/runX.py \\
        --inputs IN --outputs OUT --results RES --set-id ID \\
        --wrapper-exit N --probe-digest D

A render is stamped (``<RES>/<system>.meta.json``: the system's description
plus the identity of the run) only when every check holds:

* the renderer exited zero: a failed run is never stamped, whatever it left;
* every process that rendered left a receipt (``wrappers/receipt.py``), every
  receipt names this system and this renderer, all carry one identity digest,
  and that digest equals the environment and inputs being stamped
  (``--probe-digest``, from ``receipt.py probe`` in the same venv), so outputs
  of two backends, two checkpoints or two set builds are never one system;
* the receipts' inputs are the set build being stamped (``--set-id``);
* the inventory is exact: every input has an output of the same name, a
  48 kHz mono float32 WAV of exactly the input's length with only finite
  samples; stray files are ignored, not counted. Outputs that are entirely
  silent are counted in the stamp, not refused: silence can be a system's
  answer, and the scorer reads it as one.

The stamp records the agreed identity (``render_receipt``), its digest
(``render_receipt_sha256``) and that of its environment without the inputs
(``render_environment_sha256``); the scorer copies them into every record.
It also records ``outputs_sha256``, a digest of every output's bytes: each
score file carries the digest of the render it read, and the board refuses a
score of other audio than the render stamped now.
``runsystem.py`` stamps its own renders through ``stamp`` with its
description and the subset it rendered.
"""

import argparse
import datetime
import glob
import hashlib
import io
import json
import os
import sys

import numpy as np
import soundfile as sf

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(BENCH)
# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.join(BENCH, "wrappers"))
import receipt  # pylint: disable=wrong-import-position

SR = 48000


def check_receipts(out_dir: str, system: str, wrapper: str, set_id: str,
                   probe_digest: str) -> dict:
    """Returns the one identity every receipt in ``out_dir`` agrees on.

    Raises:
        SystemExit: A receipt is missing, retroactive, of another system,
            renderer or set build, or of another environment.
    """
    docs = receipt.read_all(out_dir)
    if not docs:
        raise SystemExit(f"{out_dir}: no render receipt; render it with a "
                         "renderer that calls receipt.begin")
    if any(d.get("retroactive") for d in docs):
        raise SystemExit(f"{out_dir}: a receipt was written after the render; "
                         "only receipts taken as the render began are trusted")
    digests = {d["digest"] for d in docs}
    if len(digests) != 1:
        raise SystemExit(
            f"{out_dir}: {len(docs)} receipts carry {len(digests)} "
            "identities: outputs of different code, weights or "
            "environments were mixed; render again")
    if probe_digest not in digests:
        raise SystemExit(
            f"{out_dir}: the receipts name another environment or "
            "other inputs than the ones being stamped: render again")
    ident = docs[0]["identity"]
    if ident.get("system") != system:
        raise SystemExit(
            f"{out_dir}: rendered as {ident.get('system')!r}, not {system!r}")
    if ident.get("wrapper") != os.path.basename(wrapper):
        raise SystemExit(f"{out_dir}: rendered by {ident.get('wrapper')!r}, "
                         f"not {os.path.basename(wrapper)!r}")
    if (ident.get("inputs") or {}).get("set_id") != set_id:
        raise SystemExit(
            f"{out_dir}: rendered from {ident.get('inputs')}, not set {set_id}")
    return {
        "identity": ident,
        "digest": docs[0]["digest"],
        "environment_digest": receipt.environment_digest(ident),
        "n_receipts": len(docs),
        "first": min(d["time"] for d in docs),
        "last": max(d["time"] for d in docs),
    }


def digest_of(hashes: dict) -> str:
    """Returns the outputs digest of {output name: sha256 of its bytes}.

    The sha256 of one line per output, its name and the sha256 of its bytes,
    in name order: what a stamp vouches for, and what a scorer recomputes from
    the bytes it read.
    """
    lines = [f"{name} {hashes[name]}" for name in sorted(hashes)]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def outputs_digest(files: list, out_dir: str) -> str:
    """Returns the digest of the outputs in ``out_dir`` answering ``files``.

    What ``check_inventory`` computes for a stamp, for a render linked in rather
    than stamped.
    """
    hashes = {}
    for path in files:
        name = os.path.basename(path)
        with open(os.path.join(out_dir, name), "rb") as fh:
            hashes[name] = hashlib.sha256(fh.read()).hexdigest()
    return digest_of(hashes)


def check_inventory(files: list, out_dir: str) -> tuple[int, int, str]:
    """Returns (outputs, silent ones, digest) once every input has a valid one.

    The digest is the sha256 of one line per output, its name and the sha256
    of its bytes, in name order: the exact audio the stamp vouches for.

    Args:
        files: The input files the render must answer.
        out_dir: The render.

    Raises:
        SystemExit: An output is missing, a link (a render writes files; a
            folder linked in is stamped by runsystem.py --outputs, never
            here), of another format or length, or holds NaN or inf.
    """
    if not files:
        raise SystemExit(f"{out_dir}: no inputs to answer")
    missing, bad, silent, hashes = [], [], 0, {}
    for path in sorted(files, key=os.path.basename):
        name = os.path.basename(path)
        out = os.path.join(out_dir, name)
        if not os.path.isfile(out):
            missing.append(name)
            continue
        if os.path.islink(out):
            bad.append(f"{name}: a link to {os.readlink(out)}")
            continue
        with open(out, "rb") as fh:
            data = fh.read()
        hashes[name] = hashlib.sha256(data).hexdigest()
        try:
            info, want = sf.info(io.BytesIO(data)), sf.info(path)
        except RuntimeError as e:  # not a readable sound file
            bad.append(f"{name}: {e}")
            continue
        if (info.samplerate, info.channels, info.subtype) != (SR, 1, "FLOAT"):
            bad.append(f"{name}: {info.samplerate} Hz, {info.channels} ch, "
                       f"{info.subtype}")
        elif info.frames != want.frames:
            bad.append(f"{name}: {info.frames} frames for {want.frames}")
        else:
            samples, _ = sf.read(io.BytesIO(data), dtype="float32")
            if not np.isfinite(samples).all():
                bad.append(f"{name}: NaN or inf")
            elif not samples.any():
                silent += 1
    if missing or bad:
        head = "; ".join(missing[:3] + bad[:3])
        raise SystemExit(
            f"{out_dir}: {len(missing)} outputs missing, {len(bad)} "
            f"invalid (first: {head})")
    return len(files), silent, digest_of(hashes)


def stamp(
    system: str,
    renderer: str,
    files: list,
    out_dir: str,
    results: str,
    set_id: str,
    probe_digest: str,
    desc: dict,
    subset: str = "all",
) -> dict:
    """Writes ``<results>/<system>.meta.json`` for a verified render.

    Args:
        system: The system's name.
        renderer: The script that rendered it.
        files: The inputs it had to answer.
        out_dir: The render.
        results: Where the stamp goes.
        set_id: The set build the inputs belong to.
        probe_digest: The identity digest the environment gives now.
        desc: The system's description, the stamp's first fields.
        subset: The part of the set the files are (``BENCH_SUBSET``).

    Returns:
        The stamp.
    """
    receipts = check_receipts(out_dir, system, renderer, set_id, probe_digest)
    now = datetime.datetime.now(datetime.timezone.utc)
    n, silent, outputs = check_inventory(files, out_dir)
    meta = dict(desc)
    meta.update(
        name=system,
        wrapper=os.path.relpath(os.path.abspath(renderer), REPO),
        wrapper_sha256=receipts["identity"]["wrapper_sha256"],
        set_id=set_id,
        subset=subset,
        n_files=n,
        n_silent=silent,
        outputs_sha256=outputs,
        rendered_at=receipts["last"],
        stamped_at=now.strftime("%Y-%m-%dT%H:%MZ"),
        render_receipt=receipts,
        render_receipt_sha256=receipts["digest"],
        render_environment_sha256=receipts["environment_digest"],
    )
    meta.setdefault("display", system)
    os.makedirs(results, exist_ok=True)
    dst = os.path.join(results, f"{system}.meta.json")
    tmp = f"{dst}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(tmp, dst)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("--wrapper", required=True, help="the renderer's path")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--set-id", required=True)
    ap.add_argument("--wrapper-exit", type=int, required=True)
    ap.add_argument("--probe-digest", required=True)
    a = ap.parse_args()
    if a.wrapper_exit != 0:
        raise SystemExit(f"{a.system}: the renderer exited {a.wrapper_exit}; "
                         "a failed run is not stamped")
    files = glob.glob(os.path.join(a.inputs, "*.wav"))
    meta = stamp(
        a.system,
        a.wrapper,
        files,
        a.outputs,
        a.results,
        a.set_id,
        a.probe_digest,
        receipt.description(a.system),
    )
    print(f"  stamped {a.system}: {meta['n_files']} files "
          f"({meta['n_silent']} silent), "
          f"{meta['render_receipt']['n_receipts']} receipt(s), identity "
          f"{meta['render_receipt_sha256'][:12]}")


if __name__ == "__main__":
    main()
