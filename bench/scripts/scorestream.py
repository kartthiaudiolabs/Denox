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
"""The streaming probe: whether a system can run as audio arrives and hold up.

Three steps: cut the probe inputs, render them through every system (each in
its own venv, our model through runsystem.py), score the renders::

    python bench/scripts/scorestream.py --probe
    PROBE_IN=<probe>/in BENCH_OUT=<probe>/out \
        bench/scripts/runfield.sh <systems>
    BENCH_OUT=<probe>/out BENCH_RESULTS=<probe>/results \
        python bench/scripts/runsystem.py --name voice_isolator_v0 \
        --model ... --inputs <probe>/in
    python bench/scripts/scorestream.py [--systems a,b]   # <system>.stream.json

The leaderboard's ``streaming probe`` column says yes or no from measurements
made on renders rather than on claims, so a model that is streaming by design
but is shipped through an offline path reads what it ships. The probe takes
every paired long item of the set (``long@+4``, 45 s with several talkers and a
changing ambience) and cuts from each a ``tail``: the fifteen seconds from 28.8
s to 43.8 s, preceded by 3.2 s of the real audio before them as a lead-in, and
three ``future`` inputs, the same tail whose audio from a splice point on is
another recording. The cut starts on a 3.2 s grid, a multiple of every common
frame hop at 48 kHz, so the fresh render's frames fall where the whole render's
did; a system that frames otherwise (another internal rate, windows placed from
the input's start) differs by that placement too, and the drift includes it. The
lead-in lets a streaming model's state warm up before the measured stretch
begins. The splice points are not round numbers, so a processor that works on
independent blocks of some common length cannot be lucky on every splice.

* **Causal with a bounded look-ahead.** The outputs on ``tail`` and on a
  ``future`` are identical up to the splice minus the system's look-ahead: the
  first 10 ms block whose difference rises above the floor (``FLOOR_DB`` under
  the tail output's mean power) is where the future began to matter. The
  look-ahead is the time from that block to the splice, read on every splice and
  taken at its worst per item; an offline system reads the whole stretch before
  the splice and is reported as reaching that boundary. A system is
  streaming-compatible when every item's look-ahead is at most
  ``LOOK_AHEAD_MAX_MS``.
* **No drift over a long input.** The published render of the whole 45 s item
  and the render of the cut are two outputs of the same fifteen seconds of
  audio, once after 28.8 s of context and once after 3.2 s; the SI-SDR of each
  against the reference is compared on the stretch. A model that carries state
  should give the same answer both times. The drift is the first minus the
  second, negative when the long context hurt; a system holds up when the mean
  is at least ``DRIFT_MEAN_DB``, no item is under ``DRIFT_MIN_DB``, and no
  second of speech is dropped: a second where the reference speaks, the whole
  render sits ``DROP_DB`` under it and the fresh render keeps it within
  ``KEEP_DB``. Reported beside these, not gated: the cold start, how the first
  second of the cut compares with the same second inside the whole render (a
  streaming model's fresh state, an offline model's edge of input); and, for a
  release whose inference path depends on the input's length
  (``path_switch_seconds`` in the wrapper's meta), that the whole item and the
  cut took different paths, which the drift then includes.

The verdict is yes when both hold on the complete inventory of long items, no
otherwise, and the measurements are reported for every probed system so a no
says which failed and by how much. What the probe reads is bound to what it was
cut from and what the board scored: the probe inputs are identified by their
samples (``probe.json``; a float WAV's bytes also carry the time it was written)
and summarised in one identity that the receipts of every probe render record
(``bench/wrappers/receipt.py``, for a wrapper and for ``runsystem.py --inputs``
alike); those receipts must name one environment, the one the board's stamp
records; every output must be exactly its input's length and finite; and the
whole render must hash to the identity in the system's scoring records. The
sidecar carries the rule version, the scorer digest and the output hashes it
read, and the leaderboard shows no verdict whose rule, scorer or render has
since changed. Output: ``<system>.stream.json`` beside the other sidecars,
whole-set only. This is a screening probe: it establishes a bounded dependence
on future input and stable state, not incremental delivery or real-time
throughput.
"""

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import observations

sys.path.insert(0, os.path.join(common.BENCH_DIR, "wrappers"))
import receipt

# scorebench supplies the sample hash the whole render is matched by; it is
# imported where used
SCORER_SHA256 = observations.implementation(
    __file__,
    common.__file__,
    observations.__file__,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "scorebench.py"),
)
# the geometry of the probe inputs; a change re-cuts the probe and its renders
PROBE_VERSION = "3"
# how renders are read into a verdict; a change re-reads the renders only
RULE_VERSION = "5"
SOURCE_CONDITION = "long@+4"
# 153,600 samples, a multiple of every common frame hop (200, 240, 480, 512,
GRID_SECONDS = 3.2
# 1024, 1920, 2048 samples): a cut that starts on this grid puts its frames
# where the whole render's frames were, so at 48 kHz the two renders differ by
# context and not by frame placement nine grids: the measured stretch of every
# long item begins here
TAIL_START = 28.8
# the real preceding audio before the stretch, so the fresh render is warm
LEAD_IN = GRID_SECONDS
TAIL_SECONDS = 15.0  # the measured stretch
# seconds into the stretch where a future input changes
SPLICES = (9.3137, 10.0, 10.6809)
# the cold start is read on the first second of the cut
COLD_START_SECONDS = 1.0
BLOCK_MS = 10
# a block differs when its difference exceeds this, relative to the tail output
FLOOR_DB = -60.0
# streaming-compatible: every item's look-ahead at most this
LOOK_AHEAD_MAX_MS = 100.0
# holds up over a long input: mean SI-SDR delta at least this
DRIFT_MEAN_DB = -1.0
DRIFT_MIN_DB = -3.0  # and no item under this
# a second of the stretch holds speech when within this of its loudest second
SPEECH_DB = 25.0
# and is dropped when the whole render sits this far under the reference there
DROP_DB = 20.0
# while the fresh render keeps it, within this of the reference: speech the long
KEEP_DB = 3.0
# context cost, not speech the system loses either way
MIN_ITEMS = 8  # a set with fewer long items has no probe


def probe_dir(path: str = "") -> str:
    """Returns the probe's directory.

    The flag, ``BENCH_PROBE``, or ``probe_stream`` beside ``BENCH_OUT``.
    """
    return (path or os.environ.get("BENCH_PROBE") or os.path.join(
        os.path.dirname(os.path.abspath(common.OUT_ROOT.rstrip("/"))),
        "probe_stream",
    ))


def rule_sentence() -> str:
    """One sentence saying what the streaming probe column reads."""
    return (
        "the streaming probe is yes when the output depends on at most"
        f" {LOOK_AHEAD_MAX_MS:.0f} ms of future input and {TAIL_SECONDS:.0f} s"
        f" of a 45 s item score within {-DRIFT_MEAN_DB:.0f} dB SI-SDR (no item"
        f" under {-DRIFT_MIN_DB:.0f} dB, and no second of speech lost that a"
        " fresh start keeps) of the same stretch processed after a"
        f" {LEAD_IN:.1f} s lead-in")


def source_rows(rows: list) -> list:
    """Returns the paired long items the probe is cut from, in id order.

    The complete inventory.
    """
    return sorted(
        (r for r in rows
         if r["condition"] == SOURCE_CONDITION and r.get("clean_path")),
        key=lambda r: r["id"],
    )


def stem(item_id: str) -> str:
    return item_id[:-4] if item_id.endswith(".wav") else item_id


def cut_bounds() -> tuple:
    """Returns the cut's first sample, the stretch's first sample and the last.

    The same for every item: the cut starts a lead-in before the stretch, on the
    frame grid.
    """
    a = int(round((TAIL_START - LEAD_IN) * common.SR))
    k = int(round(TAIL_START * common.SR))
    return a, k, k + int(round(TAIL_SECONDS * common.SR))


def sample_sha256(path: str) -> str:
    """Returns the identity of a probe input: its decoded samples.

    As the scorer identifies every output.
    """
    # pylint: disable-next=import-outside-toplevel
    import scorebench

    return scorebench.sample_hash(common.load48(path))


def identity_of(hashes) -> str:
    """Returns one identity for a set of probe inputs.

    The sha256 of their sorted sample hashes, which runfield.sh computes from
    probe.json and runsystem.py from the files themselves.
    """
    return hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest()


def probe_hashes(probe: dict) -> list:
    return [
        h for e in probe["items"]
        for h in [e["tail_sha256"]] + [f["sha256"] for f in e["futures"]]
    ]


def cut(mix: np.ndarray, other: np.ndarray, a: int, b: int) -> tuple:
    """Returns (tail, [future per splice]) of one item.

    The stretch [a, b) of its mixture, and the same stretch whose audio from
    each splice on is the start of another item's stretch.
    """
    tail = mix[a:b].astype(np.float32)
    futures = []
    for splice in SPLICES:
        # a splice sits LEAD_IN + splice into the cut
        k = int(round((LEAD_IN + splice) * common.SR))
        future = tail.copy()
        future[k:] = other[a:a + len(tail) - k]
        futures.append(future)
    return tail, futures


def probe_problem(root: str, rows: list) -> str | None:
    """Returns why ``<root>/probe.json`` is not this build's probe, or None.

    It must match this build, rule and inventory, with every input verified by
    its samples.
    """
    path = os.path.join(root, "probe.json")
    if not os.path.isfile(path):
        return "no probe.json"
    with open(path, encoding="utf-8") as fh:
        probe = json.load(fh)
    if probe.get("set_id") != common.set_id():
        return (
            f"cut from set {str(probe.get('set_id'))[:12]}, the current set is"
            f" {str(common.set_id())[:12]}")
    if probe.get("probe_version") != PROBE_VERSION:
        return (
            "cut by probe version"
            f" {probe.get('probe_version', probe.get('rule_version'))!r}, this"
            f" scorer cuts version {PROBE_VERSION!r}")
    geometry = {
        "cut_seconds": cut_bounds()[0] / common.SR,
        "lead_in_seconds": LEAD_IN,
        "stretch_seconds": TAIL_SECONDS,
        "splices_seconds": [LEAD_IN + x for x in SPLICES],
    }
    differ = [k for k, v in geometry.items() if probe.get(k) != v]
    if differ:
        return ("its recorded geometry differs from this scorer's"
                f" ({', '.join(differ)})")
    wanted = [r["id"] for r in source_rows(rows)]
    if [e["id"] for e in probe.get("items", [])] != wanted:
        return (
            "its items are not the complete inventory of the set's long items")
    if probe.get("identity") != identity_of(probe_hashes(probe)):
        return "its identity does not match its inputs"
    listed = {e["tail"] for e in probe["items"]
             } | {f["name"] for e in probe["items"] for f in e["futures"]}
    stray = (sorted(set(os.listdir(os.path.join(root, "in"))) -
                    listed) if os.path.isdir(os.path.join(root, "in")) else [])
    if stray:  # runsystem.py --inputs identifies the directory as it stands
        return (f"{len(stray)} files under in/ are not probe inputs (first:"
                f" {stray[0]})")
    for entry in probe["items"]:
        for name, sha in [(entry["tail"], entry["tail_sha256"])] + [
            (f["name"], f["sha256"]) for f in entry["futures"]
        ]:
            file = os.path.join(root, "in", name)
            if not os.path.isfile(file) or sample_sha256(file) != sha:
                return f"{name} is not the recorded probe input"
    return None


def write_probe(rows: list, root: str) -> dict:
    """Cuts the probe inputs under ``<root>/in`` and records them.

    The record is ``<root>/probe.json``; a probe that is already current is left
    as it is, so renders made from it stay valid.
    """
    items = source_rows(rows)
    if len(items) < MIN_ITEMS:
        raise SystemExit(
            f"{len(items)} paired items of {SOURCE_CONDITION}: the probe needs "
            f"{MIN_ITEMS}")
    if probe_problem(root, rows) is None:
        with open(os.path.join(root, "probe.json"), encoding="utf-8") as fh:
            return json.load(fh)
    in_dir = os.path.join(root, "in")
    os.makedirs(in_dir, exist_ok=True)
    # the directory is exactly the probe, nothing else
    for stale in os.listdir(in_dir):
        os.remove(os.path.join(in_dir, stale))
    mixes = [common.load48(r["mix_path"]).numpy() for r in items]
    a, k, b = cut_bounds()
    short = [r["id"] for r, m in zip(items, mixes) if len(m) < b]
    if short:
        raise SystemExit(
            f"{len(short)} items of {SOURCE_CONDITION} are shorter than "
            f"{b / common.SR:.2f} s (first: {short[0]})")
    listed = []
    for i, (r, mix) in enumerate(zip(items, mixes)):
        tail, futures = cut(mix, mixes[(i + 1) % len(mixes)], a, b)
        name = f"{stem(r['id'])}__tail.wav"
        sf.write(os.path.join(in_dir, name), tail, common.SR, subtype="FLOAT")
        entry = {
            "id": r["id"],
            "cut_seconds": a / common.SR,
            "stretch_seconds": k / common.SR,
            "tail": name,
            "tail_sha256": sample_sha256(os.path.join(in_dir, name)),
            "futures": [],
        }
        for j, (splice, future) in enumerate(zip(SPLICES, futures), 1):
            name = f"{stem(r['id'])}__future{j}.wav"
            sf.write(os.path.join(in_dir, name),
                     future,
                     common.SR,
                     subtype="FLOAT")
            entry["futures"].append({
                "name": name,
                "splice_seconds": LEAD_IN + splice,  # into the cut
                "sha256": sample_sha256(os.path.join(in_dir, name)),
            })
        listed.append(entry)
    probe = {
        "probe_version": PROBE_VERSION,
        "set_id": common.set_id(),
        "source_condition": SOURCE_CONDITION,
        "cut_seconds": a / common.SR,
        "lead_in_seconds": LEAD_IN,
        "stretch_seconds": TAIL_SECONDS,
        "splices_seconds": [LEAD_IN + x for x in SPLICES],
        "items": listed,
    }
    probe["identity"] = identity_of(probe_hashes(probe))
    tmp = os.path.join(root, f"probe.json.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(probe, fh, indent=1)
    os.replace(tmp, os.path.join(root, "probe.json"))
    return probe


def read_probe(root: str, rows: list) -> dict:
    """Returns the probe's manifest once its inputs are verified.

    They must be the recorded files of this build, this rule and the complete
    inventory.
    """
    problem = probe_problem(root, rows)
    if problem:
        raise SystemExit(
            f"{os.path.join(root, 'probe.json')}: {problem}; write the probe"
            " again (scorestream.py --probe) and re-render it")
    with open(os.path.join(root, "probe.json"), encoding="utf-8") as fh:
        return json.load(fh)


def scored_outputs(results: str, name: str) -> dict:
    """Returns {item id: hash of the output samples scored} of a system.

    Read from the system's whole-set records.
    """
    path = os.path.join(results, f"{name}.items.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: score {name} before probing it")
    metadata, records = observations.read(path)
    if (metadata.get("set_id") != common.set_id() or
            metadata.get("subset", "all") != "all"):
        raise SystemExit(
            f"{path}: records of set {str(metadata.get('set_id'))[:8]} subset "
            f"{metadata.get('subset', 'all')!r}, not the whole current set")
    return {r["id"]: (r.get("identity") or {}).get("output") for r in records}


def board_meta(results: str, name: str) -> dict | None:
    path = os.path.join(results, f"{name}.meta.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def environment_problem(name: str, results: str, root: str,
                        probe: dict) -> str | None:
    """Returns why the probe renders of ``name`` do not stand for it, or None.

    They must be the board's system, on this probe.

    Every probe render carries the receipts its processes wrote: they must agree
    on the environment the board's stamp records (``render_environment_sha256``:
    code, weights, packages, device and arguments, the inputs aside) and name
    this probe's inputs.
    """
    board = board_meta(results, name)
    if board is None:
        return "no stamp for the board's render (meta.json missing)"
    if not board.get("render_environment_sha256"):
        return ("the board's stamp records no render environment: re-render the"
                " board")
    docs = receipt.read_all(os.path.join(root, "out", name))
    if not docs:
        return "the probe render carries no receipt"
    environments = {d.get("environment_digest") for d in docs}
    if environments != {board["render_environment_sha256"]}:
        return (
            "the probe was rendered in another environment than the board's"
            " render (receipts"
            f" {', '.join(str(e)[:12] for e in sorted(map(str, environments)))}"
            f" vs stamp {board['render_environment_sha256'][:12]})")
    probes = {
        (d.get("identity", {}).get("inputs") or {}).get("probe") for d in docs
    }
    if probes != {probe["identity"]}:
        return "the probe render was made from other probe inputs: re-render it"
    return None


def sisdr(est: np.ndarray, ref: np.ndarray) -> float:
    g = float((est * ref).sum() / ((ref * ref).sum() + 1e-12))
    target = g * ref
    err = est - target
    return float(10 * np.log10(
        ((target * target).sum() + 1e-12) / ((err * err).sum() + 1e-12)))


def diff_db(x: np.ndarray, y: np.ndarray) -> float:
    """10 log10 |x - y|^2 / |y|^2 over the common length, floored at -100 dB."""
    n = min(len(x), len(y))
    num = float(((x[:n] - y[:n])**2).mean())
    den = float((y[:n]**2).mean())
    return max(-100.0, float(10 * np.log10(num / (den + 1e-14) + 1e-10)))


def look_ahead_ms(alone: np.ndarray, future: np.ndarray,
                  splice_s: float) -> tuple:
    """Returns (ms, capped): how far before the splice the output differs.

    Read from the first block before it whose difference rises above the floor;
    capped when that is the very first block, so the dependence reaches at least
    to the start of the stretch.
    """
    block = common.SR * BLOCK_MS // 1000
    n = min(len(alone), len(future), int(round(splice_s * common.SR)))
    nb = n // block
    a = alone[:nb * block].reshape(nb, block)
    f = future[:nb * block].reshape(nb, block)
    floor = float((alone[:n]**2).mean()) * 10**(FLOOR_DB / 10)
    hits = np.nonzero(((a - f)**2).mean(1) > floor)[0]
    first = int(hits[0]) if len(hits) else nb
    return (nb - first) * BLOCK_MS, first == 0


def probe_output(out_dir: str, name: str, n: int) -> np.ndarray:
    """One probe output, exactly its input's length and finite, or a refusal."""
    path = os.path.join(out_dir, name)
    if not os.path.isfile(path):
        raise SystemExit(
            f"{path} is missing: the probe render must be complete")
    y = common.load48(path).numpy()
    if len(y) != n:
        raise SystemExit(f"{path}: {len(y)} samples for an input of {n}")
    if not np.isfinite(y).all():
        raise SystemExit(f"{path}: not finite")
    return y


def seconds_db(x: np.ndarray) -> np.ndarray:
    """RMS level per whole second, in dB."""
    n = len(x) // common.SR
    return 10 * np.log10((x[:n * common.SR].reshape(n, common.SR)**2).mean(1) +
                         1e-12)


def dropped(
    whole: np.ndarray,
    alone: np.ndarray,
    ref: np.ndarray,
    level_cal_db: float = 0.0,
) -> tuple:
    """Returns (dropped, speech) seconds of the stretch.

    Speech seconds are those where the reference speaks (within SPEECH_DB of its
    loudest second); dropped ones are those where the whole render sits DROP_DB
    under it while the fresh render stays within KEEP_DB of it either way:
    speech the long context cost. Both renders are read with the system's output
    gain undone (``levels.py``), so neither a quieter nor a louder convention
    decides it, and a fresh render made loud by noise is not speech kept.
    """
    r = seconds_db(ref)
    if not r.size:
        return 0, 0
    w = seconds_db(whole) - level_cal_db
    f = seconds_db(alone) - level_cal_db
    speech = r >= r.max() - SPEECH_DB
    lost = speech & (w <= r - DROP_DB) & (np.abs(f - r) <= KEEP_DB)
    return int(lost.sum()), int(speech.sum())


def path_switch(board: dict, whole_seconds: float, cut_seconds: float) -> bool:
    """Returns whether the release takes another inference path for the cut.

    As its wrapper's description declares (``path_switch_seconds`` in
    ``wrappers/meta/<system>.json``: whole-file inference up to that length), or
    the board's stamp when it carries the field.
    """
    desc = receipt.description((board or {}).get("name", "")) if board else {}
    limit = (desc.get("path_switch_seconds") or
             (board or {}).get("path_switch_seconds"))
    return bool(limit) and whole_seconds > float(limit) >= cut_seconds


def measure_item(
    entry: dict,
    row: dict,
    root: str,
    name: str,
    scored: dict,
    board: dict,
    level_cal_db: float = 0.0,
) -> dict:
    """Returns one item's measurements.

    Read after the whole render is verified to be the audio scored and every
    probe output to be complete.
    """
    # pylint: disable-next=import-outside-toplevel
    import scorebench

    whole_all = common.load48(os.path.join(common.OUT_ROOT, name, row["id"]))
    output_sha256 = scorebench.sample_hash(whole_all)
    if not scored.get(row["id"]) or output_sha256 != scored[row["id"]]:
        raise SystemExit(
            f"{name}/{row['id']} is not the audio that was scored: re-score"
            " first")
    a = int(round(entry["cut_seconds"] * common.SR))
    k = int(round(entry["stretch_seconds"] * common.SR)) - a  # into the cut
    n = k + int(round(TAIL_SECONDS * common.SR))
    ref_all = common.load48(row["clean_path"])
    # the delay and polarity the scorer undoes, found once on the whole item and
    # undone in both renders: an aligned output must not read as drift, a lost
    # second or a cold start
    lag, sign = common.best_alignment(whole_all, ref_all)

    def aligned(x):
        lagged = common.apply_lag(torch.as_tensor(x), lag, len(x))
        return (sign * lagged).numpy()

    whole = aligned(whole_all.numpy())[a:a + n]
    ref = ref_all.numpy()[a:a + n]
    out_dir = os.path.join(root, "out", name)
    alone_raw = probe_output(out_dir, entry["tail"], n)
    alone = aligned(alone_raw)
    # the cut's aligned render runs out where the shift left no sample
    end = n - abs(lag)
    looks = [  # two renders of one system share its delay: compared as rendered
        look_ahead_ms(alone_raw, probe_output(out_dir, f["name"], n),
                      f["splice_seconds"]) for f in entry["futures"]
    ]
    r = int(COLD_START_SECONDS * common.SR)
    cold = (sisdr(alone[:r], ref[:r]) - sisdr(whole[:r], ref[:r]) if
            (ref[:r]**2).mean() > 1e-10 else float("nan"))
    lost, speech = dropped(whole[k:end], alone[k:end], ref[k:end], level_cal_db)
    return {
        "id":
            row["id"],
        "output_sha256":
            output_sha256,
        "look_ahead_ms":
            max(ms for ms, _ in looks),
        # at least that much when the splice that gave the maximum reached the
        # start of the cut
        "look_ahead_capped":
            any(capped for ms, capped in looks if ms == max(m for m, _ in looks)
               ),
        "look_ahead_by_splice_ms": [ms for ms, _ in looks],
        "drift_db":
            (sisdr(whole[k:end], ref[k:end]) - sisdr(alone[k:end], ref[k:end])),
        "cold_start_db":
            cold,
        "dropped_seconds":
            lost,
        "speech_seconds":
            speech,
        "context_diff_db":
            diff_db(whole[k:end], alone[k:end]),
        "lag":
            lag,
        "path_switch":
            path_switch(board,
                        len(whole_all) / common.SR, n / common.SR),
    }


def verdict(items: list) -> dict:
    """Returns the streaming verdict from the per-item measurements.

    The reasons come with it.
    """
    la = np.array([it["look_ahead_ms"] for it in items], dtype=float)
    dr = np.array([it["drift_db"] for it in items], dtype=float)
    lost = sum(it.get("dropped_seconds", 0) for it in items)
    speech = sum(it.get("speech_seconds", 0) for it in items)
    drop_items = sum(1 for it in items if it.get("dropped_seconds", 0))
    causal = bool(la.max() <= LOOK_AHEAD_MAX_MS)
    holds = bool(dr.mean() >= DRIFT_MEAN_DB and dr.min() >= DRIFT_MIN_DB and
                 lost == 0)
    capped = any(it.get("look_ahead_capped") for it in items)
    n_capped = sum(1 for it in items if it.get("look_ahead_capped"))
    max_capped = any(
        it.get("look_ahead_capped")
        for it in items
        if it["look_ahead_ms"] == la.max())
    colds = np.array([it.get("cold_start_db", float("nan")) for it in items],
                     dtype=float)
    reasons = []
    if not causal:
        reasons.append(
            "the output depends on"
            f" {'at least ' if max_capped else ''}{la.max():.0f} ms of future"
            f" input (limit {LOOK_AHEAD_MAX_MS:.0f} ms)")
    if lost:
        reasons.append(
            f"drops speech after a long context: {lost} of {speech} speech"
            f" seconds on {drop_items} of {len(items)} items")
    if dr.mean() < DRIFT_MEAN_DB or dr.min() < DRIFT_MIN_DB:
        reasons.append(
            f"drifts over a long input: {dr.mean():+.1f} dB SI-SDR on average,"
            f" {dr.min():+.1f} dB at worst, on {TAIL_SECONDS:.0f} s of 45 s"
            f" items (limits {DRIFT_MEAN_DB:+.0f} / {DRIFT_MIN_DB:+.0f})")
    return {
        "verdict":
            "yes" if causal and holds else "no",
        "causal":
            causal,
        "holds":
            holds,
        "look_ahead_ms":
            float(np.median(la)),
        "look_ahead_max_ms":
            float(la.max()),
        "look_ahead_capped":
            capped,
        # the median sits on the boundary
        "look_ahead_median_capped":
            n_capped * 2 >= len(items),
        "look_ahead_max_capped":
            max_capped,
        "drift_mean_db":
            float(dr.mean()),
        "drift_min_db":
            float(dr.min()),
        "cold_start_db": (float(np.nanmean(colds))
                          if np.isfinite(colds).any() else float("nan")),
        "dropped_seconds":
            lost,
        "speech_seconds":
            speech,
        "dropout_items":
            drop_items,
        "path_switch":
            any(it.get("path_switch") for it in items),
        "context_diff_db":
            float(np.mean([it["context_diff_db"] for it in items])),
        "n_items":
            len(items),
        "reasons":
            reasons,
    }


def measure(name: str, rows: list, probe: dict, root: str,
            results: str) -> dict:
    """Returns every item of the probe measured through the system.

    The probe's inventory was verified against the set.
    """
    by_id = {r["id"]: r for r in source_rows(rows)}
    scored = scored_outputs(results, name)
    board = board_meta(results, name) or {}
    level_cal_db = base_calibration(results, name)
    items = [
        measure_item(entry, by_id[entry["id"]], root, name, scored, board,
                     level_cal_db) for entry in probe["items"]
    ]
    out = verdict(items)
    out["items"] = items
    out["rule"] = {
        "version": RULE_VERSION,
        "look_ahead_max_ms": LOOK_AHEAD_MAX_MS,
        "drift_mean_db": DRIFT_MEAN_DB,
        "drift_min_db": DRIFT_MIN_DB,
        "speech_db": SPEECH_DB,
        "drop_db": DROP_DB,
        "floor_db": FLOOR_DB,
        "cut_seconds": TAIL_START - LEAD_IN,
        "lead_in_seconds": LEAD_IN,
        "stretch_seconds": TAIL_SECONDS,
        "splices_seconds": list(SPLICES),
        "cold_start_seconds": COLD_START_SECONDS,
        "keep_db": KEEP_DB,
        "probe_version": PROBE_VERSION,
    }
    out["_level_cal_db"] = level_cal_db
    return out


def base_calibration(results: str, name: str) -> float:
    """Returns the system's output gain from its base scores (``levels.py``).

    0 when it has none.
    """
    path = os.path.join(results, f"{name}.metrics.json")
    if not os.path.isfile(path):
        return 0.0
    with open(path, encoding="utf-8") as fh:
        return float(json.load(fh).get("_level_cal_db") or 0.0)


def write(name: str, out: dict, rows: list, probe: dict, results: str) -> str:
    board = board_meta(results, name) or {}
    # the leaderboard ignores a sidecar from another build,
    out["_set_id"] = common.set_id()
    out["_subset"] = "all"  # the probe is cut from the whole set
    out["_rule_version"] = RULE_VERSION  # another rule,
    out["_probe_version"] = PROBE_VERSION
    out["_scorer_sha256"] = SCORER_SHA256  # another implementation,
    out["_probe_identity"] = probe["identity"]
    # or another render
    out["_render_receipt_sha256"] = board.get("render_receipt_sha256")
    out["_outputs_sha256"] = board.get("outputs_sha256")
    out["_sealed"] = common.is_sealed(rows)
    path = os.path.join(results, f"{name}.stream.json")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, path)
    return path


def look_ahead_text(side: dict) -> str:
    """Returns the look-ahead as shown: the median and the maximum over items.

    Each is marked as at least that much when it reached the start of the cut,
    the most the probe can observe.
    """
    capped = side.get("look_ahead_capped")
    med = side.get("look_ahead_median_capped", capped)
    top = side.get("look_ahead_max_capped", capped)
    return (f"{'≥' if med else ''}{side['look_ahead_ms']:.0f} (max"
            f" {'≥' if top else ''}{side['look_ahead_max_ms']:.0f})")


def describe(out: dict) -> str:
    cold = out.get("cold_start_db", float("nan"))
    return (
        f"streaming probe {out['verdict']}: look-ahead {look_ahead_text(out)}"
        f" ms, drift {out['drift_mean_db']:+.1f} dB (min"
        f" {out['drift_min_db']:+.1f}), cold start"
        f" {'n/a' if cold != cold else f'{cold:+.1f} dB'}, dropped"
        f" {out.get('dropped_seconds', 0)}/{out.get('speech_seconds', 0)}"
        f" speech s over {out['n_items']} items" +
        ("; the whole item and the cut take different inference paths"
         if out.get("path_switch") else "") +
        ("; " + "; ".join(out["reasons"]) if out["reasons"] else ""))


def rendered(root: str) -> list:
    return sorted(
        os.path.basename(d)
        for d in glob.glob(os.path.join(root, "out", "*"))
        if os.path.isdir(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--probe",
        action="store_true",
        help="cut the probe inputs (or confirm the current ones) and stop",
    )
    ap.add_argument(
        "--probe-dir",
        default="",
        help="default: BENCH_PROBE or probe_stream beside BENCH_OUT",
    )
    ap.add_argument(
        "--systems",
        default="",
        help="comma-separated; default: every probe render",
    )
    a = ap.parse_args()
    root = probe_dir(a.probe_dir)
    rows = common.manifest(which="all")
    common.guard_private_results(rows)
    if a.probe:
        before = probe_problem(root, rows)
        probe = write_probe(rows, root)
        n = len(probe["items"])
        print(f"probe is current: {n} items x {1 + len(SPLICES)} inputs under"
              f" {root}/in" if before is None else
              (f"{n} items x {1 + len(SPLICES)} probe inputs -> {root}/in"
               f" ({before}); renders of the earlier inputs no longer match:"
               " render the probe again into a fresh folder (runfield.sh with"
               " DISCARD=1, runsystem.py --overwrite)"))
        return
    probe = read_probe(root, rows)
    systems = [s for s in a.systems.split(",") if s] or rendered(root)
    os.makedirs(common.RESULTS, exist_ok=True)
    for name in systems:
        if name in ("INPUT", "REF"):
            continue
        if not os.path.isdir(os.path.join(root, "out", name)):
            print(f"skip {name}: no probe render under {root}/out", flush=True)
            continue
        problem = environment_problem(name, common.RESULTS, root, probe)
        if problem:
            raise SystemExit(f"{name}: {problem}")
        out = measure(name, rows, probe, root, common.RESULTS)
        write(name, out, rows, probe, common.RESULTS)
        print(f"{name:24s} {describe(out)}", flush=True)


if __name__ == "__main__":
    main()
